package thunder.streaming

import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.util.Vector

import thunder.util.Load
import thunder.util.Load.DataPreProcessor
import thunder.regression.LinearRegressionModel
import org.apache.spark.Logging


/**
 * Stateful linear regression on streaming data
 *
 * The underlying model is that every batch of data
 * contains one or more features, and keyed data points
 * ("labels") each of which are predicted by those features through
 * a potentially different linear model. We collect the features and
 * labels over time through concatenation, and continually
 * estimate a regression fit for each key.
 * Returns a state stream with the
 * parameters of the regression for every key.
 *
 * Features and labels from different batches
 * can have variable length
 *
 * See also: StreamingLinearRegression
 */
class StatefulLinearRegression (
  var inds: Array[Int],
  var preProcessMethod: String)
  extends Serializable with Logging
{

  def this() = this(Array(0), "raw")

  /** Set the preprocessing method. Default: raw (no preprocessing) */
  def setPreProcessMethod(preProcessMethod: String): StatefulLinearRegression = {
    this.preProcessMethod = preProcessMethod
    this
  }

  /** Set which indices that correspond to labels. Default: Array(0). */
  def setInds(inds: Array[Int]): StatefulLinearRegression = {
    this.inds = inds
    this
  }

  /** State updating function that concatenates arrays. */
  def concatenate(values: Seq[Array[Double]], state: Option[Array[Double]]) = {
    val previousState = state.getOrElse(Array.empty[Double])
    Some(Array.concat(previousState, values.flatten.toArray))
  }

  /** Compute regression on each data point by first selecting
    * the features, and then regressing all data points against them.
    *
    * @param rdd RDD of data points as Int, Array[Double] pairs
    * @return RDD of regression statistics as Int, Array[Double] pairs
    * */
  def regress(rdd: RDD[(Int, Array[Double])]): RDD[(Int, Array[Double])] = {
    if (rdd.count() > 0) {
      val features = rdd.filter(x => inds.contains(x._1)).values.collect()
      val model = new LinearRegressionModel(features)
      val data = rdd.filter{case (k, v) => (k != 0) & (v.length == model.n)}
      data.mapValues(model.fit)
    } else {
      rdd
    }
  }

  def runStreaming(data: DStream[(Int, Array[Double])]): DStream[(Int, Array[Double])] = {
    if (preProcessMethod != "raw") {
      data.updateStateByKey(concatenate).transform(regress _)
    } else {
      val PreProcessor = new DataPreProcessor(preProcessMethod)
      data.updateStateByKey(concatenate).mapValues(PreProcessor.get).transform(regress _)
    }
  }

}

/**
 * Top-level methods for calling Stateful Linear Regression.
 */
object StatefulLinearRegression {

  /**
   * Train a Stateful Linear Regression model.
   * We assume that in each batch of streaming data we receive
   * a single vector of features and several vectors of labels.
   * We are fitting a separate model to the labels associated with
   * each key.
   * @param input DStream of (Int, Array[Double]) keyed data point
   * @param preProcessMethod How to pre process data
   * @param inds List of keys associated with features (all others are labels)
   * @return DStream of (Int, Array[Double]) with fitted regression models
   */
  def trainStreaming(input: DStream[(Int, Array[Double])],
            preProcessMethod: String,
            inds: Array[Int]): DStream[(Int, Array[Double])] =
  {
    new StatefulLinearRegression().setInds(inds).runStreaming(input)
  }

  def main(args: Array[String]) {
    if (args.length != 4) {
      System.err.println("Usage: StatefulLinearRegression <master> <directory> <preProcessMethod> <batchTime>")
      System.exit(1)
    }

    val (master, directory, preProcessMethod, batchTime) = (
      args(0), args(1), args(2), args(3).toLong)

    // create streaming context
    val ssc = new StreamingContext(master, "StreamingKMeans", Seconds(batchTime))
    ssc.checkpoint(System.getenv("CHECKPOINT"))

    // main streaming operations
    val data = Load.loadStreamingDataWithKeys(ssc, directory, 1)

    // train linear regression models
    val state = StatefulLinearRegression.trainStreaming(data, preProcessMethod, Array(0))
    state.mapValues(x => Vector(x)).print()

    ssc.start()
  }

}

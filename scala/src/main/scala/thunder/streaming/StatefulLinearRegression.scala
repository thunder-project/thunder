package thunder.streaming

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.streaming.dstream.DStream

import thunder.util.Load
import thunder.util.Load.DataPreProcessor
import thunder.regression.{SharedLinearRegressionModel, LinearRegressionModel}
import org.apache.spark.Logging


/**
 * Stateful linear regression on streaming data
 *
 * The underlying model is that every batch of streaming
 * data contains some records as features and others as labels
 * (each with unique keys), and each set of labels can be predicted
 * as a linear function of the common features. We collected
 * the features and labels over time through concatenation,
 * and estimate a Linear Regression Model for each key.
 * Returns a state stream with the Linear Regression Models.
 *
 * Features and labels from different batches
 * can have different lengths.
 *
 * See also: StreamingLinearRegression
 */
class StatefulLinearRegression (
  var featureKeys: Array[Int],
  var preProcessMethod: String)
  extends Serializable with Logging
{

  def this() = this(Array(0), "raw")

  /** Set the pre processing method. Default: raw (no pre processing) */
  def setPreProcessMethod(preProcessMethod: String): StatefulLinearRegression = {
    this.preProcessMethod = preProcessMethod
    this
  }

  /** Set which indices that correspond to features. Default: Array(0). */
  def setFeatureKeys(featureKeys: Array[Int]): StatefulLinearRegression = {
    this.featureKeys = featureKeys
    this
  }

  /** State updating function that concatenates arrays. */
  def concatenate(values: Seq[Array[Double]], state: Option[Array[Double]]) = {
    val previousState = state.getOrElse(Array.empty[Double])
    Some(Array.concat(previousState, values.flatten.toArray))
  }

  /** Compute a Linear Regression Model for each data point by first selecting
    * the features, and then regressing all data points against them.
    * Features are the subset of data points with the specified keys.
    *
    * @param rdd RDD of data points as (Int, Array[Double]) pairs
    * @return RDD of fitted models as (Int, LinearRegressionModel) pairs
    * */
  def update(rdd: RDD[(Int, Array[Double])]): RDD[(Int, LinearRegressionModel)] = {
    val features = rdd.filter(x => featureKeys.contains(x._1)).values.collect()
    val model = new SharedLinearRegressionModel(features)
    val data = rdd.filter{case (k, v) => (!featureKeys.contains(k)) & (v.length == model.n)}
    data.mapValues(model.fit)
  }

  def runStreaming(data: DStream[(Int, Array[Double])]): DStream[(Int, LinearRegressionModel)] = {
    if (preProcessMethod != "raw") {
      data.updateStateByKey(concatenate).transform(update _)
    } else {
      val PreProcessor = new DataPreProcessor(preProcessMethod)
      data.updateStateByKey(concatenate).mapValues(PreProcessor.get).transform(update _)
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
   * one or more features and several vectors of labels, each
   * with a unique key, and a subset of keys indicate the features.
   * We fit separate linear models that relate the common features
   * to the labels associated with each key.
   *
   * @param input DStream of (Int, Array[Double]) keyed data point
   * @param preProcessMethod How to pre process data
   * @param featureKeys Array of keys associated with features
   * @return DStream of (Int, LinearRegressionModel) with fitted regression models
   */
  def trainStreaming(input: DStream[(Int, Array[Double])],
            preProcessMethod: String,
            featureKeys: Array[Int]): DStream[(Int, LinearRegressionModel)] =
  {
    new StatefulLinearRegression().setFeatureKeys(featureKeys).runStreaming(input)
  }

  def main(args: Array[String]) {
    if (args.length != 5) {
      System.err.println("Usage: StatefulLinearRegression <master> <directory> <preProcessMethod> <batchTime> <featureKeys>")
      System.exit(1)
    }

    val (master, directory, preProcessMethod, batchTime, featureKeys) = (
      args(0), args(1), args(2), args(3).toLong, args(4).drop(1).dropRight(1).split(",").map(_.trim.toInt))

    val conf = new SparkConf().setMaster(master).setAppName("StatefulLinearRegression")

    if (!master.contains("local")) {
      conf.setSparkHome(System.getenv("SPARK_HOME"))
          .setJars(List("target/scala-2.10/thunder_2.10-0.1.0.jar"))
          .set("spark.executor.memory", "100G")
    }

    /** Create Streaming Context */
    val ssc = new StreamingContext(conf, Seconds(batchTime))
    ssc.checkpoint(System.getenv("CHECKPOINT"))

    /** Load streaming data */
    val data = Load.loadStreamingDataWithKeys(ssc, directory)

    /** Train Linear Regression models */
    val state = StatefulLinearRegression.trainStreaming(data, preProcessMethod, featureKeys)

    /** Print results (for testing) */
    state.mapValues(x => "\n" + "weights: " + x.weights.mkString(",") + "\n" +
                         "intercept: " + x.intercept.toString + "\n" +
                         "r2: " + x.r2.toString + "\n").print()

    ssc.start()
  }

}

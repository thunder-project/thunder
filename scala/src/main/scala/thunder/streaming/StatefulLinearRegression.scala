package thunder.streaming

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.streaming.dstream.DStream

import thunder.util.Load
import thunder.util.Save
import thunder.util.Load.subToInd
import org.apache.spark.Logging
import scala.math.sqrt
import scala.Some
import cern.colt.matrix.DoubleFactory2D
import cern.colt.matrix.DoubleFactory1D
import cern.colt.matrix.DoubleMatrix2D
import cern.colt.matrix.DoubleMatrix1D
import cern.jet.math.Functions.{plus, minus, bindArg2, pow}
import cern.colt.matrix.linalg.Algebra.DEFAULT.{inverse, mult, transpose}


/** Class for representing parameters and sufficient statistics for a fitted linear regression model */
class FittedModel(
   var count: Double,
   var mean: Double,
   var sumOfSquaresTotal: Double,
   var sumOfSquaresError: Double,
   var Xy: DoubleMatrix1D,
   var beta: DoubleMatrix1D) extends Serializable {

  def variance = sumOfSquaresTotal / (count - 1)

  def std = sqrt(variance)

  def R2 = 1 - sumOfSquaresError / sumOfSquaresTotal

}

/** Class for representing and updating sufficient statistics of a shared linear regression model */
class SharedModel(var X: Option[DoubleMatrix2D], var XX: Option[DoubleMatrix2D], var d: Option[Int]) extends Serializable {

  def create(features: Array[Array[Double]]): SharedModel = {
    val d = features.size // features
    val n = features(0).size // data points

    val X = DoubleFactory2D.dense.make(n, d + 1)
    for (i <- 0 until n) {
      X.set(i, 0, 1)
    }
    for (i <- 0 until n ; j <- 1 until d + 1) {
      X.set(i, j, features(j - 1)(i))
    }

    val XX = mult(transpose(X), X)

    this.X = Some(X)
    this.XX = Some(XX)
    this.d = Some(d)
    this
  }

  def update(features: Array[Array[Double]]) {
    if (XX.isEmpty) {
      val newModel = create(features)
      X = newModel.X
      XX = newModel.XX
    } else {
      val newModel = create(features)
      X = newModel.X
      XX = Some(XX.get.assign(newModel.XX.get, plus))
    }
  }

}

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

  val runningLinearRegression = (values: Seq[Array[Double]], state: Option[FittedModel], model: SharedModel) => {
    val updatedState = state.getOrElse(new FittedModel(0.0, 0.0, 0.0, 0.0,
      DoubleFactory1D.dense.make(model.d.get + 1), DoubleFactory1D.dense.make(model.d.get + 1)))
    val y = values.flatten
    val currentCount = y.size

    if (currentCount != 0) {

      // create matrix version of y
      val ymat = DoubleFactory1D.dense.make(currentCount)
      for (i <- 0 until currentCount) {
        ymat.set(i, y(i))
      }

      // store values from previous iteration (needed for update equations)
      val oldCount = updatedState.count
      val oldMean = updatedState.mean
      val oldXy = updatedState.Xy.copy
      val oldXX = model.XX.get.copy.assign(mult(transpose(model.X.get), model.X.get), minus)
      val oldBeta = updatedState.beta

      // compute current estimates for sufficient statistics
      val currentMean = y.foldLeft(0.0)(_+_) / currentCount
      val currentSumOfSquaresTotal = y.map(x => pow(x - currentMean, 2)).foldLeft(0.0)(_+_)
      val currentXy = mult(transpose(model.X.get), ymat)

      // compute new values for statistics (needed for update equations)
      val newXy = updatedState.Xy.copy.assign(currentXy, plus)
      val newBeta = mult(inverse(model.XX.get), newXy)

      // compute terms for update equations
      val delta = currentMean - oldMean
      val term1 = ymat.copy.assign(mult(model.X.get, newBeta), minus).assign(bindArg2(pow, 2)).zSum
      val term2 = mult(mult(oldXX, newBeta), newBeta)
      val term3 = mult(mult(oldXX, oldBeta), oldBeta)
      val term4 = 2 * mult(oldBeta.copy.assign(newBeta, minus), oldXy)

      // update the components of the fitted model
      updatedState.count += currentCount
      updatedState.mean += (delta * currentCount / (oldCount + currentCount))
      updatedState.Xy = newXy
      updatedState.beta = newBeta
      updatedState.sumOfSquaresTotal += currentSumOfSquaresTotal + delta * delta * (oldCount * currentCount) / (oldCount + currentCount)
      updatedState.sumOfSquaresError += term1 + term2 - term3 + term4

    }

    Some(updatedState)
  }


  def runStreaming(data: DStream[(Int, Array[Double])]): DStream[(Int, FittedModel)] = {
    val model = new SharedModel(None, None, None)
    data.foreachRDD{rdd => val features = rdd.filter(x => featureKeys.contains(x._1))
                                             .groupByKey().values.map(x => x.toArray.flatten)
                                             .collect()
                            if (features.size != 0) {
                              model.update(features)
                            }}
    data.filter{case (k, v) => !featureKeys.contains(k)}
        .updateStateByKey{ case (x,y) => runningLinearRegression(x,y, model)}
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
            featureKeys: Array[Int]): DStream[(Int, FittedModel)] =
  {
    new StatefulLinearRegression().setFeatureKeys(featureKeys).runStreaming(input)
  }

  def main(args: Array[String]) {
    if (args.length != 7) {
      System.err.println(
        "Usage: StatefulLinearRegression <master> <directory> <preProcessMethod> <batchTime> <outputDirectory> <dims> <featureKeys>")
      System.exit(1)
    }

    val (master, directory, preProcessMethod, batchTime, outputDirectory, dims, features) = (
      args(0), args(1), args(2), args(3).toLong, args(4),
      args(5).drop(1).dropRight(1).split(",").map(_.trim.toInt),
      Array(args(6).drop(1).dropRight(1).split(",").map(_.trim.toInt)))

    val conf = new SparkConf().setMaster(master).setAppName("StatefulLinearRegression")

    if (!master.contains("local")) {
      conf.setSparkHome(System.getenv("SPARK_HOME"))
          .setJars(List("target/scala-2.10/thunder_2.10-0.1.0.jar"))
          .set("spark.executor.memory", "100G")
          .set("spark.default.parallelism", "100")
    }

    /** Get feature keys with linear indexing */
    val featureKeys = subToInd(features, dims)

    /** Create Streaming Context */
    val ssc = new StreamingContext(conf, Seconds(batchTime))
    ssc.checkpoint(System.getenv("CHECKPOINT"))

    /** Load streaming data */
    val data = Load.loadStreamingDataWithKeys(ssc, directory, dims.size, dims)

    /** Train Linear Regression models */
    val state = StatefulLinearRegression.trainStreaming(data, preProcessMethod, featureKeys)

    /** Print results (for testing) */
    state.mapValues(x => "\n" + "mean: " + "%.5f".format(x.mean) +
                         "\n" + "variance: " + "%.5f".format(x.variance) +
                         "\n" + "beta: " + x.beta.toArray.mkString(",") +
                         "\n" + "R2: " + "%.5f".format(x.R2) +
                         "\n" + "Xy: " + x.Xy.toArray.mkString(",") + "\n").print()

    ///** Collect output */
    //val out = state.mapValues(x => Array(x.r2) ++ x.tuning)
    //Save.saveStreamingDataAsText(out, outputDirectory, Seq("r2", "tuning"))

    ssc.start()
  }

}

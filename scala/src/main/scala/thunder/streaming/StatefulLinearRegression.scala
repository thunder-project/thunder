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


/** Class for representing parameters and sufficient statistics for a running linear regression model */
class FittedModel(
   var count: Double,
   var mean: Double,
   var sumOfSquaresTotal: Double,
   var sumOfSquaresError: Double,
   var XX: DoubleMatrix2D,
   var Xy: DoubleMatrix1D,
   var beta: DoubleMatrix1D) extends Serializable {

  def variance = sumOfSquaresTotal / (count - 1)

  def std = sqrt(variance)

  def R2 = 1 - sumOfSquaresError / sumOfSquaresTotal

  def intercept = beta.toArray()(0)

  def weights = beta.toArray.drop(1)

}

/**
 * Stateful linear regression on streaming data
 *
 * The underlying model is that every batch of streaming
 * data contains a set of records with unique keys,
 * a subset are features, and the rest can be predicted
 * as different linear functions of the common features.
 * We estimate the sufficient statistics of the features,
 * and each of the data points, to computing a running
 * estimate of the linear regression model for each key.
 * Returns a state stream of fitted models.
 *
 * Features and labels from different batches
 * can have different lengths.
 *
 * See also: StreamingLinearRegression
 */
class StatefulLinearRegression (
  var featureKeys: Array[Int])
  extends Serializable with Logging
{

  def this() = this(Array(0))

  /** Set which indices that correspond to features. Default: Array(0). */
  def setFeatureKeys(featureKeys: Array[Int]): StatefulLinearRegression = {
    this.featureKeys = featureKeys
    this
  }

  val runningLinearRegression = (input: Seq[Array[Double]], state: Option[FittedModel], features: Array[Double]) => {

    //val tmp = input.foldLeft((Array[Double](), Array[Double]())) { (acc, i) => (acc._1 ++ i._1, acc._2 ++ i._2)}
    val tmp = input.foldLeft(Array[Double]()) { (acc, i) => acc ++ i}
    val values = tmp
    //val features = tmp._2
    val d = 1 // number of features

    val updatedState = state.getOrElse(new FittedModel(0.0, 0.0, 0.0, 0.0, DoubleFactory2D.dense.make(d + 1, d + 1),
      DoubleFactory1D.dense.make(d + 1), DoubleFactory1D.dense.make(d + 1)))

    if ((values.size != 0) & (features.size != 0)) {

      val n = features.size // number of data points

      // append column of 1s
      val X = DoubleFactory2D.dense.make(n, d + 1)
      for (i <- 0 until n) {
        X.set(i, 0, 1)
      }

      // append features
      for (i <- 0 until n ; j <- 1 until d + 1) {
        X.set(i, j, features(i))
      }

      val y = values
      val currentCount = y.size

      // create matrix version of y
      val ymat = DoubleFactory1D.dense.make(currentCount)
      for (i <- 0 until currentCount) {
        ymat.set(i, y(i))
      }

      // store values from previous iteration (needed for update equations)
      val oldCount = updatedState.count
      val oldMean = updatedState.mean
      val oldXy = updatedState.Xy.copy
      val oldXX = updatedState.XX.copy
      val oldBeta = updatedState.beta

      // compute current estimates of all statistics
      val currentMean = y.foldLeft(0.0)(_+_) / currentCount
      val currentSumOfSquaresTotal = y.map(x => pow(x - currentMean, 2)).foldLeft(0.0)(_+_)
      val currentXy = mult(transpose(X), ymat)
      val currentXX = mult(transpose(X), X)

      // compute new values for X*y (the sufficient statistic) and new beta (needed for update equations)
      val newXX = oldXX.copy.assign(currentXX, plus)
      val newXy = updatedState.Xy.copy.assign(currentXy, plus)
      val newBeta = mult(inverse(newXX), newXy)

      // compute terms for update equations
      val delta = currentMean - oldMean
      val term1 = ymat.copy.assign(mult(X, newBeta), minus).assign(bindArg2(pow, 2)).zSum
      val term2 = mult(mult(oldXX, newBeta), newBeta)
      val term3 = mult(mult(oldXX, oldBeta), oldBeta)
      val term4 = 2 * mult(oldBeta.copy.assign(newBeta, minus), oldXy)

      // update the all statistics of the fitted model
      updatedState.count += currentCount
      updatedState.mean += (delta * currentCount / (oldCount + currentCount))
      updatedState.Xy = newXy
      updatedState.XX = newXX
      updatedState.beta = newBeta
      updatedState.sumOfSquaresTotal += currentSumOfSquaresTotal + delta * delta * (oldCount * currentCount) / (oldCount + currentCount)
      updatedState.sumOfSquaresError += term1 + term2 - term3 + term4
    }

    Some(updatedState)
  }


  def runStreaming(data: DStream[(Int, Array[Double])]): DStream[(Int, FittedModel)] = {

    var features = Array[Double]()

    data.filter{case (k, v) => featureKeys.contains(k)}.foreachRDD{rdd =>
        val tmp = rdd.values.collect().flatten
        if (tmp.size != 0) {
          features = tmp
        } else {
          features = Array[Double]()
        }
    }

    data.filter{case (k, v) => !featureKeys.contains(k)}.updateStateByKey{
      case (x, y) => runningLinearRegression(x, y, features)}
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
   * @param featureKeys Array of keys associated with features
   * @return DStream of (Int, LinearRegressionModel) with fitted regression models
   */
  def trainStreaming(input: DStream[(Int, Array[Double])],
            featureKeys: Array[Int]): DStream[(Int, FittedModel)] =
  {
    new StatefulLinearRegression().setFeatureKeys(featureKeys).runStreaming(input)
  }

  def main(args: Array[String]) {
    if (args.length != 6) {
      System.err.println(
        "Usage: StatefulLinearRegression <master> <directory> <batchTime> <outputDirectory> <dims> <featureKeys>")
      System.exit(1)
    }

    val (master, directory, batchTime, outputDirectory, dims, features) = (
      args(0), args(1), args(2).toLong, args(3),
      args(4).drop(1).dropRight(1).split(",").map(_.trim.toInt),
      Array(args(5).drop(1).dropRight(1).split(",").map(_.trim.toInt)))

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
    val state = StatefulLinearRegression.trainStreaming(data, featureKeys)

    /** Print output (for testing) */
    state.mapValues(x => "\n" + "mean: " + "%.5f".format(x.mean) +
                         "\n" + "count: " + "%.5f".format(x.count) +
                         "\n" + "variance: " + "%.5f".format(x.variance) +
                         "\n" + "beta: " + x.beta.toArray.mkString(",") +
                         "\n" + "R2: " + "%.5f".format(x.R2) +
                         "\n" + "SSE: " + "%.5f".format(x.sumOfSquaresError) +
                         "\n" + "SST: " + "%.5f".format(x.sumOfSquaresTotal) +
                         "\n" + "Xy: " + x.Xy.toArray.mkString(",")).print()

    ///** Save output (for production) */
    //val out = state.mapValues(x => Array(x.R2) ++ Array(x.weights))
    //Save.saveStreamingDataAsText(out, outputDirectory, Seq("r2", "weights"))

    ssc.start()
  }

}

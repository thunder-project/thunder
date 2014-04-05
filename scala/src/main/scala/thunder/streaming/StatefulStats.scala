package thunder.streaming

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext._
import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.streaming.dstream.DStream

import thunder.util.Load
import thunder.util.Save
import org.apache.spark.Logging

import math.pow
import math.sqrt


/** Class for representing running statistics */
class StatsModel(var count: Double, var mean: Double, var sumOfSquares: Double) extends Serializable {

  def variance = sumOfSquares / (count - 1)

  def std = sqrt(variance)

}


/**
 * Stateful statistics
 */
class StatefulStats (
  var stat: String)
  extends Serializable with Logging
{

  def this() = this("mean")

  /** Set the statistic */
  def setStatistic(stat: String): StatefulStats = {
    this.stat = stat
    this
  }

  /** State updating function that computes the statistic. */
  val runningStats = (values: Seq[Array[Double]], state: Option[StatsModel]) => {
    val updatedState = state.getOrElse(new StatsModel(0.0, 0.0, 0.0))
    val vec = values.flatten
    val newCount = vec.size

    if (newCount != 0) {
      val oldCount = updatedState.count
      val oldMean = updatedState.mean
      val newMean = vec.foldLeft(0.0)(_+_) / newCount
      val delta = newMean - oldMean
      val ss2 = vec.map(x => pow(x - newMean, 2)).foldLeft(0.0)(_+_)
      updatedState.count += newCount
      updatedState.mean += (delta * newCount / (oldCount + newCount))
      updatedState.sumOfSquares += ss2 + pow(delta, 2) * (oldCount * newCount) / (oldCount + newCount)
    }

    Some(updatedState)
  }

  def runStreaming(data: DStream[(Int, Array[Double])]): DStream[(Int, StatsModel)] = {
      data.updateStateByKey{runningStats}
  }

}

/**
 * Top-level methods for calling Stateful Stats.
 */
object StatefulStats {

  /**
   * Train a Stateful Linear Regression model.
   * We assume that in each batch of streaming data we receive
   * one or more features and several vectors of labels, each
   * with a unique key, and a subset of keys indicate the features.
   * We fit separate linear models that relate the common features
   * to the labels associated with each key.
   *
   * @param input DStream of (Int, Array[Double]) keyed data point
   * @param stat What statistic to compute
   * @return DStream of (Int, Double) with statistic
   */
  def trainStreaming(input: DStream[(Int, Array[Double])],
            stat: String): DStream[(Int, StatsModel)] =
  {
    new StatefulStats().setStatistic(stat).runStreaming(input)
  }

  def main(args: Array[String]) {
    if (args.length != 6) {
      System.err.println(
        "Usage: StatefulStats <master> <directory> <stat> <batchTime> <outputDirectory> <dims>")
      System.exit(1)
    }

    val (master, directory, stat, batchTime, outputDirectory, dims) = (
      args(0), args(1), args(2), args(3).toLong, args(4),
      args(5).drop(1).dropRight(1).split(",").map(_.trim.toInt))

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
    val data = Load.loadStreamingDataWithKeys(ssc, directory, dims.size, dims)
    data.print()

    /** Train Linear Regression models */
    val state = StatefulStats.trainStreaming(data, stat)

    /** Print results (for testing) */
    state.mapValues(x => "\n" + "mean: " + "%.4f".format(x.mean) + "\n" +
                         "\n" + "variance: " + "%.4f".format(x.variance) + "\n").print()

    /** Collect output */
    //val out = state.mapValues(x => Array(x.count, x.mean, x.variance))
    //Save.saveStreamingDataAsText(out, outputDirectory, Seq("count", "mean", "variance"))

    ssc.start()
  }

}

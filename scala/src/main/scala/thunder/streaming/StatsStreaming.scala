package thunder.streaming

import thunder.util.Load

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.streaming._
import StreamingContext._
import org.apache.spark.util.Vector
import scala.math.min

object StatsStreaming {

  def sum(vec1: Vector, vec2: Vector): Vector = {
    val l = min(vec1.elements.length, vec2.elements.length)
    val vec1prime = Vector(vec1.elements.slice(0,l))
    val vec2prime = Vector(vec2.elements.slice(0,l))
    vec1prime + vec2prime
  }

  def getstats(rdd: RDD[(Int, Vector)]): RDD[(Int, Vector)] = {
    if (rdd.count() > 0) {
      val covar = rdd.filter(_._1 == 0).first()._2
      val data = rdd.filter(_._1 != 0)
      data.mapValues(x => sum(x, covar))
    } else {
      rdd
    }
  }

  // update state
  def concatenate(values: Seq[Array[Double]], state: Option[Array[Double]]) = {
    val previousState = state.getOrElse(Array.empty[Double])
    Some(Array.concat(previousState, values.flatten.toArray))
  }

  def main(args: Array[String]) {
    if (args.length < 4) {
      System.err.println("Usage: streamingStats <master> <directory> <batchTime> <windowTime>")
      System.exit(1)
    }

    // create spark context
    System.setProperty("spark.executor.memory", "120g")
    System.setProperty("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val batchTime = args(2).toLong
    val windowTime = args(3).toLong
    val ssc = new StreamingContext(args(0), "SimpleStreaming", Seconds(batchTime),
      System.getenv("SPARK_HOME"), List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))
    ssc.checkpoint(System.getenv("CHECKPOINTSTREAMING"))

    // main streaming operations
    val data = Load.loadStreamingData(ssc, args(1))
    val state = data.updateStateByKey(concatenate).mapValues(x => Vector(x)).transform(rdd => getstats(rdd))

    // for debugging
    state.print()

    ssc.start()
  }
}
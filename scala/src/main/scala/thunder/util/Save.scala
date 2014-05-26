package thunder.util

import thunder.util.io.Keys
import thunder.util.io.{TextWriter, ImageWriter}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._

/** Utilities for saving results from RDDs */

object Save {

  def asText(data: RDD[Array[Double]], directory: String, fileName: Seq[String]) {

    val saver = new TextWriter(directory)
    val n = data.first().size
    for (i <- 0 until n) {
      saver.write(data.map(x => x(i)), fileName(i))
    }

  }

  def asTextWithKeys(data: RDD[(Array[Int], Array[Double])], directory: String, fileName: Seq[String]) {

    val saver = new TextWriter(directory)
    val dims = Keys.getDims(data)
    val sorted = Keys.subToInd(data, dims).sortByKey().values
    val n = sorted.first().size
    for (i <- 0 until n) {
      saver.write(sorted.map(x => x(i)), fileName(i))
    }

  }

}

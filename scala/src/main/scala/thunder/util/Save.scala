/** Utilities for saving results */

package thunder.util

import thunder.util.Load.{getDims, subToInd}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.streaming.dstream.DStream
import java.io.File

object Save {

  case class ImageSaver(directory: String) {

    var colorMode = new String

    def setColorMode(colorMode: String): ImageSaver = {
      this.colorMode = colorMode
      this
    }

    def write(rdd: RDD[(Array[Int], Double)]) = {
      rdd.collect()

    }

  }

  case class TextSaver(directory: String) {

    def write(rdd: RDD[Double], fileName: String) = {
      val out = rdd.collect()
      printToFile(new File(directory ++ File.separator ++ fileName ++ ".txt"))(p => {
        out.foreach(x => p.println(x))
      })
    }

  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try {
      op(p)
    } finally {
      p.close()
    }
  }

  def saveDataAsText(data: RDD[Array[Double]], directory: String, fileName: Seq[String]) {

    val saver = new TextSaver(directory)
    val nout = data.first().size
    for (i <- 0 until nout) {
      saver.write(data.map(x => x(i)), fileName(i))
    }

  }

  def saveDataWithKeysAsText(data: RDD[(Array[Int], Array[Double])], directory: String, fileName: Seq[String]) {

    val saver = new TextSaver(directory)
    val dims = getDims(data)
    val sorted = subToInd(data, dims).sortByKey().values
    val nout = sorted.first().size
    for (i <- 0 until nout) {
      saver.write(sorted.map(x => x(i)), fileName(i))
    }

  }

  def saveStreamingDataAsText(data: DStream[(Int, Array[Double])], directory: String, fileName: Seq[String]) {

    val saver = new TextSaver(directory)
    data.foreachRDD{rdd =>
      val sorted = rdd.sortByKey().values
      val nout = sorted.first().size
      for (i <- 0 until nout) {
        saver.write(sorted.map(x => x(i)), fileName(i))
      }
    }

  }

}

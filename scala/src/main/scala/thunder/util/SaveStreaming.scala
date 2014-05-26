package thunder.util

import thunder.util.io.TextWriter
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.SparkContext._

/** Utilities for saving streaming results from DStreams */

object SaveStreaming {

  def asTextWithKeys(data: DStream[(Int, Array[Double])], directory: String, fileName: Seq[String]) {

    val saver = new TextWriter(directory)
    data.foreachRDD{rdd =>
      val sorted = rdd.sortByKey().values
      val nout = sorted.first().size
      for (i <- 0 until nout) {
        saver.write(sorted.map(x => x(i)), fileName(i))
      }
    }

  }

}

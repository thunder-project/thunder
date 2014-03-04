/** Utilities for saving results */

package thunder.util

import org.apache.spark.rdd.RDD


object Save {

  abstract class DataSaver(var directory: String) {

    def write(rdd: RDD[Int])

  }

  class ImageSaver(override var directory: String) extends DataSaver(directory) {

    var colorMode = new String

    def setColorMode(colorMode: String): ImageSaver = {
      this.colorMode = colorMode
      this
    }

    def write(rdd: RDD[Int]) = {
      rdd.collect()

    }

  }

  class TextSaver(override var directory: String) extends DataSaver(directory) {

    def write(rdd: RDD[Int]) = {
      rdd.collect()

    }

  }

  def save(rdd: RDD[Int], format: String, directory: String) {

    format match {
      case "text" =>
        val saver = new TextSaver(directory)
      case "image" =>
        val saver = new ImageSaver(directory)
        saver.setColorMode("test")
    }


  }

}

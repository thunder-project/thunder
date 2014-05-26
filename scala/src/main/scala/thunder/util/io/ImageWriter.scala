package thunder.util.io

import org.apache.spark.rdd.RDD


case class ImageWriter(directory: String) {

  var colorMode = new String

  def setColorMode(colorMode: String): ImageWriter = {
    this.colorMode = colorMode
    this
  }

  def write(rdd: RDD[(Array[Int], Double)]) = {
    rdd.collect()

  }

}

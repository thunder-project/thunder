/**
* mantis <master> <directory> <outputFile> <batchTime> <windowTime> <width> <height> <nSlices>
*
* real time processing of streaming neuroimaging data
*
* (in progress)
*
*/

package thunder

import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Vector
import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.io.File

object mantis {

  def parseVector(line: String): (Int, Vector) = {
    val nums = line.split(' ') // split line into numbers: (0) key (1) ca (2) ephys (3) id
    val k = nums(0).toInt // get index as key
    val id = nums(3).toInt
    val vals, counts = Array.fill[Double](2)(0)
    vals(id) = nums(1).toDouble
    counts(id) += 1
    return (k, Vector(vals ++ counts))
  }

  def getDiffs(vals: (Int, Vector)): (Int, Vector) = {
    val baseLine = (vals._2(0) + vals._2(1)) / (vals._2(2) + vals._2(3))
    val diff0 = ((vals._2(0) / vals._2(2)) - baseLine) / (baseLine + 0.1)
    val diff1 = ((vals._2(1) / vals._2(3)) - baseLine) / (baseLine + 0.1)
    return (vals._1, Vector(diff0, diff1))
  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try {
      op(p)
    } finally {
      p.close()
    }
  }

  def printVector(rdd: RDD[Vector], saveFile: String): Unit = {
    val data = rdd.collect().map(_.toString).map(x => x.slice(1, x.length - 1)).map(_.replace(",", ""))
    printToFile(new File(saveFile))(p => {
      data.foreach(p.println)
    })
  }

  def printToImage(rdd: RDD[Double], width: Int, height: Int, fileName: String): Unit = {
    val nPixels = width * height
    val R, G, B = rdd.collect().map(_ * 1000).map(_ + 255 / 2).map(_ toInt).map(x => if (x < 0) {
      0
    } else if (x > 255) {
      255
    } else {
      x
    })
    val RGB = Array.range(0, nPixels).flatMap(x => Array(R(x), G(x), B(x)))
    val img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val raster = img.getRaster()
    raster.setPixels(0, 0, width, height, RGB)
    ImageIO.write(img, "png", new File(fileName))
  }

  def main(args: Array[String]) {
    if (args.length < 8) {
      System.err.println("Usage: mantis <master> <directory> <outputFile> <batchTime> <windowTime> <width> <height> <nSlices>")
      System.exit(1)
    }

    // create spark context
    System.setProperty("spark.executor.memory", "120g")
    System.setProperty("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    if (args(7).toInt != 0) {
      System.setProperty("spark.default.parallelism", args(7).toString)
    }
    val batchTime = args(3).toLong
    val windowTime = args(4).toLong
    val ssc = new StreamingContext(args(0), "SimpleStreaming", Seconds(batchTime),
      System.getenv("SPARK_HOME"), List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))
    ssc.checkpoint(System.getenv("CHECKPOINTSTREAMING"))

    // update state
//    val updateFunc = (values: Seq[Vector], state: Option[Vector]) => {
//      var currentState = Vector(0, 0, 0, 0)
//      if (values.length > 0) {
//        currentState = currentState + values(0)
//      }
//      //val currentState = values(0) // ca0, ca1, n0, n1
//      val previousState = state.getOrElse(Vector(0, 0, 0, 0))
//      Some(currentState + previousState)
//    }

    // main streaming operations
    val lines = ssc.textFileStream(args(1)) // directory to monitor
    val dataStream = lines.map(parseVector _) // parse data
    val stateStream = dataStream.reduceByKeyAndWindow(_ + _, _ - _, Seconds(windowTime), Seconds(batchTime))
    val sortedStates = stateStream.map(getDiffs _).transform(rdd => rdd.sortByKey(true)).map(x => x._2(1))

    // for debugging
    //stateStream.print()
    //sortedStates.print()
    sortedStates.foreach(rdd => printToImage(rdd, args(5).toInt, args(6).toInt, args(2)))

    ssc.start()
  }
}
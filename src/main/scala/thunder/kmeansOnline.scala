package thunder

import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Vector
import scala.util.Random.nextDouble
import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.io.File

object kmeansOnline {

  def parseVector(line: String, t: Int): (Int, Vector) = {
    val nums = line.split(' ') // split line into numbers: (0) key (1) ca (2) id
    val k = nums(0).toInt // get index as key
    val id = nums(2).toInt - 1
    val vals, counts = Array.fill[Double](t)(0)
    vals(id) = nums(1).toDouble
    counts(id) += 1
    return (k, Vector(vals ++ counts))
  }

  def getDffs(vals: (Int, Vector), t: Int): (Int, Vector) = {
    val resp = vals._2.elements.slice(0,t)
    val counts = vals._2.elements.slice(t,2*t)
    val baseLine = resp.sum / counts.sum
    val dff = resp.zip(counts).map{
      case (x,y) => if (y == 0) {0} else {x/y}}.map(
      x => if (x == 0) {0} else {(x - baseLine) / (baseLine + 0.1)})
    return (vals._1, Vector(dff))
  }

  def getMeanResp(vals: (Int, Vector), t: Int) : (Int,Double) = {
    val resp = vals._2.elements.slice(0,t)
    val counts = vals._2.elements.slice(t,2*t)
    val baseLine = resp.sum / counts.sum
    return (vals._1,baseLine)
  }

  def clip(num: Double): Double = {
    var out = num
    if (num < 0) {
      out = 0
    } else if (num > 255) {
      out = 255
    }
    return out
  }

  def printToImage(rdd: RDD[(Double,Double)], width: Int, height: Int, fileName: String): Unit = {
    val nPixels = width * height
    val H = rdd.map(x => x._1).collect().map(_ * 255).map(_ toInt).map(x => clip(x))
    val B = rdd.map(x => x._2).collect().map(_ * 20).map(_ toInt).map(x => clip(x))
    val RGB = Array.range(0, nPixels).flatMap(x => Array(H(x), B(x), B(x)))
    val img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val raster = img.getRaster()
    raster.setPixels(0, 0, width, height, RGB)
    ImageIO.write(img, "png", new File(fileName))
  }

//  def printToImage(rdd: RDD[Double], width: Int, height: Int, fileName: String): Unit = {
//    val nPixels = width * height
//    val R, G, B = rdd.collect().map(_ - 1000).map(_ * 255 / 5000).map(_ toInt).map(x => if (x < 0) {
//      0
//    } else if (x > 255) {
//      255
//    } else {
//      x
//    })
//    val RGB = Array.range(0, nPixels).flatMap(x => Array(R(x), G(x), B(x)))
//    val img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
//    val raster = img.getRaster()
//    raster.setPixels(0, 0, width, height, RGB)
//    ImageIO.write(img, "png", new File(fileName))
//  }

  def closestPoint(p: Vector, centers: Array[Vector]): (Int,Double) = {
    var index = 0
    var bestIndex = 0
    var closest = Double.PositiveInfinity

    for (i <- 0 until centers.length) {
      val tempDist = p.squaredDist(centers(i))
      if (tempDist < closest) {
        closest = tempDist
        bestIndex = i
      }
    }
    return (bestIndex,closest)
  }

  def updateCenters(rdd: RDD[Vector], centers: Array[Vector]): Array[Vector] = {
    val closest = rdd.map (p => (closestPoint(p, centers)._1, (p, 1)))
    val pointStats = closest.reduceByKey{case ((x1, y1), (x2, y2)) => (x1 + x2, y1 + y2)}
    val newPoints = pointStats.map {pair => (pair._1, pair._2._1 / pair._2._2)}.collectAsMap()
    for (newP <- newPoints) {
      centers(newP._1) = newP._2
    }
    print(centers(0))
    print(centers(1))
    return centers
  }

  def main(args: Array[String]) {
    if (args.length < 8) {
      System.err.println("Usage: kmeansOnline <master> <directory> <outputFile> <batchTime> <windowTime> <k> <t> <width> <height> <nSlices>")
      System.exit(1)
    }

    // create spark context
    System.setProperty("spark.executor.memory", "120g")
    System.setProperty("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    if (args(7).toInt != 0) {
      System.setProperty("spark.default.parallelism", args(9).toString)
    }
    val saveFile = args(2)
    val batchTime = args(3).toLong
    val windowTime = args(4).toLong
    val k = args(5).toInt
    val t = args(6).toInt
    val width = args(7).toInt
    val height = args(8).toInt

    val ssc = new StreamingContext(args(0), "SimpleStreaming", Seconds(batchTime),
      System.getenv("SPARK_HOME"), List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))
    ssc.checkpoint(System.getenv("CHECKPOINTSTREAMING"))

    var centers = new Array[Vector](k)
    for (ik <- 0 until k) {
      centers(ik) = Vector(Array.fill(t)((nextDouble-0.5)/100))
    }

    // main streaming operations
    val lines = ssc.textFileStream(args(1)) // directory to monitor
    val dataStream = lines.map(x => parseVector(x,t)) // parse data
    val meanStream = dataStream.reduceByKeyAndWindow(_ + _, _ - _, Seconds(windowTime), Seconds(batchTime))
    val dffStream = meanStream.map(x => getDffs(x,t)).transform(rdd => rdd.sortByKey(true))
    dffStream.foreach(rdd =>
      centers = updateCenters(rdd.map{case (k,v) => v},centers)
    )
    dffStream.print()

    //val meanRespStream = meanStream.map(x => getMeanResp(x,t)).transform(rdd => rdd.sortByKey(true))
    //meanRespStream.foreach(rdd => printToImage(rdd.map{case (k,v) => v},width,height,saveFile))


    //meanRespStream.print()
    val dists = dffStream.transform(rdd => rdd.map{case (k,v) => closestPoint(v,centers)}.map(x => (x._1.toDouble/k,x._2)))

    dists.print()
    dists.foreach(rdd => printToImage(rdd, width, height, saveFile))

    ssc.start()
  }

}




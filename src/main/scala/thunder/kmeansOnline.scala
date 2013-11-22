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
    val out = Vector(dff)
    return (vals._1, out)
  }

  def clip(num: Int): Int = {
    var out = num
    if (num < 0) {
      out = 0
    } else if (num > 255) {
      out = 255
    }
    return out
  }

//  def corrToRGB(ind: Int): Array[Int] = {
//    var out = Array(0,0,0)
//    if (ind == 0) {out = Array(255, 0, 0)}
//    else if (ind == 1) {out = Array(0,255,0)}
//    else if (ind == 2) {out = Array(0,0,255)}
//
//    return out
//  }

  def printToImage2(rdd: RDD[Double], width: Int, height: Int, fileName: String): Unit = {
    val nPixels = width * height
    val nums = rdd.map(x => clip((x*100).toInt)).collect()
    val RGB = Array.range(0, nPixels).flatMap(x => Array(nums(x), nums(x), nums(x)))
    val img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val raster = img.getRaster()
    raster.setPixels(0, 0, width, height, RGB)
    ImageIO.write(img, "png", new File(fileName+"2.png"))
  }

  def printToImage1(rdd: RDD[Int], width: Int, height: Int, fileName: String): Unit = {
    val nPixels = width * height
    val inds = rdd.collect()
    val RGB = Array.range(0, nPixels).flatMap(x => Array(inds(x), inds(x), inds(x)))
    val img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val raster = img.getRaster()
    raster.setPixels(0, 0, width, height, RGB)
    ImageIO.write(img, "png", new File(fileName+"1.png"))
  }

  def getMeanResp(vals: (Int, Vector), t: Int) : (Int,Double) = {
    val resp = vals._2.elements.slice(0,t)
    val counts = vals._2.elements.slice(t,2*t)
    val baseLine = resp.sum / counts.sum
    return (vals._1,baseLine)
  }

    def corrcoef(p1 : Vector, p2: Vector): Double = {
    val p11 = Vector(p1.elements.map(x => x - p1.sum / p1.length))
    val p22 = Vector(p2.elements.map(x => x - p2.sum / p2.length))
    return (p11 / scala.math.sqrt(p11.dot(p11))).dot(p22 / scala.math.sqrt(p22.dot(p22)))
  }

  def closestPoint(p: Vector, centers: Array[Vector]): Int = {
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
    return bestIndex
  }

  def updateCenters(rdd: RDD[Vector], centers: Array[Vector]): Array[Vector] = {
    val closest = rdd.map (p => (closestPoint(p, centers), (p, 1)))
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
    //dffStream.print()

    val meanRespStream = meanStream.map(x => getMeanResp(x,t)).transform(rdd => rdd.sortByKey(true))
    meanRespStream.foreach(rdd => printToImage2(rdd.map{case (k,v) => v},width,height,saveFile))

    //meanRespStream.print()
    val dists = dffStream.transform(rdd => rdd.map{
      case (k,v) => closestPoint(v,centers)})

    //dists.print()
    dists.foreach(rdd => printToImage1(rdd, width, height, saveFile))

    ssc.start()
  }

}




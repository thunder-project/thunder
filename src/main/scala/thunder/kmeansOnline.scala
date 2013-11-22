package thunder

import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Vector
import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.io.File

object kmeansOnline {

  def parseVector(line: String, t: Int): (Int, Vector) = {
    val nums = line.split(' ') // split line into numbers: (0) key (1) ca (2) id
    val k = nums(0).toInt // get index as key
    val id = nums(2).toInt
    val vals, counts = Array.fill[Double](t)(0)
    vals(id) = nums(1).toDouble
    counts(id) += 1
    return (k, Vector(vals ++ counts))
  }

  def getDffs(vals: (Int, Vector), t: Int): (Int, Vector) = {
    val resp = vals._2.elements.slice(0,t)
    val counts = vals._2.elements.slice(t,2*t)
    val baseLine = resp.sum / counts.sum
    val dff = resp.zip(counts).map{case (x,y) => x/y}.map(x => (x - baseLine) / (baseLine + 0.1))
    return (vals._1, Vector(dff))
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
    //for (newP <- newPoints) {
    //  centers(newP._1) = newP._2
    //}
    centers(0) = centers(0) + Vector(Array.fill[Double](180)(1))
    print(centers(0))
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
      System.setProperty("spark.default.parallelism", args(7).toString)
    }
    val batchTime = args(3).toLong
    val windowTime = args(4).toLong
    val K = args(5).toInt
    val t = args(6).toInt
    val ssc = new StreamingContext(args(0), "SimpleStreaming", Seconds(batchTime),
      System.getenv("SPARK_HOME"), List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))
    ssc.checkpoint(System.getenv("CHECKPOINTSTREAMING"))

    var centers = Array(Vector(Array.fill[Double](t)(1)),Vector(Array.fill[Double](t)(2)),Vector(Array.fill[Double](t)(3)))

    // main streaming operations
    val lines = ssc.textFileStream(args(1)) // directory to monitor
    val dataStream = lines.map(x => parseVector(x,t)) // parse data
    val stateStream = dataStream.reduceByKeyAndWindow(_ + _, _ - _, Seconds(windowTime), Seconds(batchTime))
    val sortedStates = stateStream.map(x => getDffs(x,t))
    sortedStates.foreach(rdd =>
      centers = updateCenters(rdd.map{case (k,v) => v},centers)
    )

    //stateStream.print()
    sortedStates.print()
    print(centers(0))
    //print(centers(1))
    //print(centers(2))

    //val sortedStates = stateStream.map(x => getDffs(x,t)).transform(rdd => rdd.sortByKey(true)).map(x => x._2(1))
    // for debugging
    //stateStream.print()
    //sortedStates.print()
    //sortedStates.foreach(rdd => printToImage(rdd, args(5).toInt, args(6).toInt, args(2)))

    ssc.start()
  }

}




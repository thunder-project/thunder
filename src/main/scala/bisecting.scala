/**
 * bisecting <master> <inputFile> <outputFile>
 *
 * divisive hierarchical clustering using bisecting k-means
 * writes result to JSON for display using d3
 *
 * (in progress)
 *
 */

import java.io.File
import spark.SparkContext
import spark.SparkContext._
import spark.util.Vector
import scala.collection.mutable.ArrayBuffer
import cc.spray.json._
import cc.spray.json.DefaultJsonProtocol._


object bisecting {

  def parseVector(line: String): Vector = {
    val nums = line.split(' ')
    val k = nums(0).toDouble.toInt
    val vec = nums.slice(1, nums.length).map(_.toDouble)
    //val k3 = nums.slice(0,3).map(_.toDouble) // get xyz coordinates
    //val k = (k3(0) + (k3(1) - 1)*2034 + (k3(2) - 1)*2034*1134).toInt // convert to linear index
    //val vec = nums.slice(3,nums.length).map(_.toDouble)
    return Vector(vec)
  }

  def closestPoint(p: Vector, centers: Array[Vector]): Int = {

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

  def split(cluster: spark.RDD[Vector]): Array[Vector] = {

    // use k-means to split a cluster in two
    // try multiple splits and keep the best
    val convergeDist = 0.0001
    val iters = 5
    var best = Double.PositiveInfinity
    var centersFinal = cluster.takeSample(false,2,1).toArray
    for (iter <- 0 until iters) {
      val centers = cluster.takeSample(false, 2, iter).toArray
      var tempDist = 1.0
      // do k-means iterations till convergence
      while(tempDist > convergeDist) {
        val closest = cluster.map (p => (closestPoint(p, centers), (p, 1)))
        val pointStats = closest.reduceByKey{case ((x1, y1), (x2, y2)) => (x1 + x2, y1 + y2)}
        val newPoints = pointStats.map {pair => (pair._1, pair._2._1 / pair._2._2)}.collectAsMap()
        tempDist = 0.0
        for (i <- 0 until 2) {
          tempDist += centers(i).squaredDist(newPoints(i))
        }
        for (newP <- newPoints) {
          centers(newP._1) = newP._2
        }
      }
      // check within-class similarity of clusters
      var totalDist = 0.0
      for (i <- 0 until 2) {
        totalDist += cluster.filter(x => closestPoint(x,centers)==i).map(x => x.squaredDist(centers(i))).reduce(_+_)
      }
      if (totalDist < best) {
        best = totalDist
        centersFinal = centers
      }
    }
    return centersFinal
  }


  def main(args: Array[String]) {

    if (args.length < 3) {
      System.err.println("Usage: bisecting <master> <inputFile> <outputFile> ")
      System.exit(1)
    }

    System.setProperty("spark.executor.memory", "120g")
    System.setProperty("spark.serializer", "spark.KryoSerializer")
    //System.setProperty("spark.default.parallelism", "50")
    val sc = new SparkContext(args(0), "hierarchical", System.getenv("SPARK_HOME"),
      List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))
    //sc.setCheckpointDir(System.getenv("CHECKPOINT"))


    val k = 1
    val data = sc.textFile(args(1)).map(parseVector _).cache()

    val clusters = ArrayBuffer(data)

    while (clusters.size < k) {

      // find largest cluster for splitting
      val ind = clusters.map(_.count()).view.zipWithIndex.max._2

      // split into 2 clusters using k-means
      val centers = split(clusters(ind)) // find 2 cluster centers
      val cluster1 = clusters(ind).filter(x => closestPoint(x,centers) == 0)
      val cluster2 = clusters(ind).filter(x => closestPoint(x,centers) == 1)

      // remove old cluster, add the 2 new ones
      clusters.remove(ind)
      clusters.append(cluster1)
      clusters.append(cluster2)

      // keep track of results


    }
  }
 }

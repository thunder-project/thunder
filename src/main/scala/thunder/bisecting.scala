/**
 * bisecting <master> <inputFile> <outputFileTree> <outputFileImg> <k> <subIters> <threshold> <nSlices>
 *
 * divisive hierarchical clustering using bisecting k-means
 * assumes input is a text file of spatio-temporal time series data
 * with each row containing x,y,z,t1,t2,t3,...
 * writes results to images and JSON for easy display, e.g. using D3.js
 *
 */

package thunder

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.util.Vector
import scala.collection.mutable.ArrayBuffer
import spray.json._
import spray.json.DefaultJsonProtocol._

case class Cluster(var key: Int, var center: List[Map[String,Double]], var children: Option[List[Cluster]])

object MyJsonProtocol extends DefaultJsonProtocol {
      implicit val menuItemFormat: JsonFormat[Cluster] = lazyFormat(jsonFormat(Cluster, "key", "center", "children"))
}

import MyJsonProtocol._

object bisecting {

  def insert(node: Cluster, key: Int, children: List[Cluster]) {
    // recursively search cluster tree for desired key and insert children
    if (node.key == key) {
      node.children = Some(children)
    } else {
      if (node.children.getOrElse(0) != 0) {
        insert(node.children.get(0),key,children)
        insert(node.children.get(1),key,children)
      }
    }
  }

  def std(vec: Vector): Double = {
    //val mean = Vector(Array.fill(vec.length)(vec.sum / vec.length))
    //return scala.math.sqrt(vec.squaredDist(mean)/(vec.length - 1))
    return scala.math.sqrt(vec.dot(vec)/(vec.length - 1))
  }

  def makeMap(vec: Array[Double]): List[Map[String,Double]] = {
    return vec.toList.zipWithIndex.map(x => Map("x"->x._2.toDouble,"y"->x._1))
  }

  def parseVector(line: String): ((Array[Int]),Vector) = {
    var vec = line.split(' ').drop(3).map(_.toDouble)
    val inds = line.split(' ').take(3).map(_.toDouble.toInt) // xyz coords
    val mean = vec.sum / vec.length
    vec = vec.map(x => (x - mean)/(mean + 0.1)) // time series
    return (inds,Vector(vec))
  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try {
      op(p)
    } finally {
      p.close()
    }
  }

  def printToImage(rdd: RDD[(Array[Int],Int)], w: Int, h: Int, d: Array[Int], fileName: String): Unit = {
    for (id <- d) {
      val plane = rdd.filter(_._1(2) == id).map{case (k,v) => (k(0),k(1),v)}.toArray()
      val X = plane.map(_._1)
      val Y = plane.map(_._2)
      val RGB = plane.map(_._3)
      val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
      val raster = img.getRaster()
      (X,Y,RGB).zipped.foreach{case(x,y,rgb) => raster.setPixel(x-1, y-1, Array(rgb,rgb,rgb))}
      ImageIO.write(img, "png", new File(fileName+"-plane"+id.toString+".png"))
    }
  }

  def closestPoint(p: Vector, centers: Array[Vector]): Int = {
    if (p.squaredDist(centers(0)) < p.squaredDist(centers(1))) {
      return 0
    }
    else {
      return 1
    }
  }

  def split(cluster: RDD[Vector], subIters: Int): Array[Vector] = {
    // use k-means with k=2 to split a cluster in two
    // try multiple splits and keep the best
    val convergeDist = 0.001
    var best = Double.PositiveInfinity
    var centersFinal = cluster.takeSample(false,2,1).toArray
    for (iter <- 0 until subIters) {
      val centers = cluster.takeSample(false, 2, iter).toArray
      var tempDist = 1.0
      // do iterations of k-means till convergence
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

    if (args.length < 8) {
      System.err.println("Usage: bisecting <master> <inputFile> <outputFileTree> <outputFileImg> <k> <subIters> <threshold> <nSlices>")
      System.exit(1)
    }

    // collect arguments
    val master = args(0)
    val inputFile = args(1)
    val outputFileTree = args(2)
    val outputFileImg = args(3)
    val k = args(4).toDouble
    val subIters = args(5).toInt
    val threshold = args(6).toDouble
    val nSlices = args(7).toInt

    if (nSlices != 0) {
      println("changing parallelism")
      System.setProperty("spark.default.parallelism", nSlices.toString)
    }
    System.setProperty("spark.executor.memory", "120g")
    //System.setProperty("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    val sc = new SparkContext(master, "bisecting", System.getenv("SPARK_HOME"),
      List("target/scala-2.9.3/thunder_2.9.3-1.0.jar","project/spray-json_2.9.3-1.2.5.jar"))
   
    // load raw data
    val dataRaw = sc.textFile(inputFile).map(parseVector _).filter{case (k,x) => std(x) > threshold}.cache()

    // sort x and y keys to get bounds
    val w = dataRaw.map{case (k,v) => k(0)}.top(1).take(1)(0)
    val h = dataRaw.map{case (k,v) => k(1)}.top(1).take(1)(0)
    val d = dataRaw.filter{case (k,v) => (k(0) == 1) & (k(1) == 1)}.map{case (k,v) => k(2)}.toArray()

    // load data
    //val data = threshold match {
    //  case 0 => dataRaw.cache()
    //  case _ => dataRaw.filter{case (k,x) => std(x) > threshold}.mapValues(x => x / std(x)).cache()
    //}

    val data = dataRaw

    println("starting")

    // create array with first cluster and compute its center
    val clusters = ArrayBuffer((0,dataRaw))
    val n = dataRaw.count()
    val center = dataRaw.map(_._2).reduce(_+_).elements.map(x => x / n)
    val tree = Cluster(0,makeMap(center),None)
    var count = 1

    // print first cluster as an image
    printToImage(dataRaw.map{case (k,v) => (k,255)}, w, h, d, outputFileImg + "-cluster" + 0.toString)

    // start timer
    val startTime = System.nanoTime

    while (clusters.size < k) {

      println(clusters.size.toString + " clusters, starting new iteration")

      // find largest cluster for splitting
      val ind = clusters.map(_._2.count()).view.zipWithIndex.max._2

      // split into 2 clusters using k-means
      val centers = split(clusters(ind)._2.map(_._2),subIters) // find 2 cluster centers
      val cluster1 = clusters(ind)._2.filter(x => closestPoint(x._2,centers) == 0)
      val cluster2 = clusters(ind)._2.filter(x => closestPoint(x._2,centers) == 1)

      // get new indices
      val newInd1 = count
      val newInd2 = count + 1
      count += 2

      // update tree with results
      insert(tree,clusters(ind)._1,List(
        Cluster(newInd1,makeMap(centers(0).elements),None),
        Cluster(newInd2,makeMap(centers(1).elements),None)))

      // write clusters to images
      printToImage(cluster1.map{case (k,v) => (k,255)}, w, h, d, outputFileImg + "-cluster" + newInd1.toString)
      printToImage(cluster2.map{case (k,v) => (k,255)}, w, h, d, outputFileImg + "-cluster" + newInd2.toString)

      // remove old cluster, add the 2 new ones
      clusters.remove(ind)
      clusters.append((newInd1,cluster1))
      clusters.append((newInd2,cluster2))

      // print current tree
      val out = Array(tree.toJson.prettyPrint)
      printToFile(new File(outputFileTree))(p => {
        out.foreach(p.println)
      })

    }

    println("Bisecting took: "+(System.nanoTime-startTime)/1e9+"s")

  }
 }

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

  case class Cluster(var key: Int, var center: List[Map[String,Double]], var children: Option[List[Cluster]])

  object MyJsonProtocol extends DefaultJsonProtocol {
    implicit val menuItemFormat: JsonFormat[Cluster] = lazyFormat(jsonFormat(Cluster, "key", "center", "children"))
  }

  import MyJsonProtocol._

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
    val mean = Vector(Array.fill(vec.length)(vec.sum / vec.length))
    return scala.math.sqrt(vec.squaredDist(mean)/(vec.length - 1))
  }

  def makeXYmap(vec: Array[Double]): List[Map[String,Double]] = {
    return vec.toList.zipWithIndex.map(x => Map("x"->x._2.toDouble,"y"->x._1))
  }

  def toYesOrNo(choice: Int): String = choice match {
    case 1 => "yes"
    case 0 => "no"
    case _ => "error"
  }

  def parseVector(line: String, mode: String): Vector = mode match {
    case "raw" => Vector(line.split(' ').map(_.toDouble))
    case "ca" => {
      Vector(line.split(' ').drop(3).map(_.toDouble))
    }
    case "dff" => {
      val vec = line.split(' ').drop(3).map(_.toDouble)
      val mean = vec.sum / vec.length
      Vector(vec.map(x => (x - mean)/(mean + 0.1)))
    }
    case _ => Vector(line.split(' ').map(_.toDouble))
  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try {
      op(p)
    } finally {
      p.close()
    }
  }

  def closestPoint(p: Vector, centers: Array[Vector]): Int = {

    if (p.squaredDist(centers(0)) < p.squaredDist(centers(1))) {
      return 0
    }
    else {
      return 1
    }

//    var bestIndex = 0
//    var closest = Double.PositiveInfinity
//    for (i <- 0 until centers.length) {
//      val tempDist = p.squaredDist(centers(i))
//      if (tempDist < closest) {
//        closest = tempDist
//        bestIndex = i
//      }
//    }
//    return bestIndex
  }

  def split(cluster: spark.RDD[Vector], subIters: Int): Array[Vector] = {

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
      System.err.println("Usage: bisecting <master> <inputFile> <outputFile> <inputMode> <k> <subIters> <threshold> <nSlices>")
      System.exit(1)
    }

    // collect arguments
    val master = args(0)
    val inputFile = args(1)
    val outputFile = args(2)
    val inputMode = args(3)
    val k = args(4).toDouble
    val subIters = args(5).toInt
    val threshold = args(6).toDouble
    val nSlices = args(7).toInt
    val startTime = System.nanoTime

    System.setProperty("spark.executor.memory", "120g")
    System.setProperty("spark.serializer", "spark.KryoSerializer")
    if (nSlices != 0) {
      System.setProperty("spark.default.parallelism", nSlices.toString)
    }
    val sc = new SparkContext(master, "hierarchical", System.getenv("SPARK_HOME"),
      List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))
    //sc.setCheckpointDir(System.getenv("CHECKPOINT"))

    // load data
    val data = threshold match {
      case 0 => sc.textFile(inputFile).map(parseVector (_,inputMode)).cache()
      case _ => sc.textFile(inputFile).map(parseVector (_,inputMode)).filter(x => std(x) > threshold).map(x => x / std(x)).cache()
    }

    // create array with first cluster and compute its center
    val clusters = ArrayBuffer((0,data))
    val center = data.reduce(_+_).elements.map(x => x / data.count())
    val tree = Cluster(0,makeXYmap(center),None)
    var count = 1

    while (clusters.size < k) {

      println(clusters.size.toString + " clusters, starting new iteration")

      // find largest cluster for splitting
      val ind = clusters.map(_._2.count()).view.zipWithIndex.max._2

      // split into 2 clusters using k-means
      val centers = split(clusters(ind)._2,subIters) // find 2 cluster centers
      val cluster1 = clusters(ind)._2.filter(x => closestPoint(x,centers) == 0)
      val cluster2 = clusters(ind)._2.filter(x => closestPoint(x,centers) == 1)

      // get new indices
      val newInd1 = count
      val newInd2 = count + 1
      count += 2

      // update tree with results
      insert(tree,clusters(ind)._1,List(
        Cluster(newInd1,makeXYmap(centers(0).elements),None),
        Cluster(newInd2,makeXYmap(centers(1).elements),None)))

      // remove old cluster, add the 2 new ones
      clusters.remove(ind)
      clusters.append((newInd1,cluster1))
      clusters.append((newInd2,cluster2))

    }

    val out = Array(tree.toJson.prettyPrint)
    printToFile(new File(outputFile))(p => {
      out.foreach(p.println)
    })

    println("Bisecting took: "+(System.nanoTime-startTime)/1e9+"s")

  }
 }

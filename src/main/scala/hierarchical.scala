/**
 * hierarchical <master> <inputFile>
 *
 * efficient hierarchical clustering using the nearest neighbor chain algorithm
 * writes result to JSON for visualization in D3
 *
 */

import java.io.File
import spark.SparkContext
import spark.SparkContext._
import spark.util.Vector
import scala.collection.mutable.Stack

object hierarchical {

  def parseVector(line: String): (Int,(Double,Vector)) = {
    val nums = line.split(' ')
    //val k = nums(0).toDouble.toInt
    //val vec = nums.slice(1,nums.length).map(_.toDouble)
    val k3 = nums.slice(0,3).map(_.toDouble) // get xyz coordinates
    val k = (k3(0) + (k3(1) - 1)*2034 + (k3(2) - 1)*2034*1134).toInt // convert to linear index
    val vec = nums.slice(3,nums.length).map(_.toDouble)
    return (k,(1.toDouble,Vector(vec)))
  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try { op(p) } finally { p.close() }
  }

  def printVector(data: Array[Vector], saveFile: String): Unit = {
    val out = data.map(_.toString).map(x => x.slice(1,x.length-1)).map(_.replace(",",""))
    printToFile(new File(saveFile))(p => {
      out.foreach(p.println)
    })
  }

  def distance(p1: (Double,Vector), p2: (Double,Vector)): Double = {
    // compute Ward's distance between two clusters
    return p1._2.squaredDist(p2._2) * ((p1._1 * p2._1) / (p1._1 + p2._1))
  }

  def findMin(p1: (Double,(Int,(Double,Vector))), p2: (Double,(Int,(Double,Vector)))): (Double,(Int,(Double,Vector))) = {
    if (p1._1 < p2._1) {
      return p1
    } else {
      return p2
    }
  }

  def merge(p1: (Double,Vector), p2: (Double,Vector)): (Double,Vector) = {
    // merge two clusters, keeping track of the number of points in the cluster
    return ((p1._1 + p2._1), (p1._1 * p1._2 + p2._1 * p2._2) / (p1._1 + p2._1))
  }

  def updateKey(p: (Int,(Double,Vector)), ind1: Int, ind2: Int, newInd: Int): (Int,(Double,Vector)) = {
    // if a point has either ind1 or ind2, switch its index to newInd
    if ((p._1 == ind1) | (p._1 == ind2)) {
      return (newInd,p._2)
    } else {
      return p
    }
  }

  def main(args: Array[String]) {

    if (args.length < 3) {
      System.err.println("Usage: hierarchical <master> <inputFile> <outputFile> ")
      System.exit(1)
    }

    System.setProperty("spark.executor.memory","120g")
    System.setProperty("spark.serializer", "spark.KryoSerializer")
    //System.setProperty("spark.default.parallelism", "50")
    val sc = new SparkContext(args(0), "hierarchical", System.getenv("SPARK_HOME"),
        List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))
    sc.setCheckpointDir(System.getenv("CHECKPOINT"))

    val data = sc.textFile(args(1)).map(parseVector _).cache()

    val n = data.count().toInt
    var iter = 0
    //val clusters = Array.fill(n-1)(Vector(0,0,0))

    while (data.count() > 1) {

      var rnn = 0 // set rnn found flag to 0
      var p = data.first() // start at first point for first iteration
      var pOld, nn = p // initialize auxillary points
      while (rnn == 0) { // grow NN chain till we find an RNN
        nn = data.filter(x => x._1 != p._1) // eliminate self
          .map(x => (distance(x._2,p._2),x)) // compute distances
          .reduce((x,y) => findMin(x,y))._2 // get nearest neighbor
          //.sortByKey(true).first()._2 // get nearest neighbor
        if (nn._1 == pOld._1) {
          rnn = 1 // nearest neighbor is last point, RNN found
        } else {
          pOld = p // update last point
          p = nn
        }
      }
      //data = data.map(x => updateKey(x,p._1,nn._1,iter + n)).reduceByKey(merge _)

      if ((iter % 10) == 0) { // checkpoint
        data.checkpoint()
      }

      //clusters(iter) = Vector(p._1,nn._1,math.sqrt(distance(p._2,nn._2)*2))
      iter += 1
      println("iteration" + iter)
//      if (iterChain > 1) {
//        p = pOld2 // if the chain took more than two steps, use point prior to RNN
//      } else {
//        p = data.first() // otherwise just restart at first point
//      }
    }

    //printVector(clusters,args(2))
  }
}
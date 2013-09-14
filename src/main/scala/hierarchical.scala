/**
 * hierarchical <master> <inputFile>
 *
 * efficient hierarchical clustering using the nearest neighbor chain algorithm
 * writes result to JSON for visualization in D3
 *
 */

import spark.SparkContext
import spark.SparkContext._
import spark.util.Vector
import scala.collection.mutable.Stack

object hierarchical {

  def parseVector(line: String): (Int,(Double,Vector)) = {
    val nums = line.split(' ')
    val k = nums(0).toDouble.toInt
    val vec = nums.slice(1,nums.length-1).map(_.toDouble)
    return (k,(1.toDouble,Vector(vec)))
  }

  def distance(p1: (Double,Vector), p2: (Double,Vector)): Double = {
    return p1._2.squaredDist(p2._2) * ((p1._1 * p2._1) / (p1._1 + p2._1))
  }

  def merge(p1: (Double,Vector), p2: (Double,Vector)): (Double,Vector) = {
    return ((p1._1 + p2._1), (p1._1 * p1._2 + p2._1 * p2._2) / (p1._1 + p2._1))
  }

  def updateKey(p: (Int,(Double,Vector)), ind1: Int, ind2: Int): (Int,(Double,Vector)) = {
    if (p._1 == ind1) {
      return (ind2,p._2)
    } else {
      return p
    }
  }

  def main(args: Array[String]) {

    if (args.length < 2) {
      System.err.println("Usage: hierarchical <master> <inputFile> ")
      System.exit(1)
    }
    
    val sc = new SparkContext("local", "hierarchical", System.getenv("SPARK_HOME"),
        List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))

    var data = sc.textFile(args(1)).map(parseVector _).cache()

    while (data.count() > 1) {

      var rnn = 0 // set rnn found flag to 0
      var p = data.first() // take the first value
      var pOld, nn = p
      while (rnn == 0) { // grow NN chain till we find an RNN
        nn = data.filter(x => x._1 != p._1) // eliminate self
          .map(x => (distance(x._2,p._2),x)) // compute distances
          .sortByKey(true).map(x => x._2).first() // get nearest neighbor
        if (nn._1 == pOld._1) {
          rnn = 1
        } else {
          pOld = p
          p = nn
        }
      }
      println(p._1)
      println(nn._1)
      data = data.map(x => updateKey(x,p._1,nn._1)).reduceByKey(merge _)

      println(data.count())

    }
  }
}
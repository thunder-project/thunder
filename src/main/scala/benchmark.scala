/**
 * benchmark <master> <directory> <outputFile> <batchTime> <windowTime> <width> <height> <nSlices>
 *
 * testing timing for several different operations
 *
 */

import spark.SparkContext
import spark.SparkContext._
import spark.util.Vector
import cc.spray.json._
import cern.jet.math._
import cern.colt.matrix._
import cern.colt.matrix.linalg._

object benchmark {

  val factory2D = DoubleFactory2D.dense
  val factory1D = DoubleFactory1D.dense
  val algebra = Algebra.DEFAULT

  def parseVector(line: String): Vector = {
    val nums = line.split(' ')
    val vec = nums.slice(1, nums.length).map(_.toDouble)
    return Vector(vec)
  }

  def time[A](f: => A) = {
    val s = System.nanoTime
    val ret = f
    println("time: "+(System.nanoTime-s)/1e9+"s")
    ret
  }

  def parseVectorColt(line: String): DoubleMatrix1D = {
    val nums = line.split(' ')
    val vec = nums.slice(1, nums.length).map(_.toDouble)
    return factory1D.make(vec)
  }

  def outerProd(vec1: DoubleMatrix1D, vec2: DoubleMatrix1D): DoubleMatrix2D = {
    val out = factory2D.make(vec1.size,vec2.size)
    algebra.multOuter(vec1,vec2,out)
    return out
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


  def main(args: Array[String]) {

    System.setProperty("spark.executor.memory", "120g")
    System.setProperty("spark.serializer", "spark.KryoSerializer")
    //System.setProperty("spark.default.parallelism", "50")
    val sc = new SparkContext(args(0), "benchmark", System.getenv("SPARK_HOME"),
      List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))
    val algorithm = args(1)


    if (algorithm == "cov") {
      /** calculating a covariance matrix **/
      val data = sc.textFile(args(2)).map(parseVectorColt _).cache()
      val n = data.count()
      time {
      val cov = data.map(x => outerProd(x,x)).reduce(_.assign(_,Functions.plus))
      }
    }

    if (algorithm == "regress") {
      /** do regression on each pixel **/
      val data = sc.textFile(args(2)).map(parseVectorColt _).cache()
      val n = data.count()
      val m = data.first().size()
      val y = factory1D.random(m)
      time {
        val out = data.map(x => algebra.mult(x,y)).reduce(_+_)
      }
    }

    if (algorithm == "kmeans") {
      /** one iteration of kmeans with k = 3 **/
      val data = sc.textFile(args(2)).map(parseVector _).cache()
      val n = data.count()
      val kPoints = data.takeSample(false, 5, 42).toArray
      time {
        val closest = data.map (p => (closestPoint(p, kPoints), (p, 1)))
        val pointStats = closest.reduceByKey{case ((x1, y1), (x2, y2)) => (x1 + x2, y1 + y2)}
        val newPoints = pointStats.map {pair => (pair._1, pair._2._1 / pair._2._2)}.collectAsMap()
      }
    }






  }

}

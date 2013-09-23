/**
 * benchmark <master> <directory> <outputFile> <batchTime> <windowTime> <width> <height> <nSlices>
 *
 * time several different operations
 *
 */

import spark.SparkContext
import spark.util.Vector
import cern.jet.math._
import cern.colt.matrix._
import cern.colt.matrix.linalg._

object benchmark {

  val factory2D = DoubleFactory2D.dense
  val factory1D = DoubleFactory1D.dense
  val algebra = Algebra.DEFAULT

  def parseVector(line: String): DoubleMatrix1D = {
    val nums = line.split(' ')
    val vec = nums.slice(1, nums.length).map(_.toDouble)
    return factory1D.make(vec)
  }

  def outerProd(vec1: DoubleMatrix1D, vec2: DoubleMatrix1D): DoubleMatrix2D = {
    val out = factory2D.make(v1.size,v2.size)
    algebra.multOuter(vec1,vec2,out)
    return out
  }


  def main(args: Array[String]) {

    System.setProperty("spark.executor.memory", "120g")
    System.setProperty("spark.serializer", "spark.KryoSerializer")
    //System.setProperty("spark.default.parallelism", "50")
    val sc = new SparkContext(args(0), "benchmark", System.getenv("SPARK_HOME"),
      List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))


    val data = sc.textFile(args(1)).map(parseVector _).cache()
    val n = data.count()

    /** calculating a covariance matrix **/
    val cov = data.map(x => outerProd(x,x)).reduce(_.assign(_,Functions.plus)) / n

    /** do regression on each pixel **/


    /** one iteration of kmeans with k = 3 **/


  }

}

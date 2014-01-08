/**
 * grrr <master> <inputFileR> <inputFileX> <outputFile>
 *
 * generalized reduced rank regression
 *
 * solve the problem R = C*X + Q + e subject to rank(C) = k1 and rank(Q) = k2
 * by estimating U and V (the left and right singular vectors of C)
 * and T and P (the left and right singular vectors of Q)
 *
 * (in progress)
 *
 */

package thunder

import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import java.io.File
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import cern.colt.matrix._
import cern.colt.matrix.linalg._
import thunder.util.MatrixRDD
import scala.util.Random
import cern.jet.math.Functions

object grrr {

  val factory2D = DoubleFactory2D.dense
  val factory1D = DoubleFactory1D.dense
  val alg = Algebra.DEFAULT

  def parseVector(line: String): ((Array[Int]), DoubleMatrix1D) = {
    val vec = line.split(' ').drop(3).map(_.toDouble)
    val inds = line.split(' ').take(3).map(_.toDouble.toInt) // xyz coords
    //val mean = vec.sum / vec.length
    //vec = vec.map(x => (x - mean)/(mean + 0.1)) // time series
    return (inds,factory1D.make(vec))
  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try {
      op(p)
    } finally {
      p.close()
    }
  }

  def printMatrix(data: DoubleMatrix2D, saveFile: String): Unit = {
    // print a DoubleMatrix2D to text by writing each row as a string
    val out = data.toArray()
    printToFile(new File(saveFile))(p => {
      out.foreach(x => p.println(x.mkString(" ")))
    })
  }

  def plus(mat1: RDD[DoubleMatrix1D], mat2: RDD[DoubleMatrix1D]): RDD[DoubleMatrix1D] = {
    // add two RDD matrices
    mat1.zip(mat2).map{case (x1,x2) => x1.assign(x2,Functions.plus)}
  }

  def minus(mat1: RDD[DoubleMatrix1D], mat2: RDD[DoubleMatrix1D]): RDD[DoubleMatrix1D] = {
    // subtract two RDD matrices
    mat1.zip(mat2).map{case (x1,x2) => x1.assign(x2,Functions.minus)}
  }

  def times(mat1: RDD[DoubleMatrix1D], mat2:DoubleMatrix2D) : RDD[DoubleMatrix1D] = {
    // multiply an RDD matrix by a smaller matrix
    mat1.map(x => alg.mult(alg.transpose(mat2),x))
  }

  def times2(mat1: RDD[DoubleMatrix1D], mat2: RDD[DoubleMatrix1D]) : DoubleMatrix2D = {
    // multiply one RDD matrix by the transpose of another one
    mat1.zip(mat2).map{case (x,y) => outerProd(x,y)}.reduce(_.assign(_,Functions.plus))
  }

  def rdivide(mat1: RDD[DoubleMatrix1D], mat2: DoubleMatrix2D) : RDD[DoubleMatrix1D] = {
    // right matrix division
    times(mat1, alg.transpose(alg.inverse(alg.transpose(mat2))))
  }

  def svd(mat1: RDD[DoubleMatrix1D], k: Int, m: Int, normMode: String): (RDD[DoubleMatrix1D], DoubleMatrix2D) = {
    // get the rank-k svd of an RDD matrix
    val cov = mat1.map(x => outerProd(x,x)).reduce(_.assign(_,Functions.plus))
    val svd = new SingularValueDecomposition(cov)
    val S = svd.getSingularValues().take(k)
    val inds = Range(0,k).toArray
    var V = factory2D.make(svd.getU().viewSelection(Range(0,m).toArray,inds).toArray())
    val multFac = alg.mult(V,factory2D.diagonal(factory1D.make(S.map(x => 1 / scala.math.sqrt(x)))))
    var U = times(mat1,multFac)
    if (normMode == "norm") {
      val rescale = factory2D.diagonal(factory1D.make(S.map(x => scala.math.sqrt(scala.math.sqrt(x)))))
      println(rescale)
      U = times(U,rescale)
      V = alg.mult(V,rescale)
    }
    return (U,alg.transpose(V))
  }

  def outerProd(vec1: DoubleMatrix1D, vec2: DoubleMatrix1D): DoubleMatrix2D = {
    val out = factory2D.make(vec1.size,vec2.size)
    alg.multOuter(vec1,vec2,out)
    return out
  }


  def main(args: Array[String]) {

    if (args.length < 5) {
      System.err.println("Usage: grrr <master> <inputFileR> <inputFileX> <outputFileTxt> <outputFileImg>")
      System.exit(1)
    }

    val master = args(0)
    val inputFileR = args(1)
    val inputFileX = args(2)
    val outputFileTxt = args(3)
    val outputFileImg = args(4)

    System.setProperty("spark.executor.memory", "120g")
    val sc = new SparkContext(master, "grrr", System.getenv("SPARK_HOME"),
      List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))

    val data = sc.textFile(inputFileR).map(parseVector _)
    val X = factory2D.make(sc.textFile(inputFileX).map(x => x.split(' ').map(_.toDouble)).toArray())

    val w = data.map{case (k,v) => k(0)}.top(1).take(1)(0)
    val h = data.map{case (k,v) => k(1)}.top(1).take(1)(0)
    val d = data.map{case (k,v) => k(2)}.top(1).take(1)(0)

    val n = data.count().toInt
    val m = data.first()._2.size
    val c = X.viewColumn(0).size

    val k1 = 3
    val k2 = 3

    val nIter = 10
    var iIter = 0


    // working version

    // initialize T and P
//    var T = sc.parallelize(0 until n).map( i => factory1D.make(k2,0))
//    var P = factory2D.make(k2,m,0)
//    var C = sc.parallelize(0 until n).map( i => factory1D.make(c,0))
//
//    val R = data.map{case (k,v) => v}
//
//    while (iIter < nIter) {
//      println("starting iteration " + iIter.toString())
//      val C1 = rdivide(minus(R,times(T,P)),X)
//      println(R.first())
//      val U1 = svd(times(C1,X), k1, m, "basic")._1
//      C = times(U1,times2(U1,C1))
//      println(C.first())
//      val SVD2 = svd(minus(R,times(C,X)),k2, m, "norm")
//      T = sc.parallelize(SVD2._1.collect())
//      P = SVD2._2
//      iIter += 1
//    }
//
//    val SVD1 = svd(C,k1,c,"basic")
//    val U = sc.parallelize(SVD1._1.collect())
//    val V = SVD1._2

    // matrix RDD version

    // initialize T and P
    val R = MatrixRDD(data.map{case (k,v) => v})
    var T = MatrixRDD(R.mat.map( x => factory1D.make(k2,0)))
    var P = factory2D.make(k2,m,0)
    var C = MatrixRDD(R.mat.map( x => factory1D.make(c,0)))

    while (iIter < nIter) {
      println("starting iteration " + iIter.toString())
      val C1 = (R - T * P) / X
      println(R.mat.first())
      val U1 = (C1 * X).svd(k1, m, "basic")._1
      C = U1 * (U1 ** C1)
      println(C.mat.first())
      val SVD2 = (R - C * X).svd(k2, m, "norm")
      T = MatrixRDD(sc.parallelize(SVD2._1.mat.collect()))
      P = SVD2._2
      iIter += 1
    }

    val SVD1 = C.svd(k1,c,"basic")
    val U = SVD1._1
    val V = SVD1._2


    // add keys back in
    //val out = data.map{case (k,v) => k}.zip(U.mat)

    // print time series
    //printMatrix(V, outputFileTxt)

    // print time series
    //printMatrix(P, outputFileTxt)

    // print images
    //for (ik <- 0 until k1) {
    //  printToImage(out.map{case (k,v) => (k,v.get(ik)*100)}, w, h, d, outputFileImg + "-comp" + ik.toString() + ".png")
    //}

  }



}

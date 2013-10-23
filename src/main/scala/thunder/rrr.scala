/**
 * rrr <master> <inputFileR> <inputFileX> <outputFile>
 *
 * reduced rank regression
 *
 * solve the problem R = CX + e subject to rank(C) = k
 * by estimating U and V, the left and right singular vectors of C
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
import org.jblas.DoubleMatrix
import org.jblas.Solve
import org.jblas.Eigen
import thunder.util.MatrixRDD
import cern.jet.math.Functions
import cern.colt.matrix.{DoubleMatrix1D, DoubleMatrix2D, DoubleFactory1D, DoubleFactory2D}
import cern.colt.matrix.linalg.Algebra
import cern.colt.matrix.linalg.EigenvalueDecomposition

object rrr {

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

  def printToImage(rdd: RDD[(Array[Int],Double)], w: Int, h: Int, d: Int, fileName: String): Unit = {
    for (id <- 0 until d) {
      val plane = rdd.filter(_._1(2) == d)
      val X = plane.map(_._1(0)).collect()
      val Y = plane.map(_._1(1)).collect()
      val vals = plane.map(_._2).collect()
      val RGB = vals.map(rgb => (255*(rgb - vals.min)/(vals.max - vals.min)).toInt)
      val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
      val raster = img.getRaster()
      (X,Y,RGB).zipped.foreach{case(x,y,rgb) => raster.setPixel(x-1, y-1, Array(rgb,rgb,rgb))}
      ImageIO.write(img, "png", new File("plane"+id.toString+"-"+fileName))
    }
  }

  def outerProd(vec1: DoubleMatrix1D, vec2: DoubleMatrix1D): DoubleMatrix2D = {
    val out = factory2D.make(vec1.size,vec2.size)
    alg.multOuter(vec1,vec2,out)
    return out
  }

//  def printMatrix(data: DoubleMatrix2D, saveFile: String): Unit = {
//    // print a DoubleMatrix2D to text by writing each row as a string
//    val out = data.toArray()
//    printToFile(new File(saveFile))(p => {
//      out.foreach(x => p.println(x.mkString(" ")))
//    })
//  }

  def main(args: Array[String]) {

    if (args.length < 5) {
      System.err.println("Usage: rrr <master> <inputFileR> <inputFileX> <outputFileTxt> <outputFileImg>")
      System.exit(1)
    }

    val master = args(0)
    val inputFileR = args(1)
    val inputFileX = args(2)
    val outputFileTxt = args(3)
    val outputFileImg = args(4)

    System.setProperty("spark.executor.memory", "120g")
    val sc = new SparkContext(master, "rrr", System.getenv("SPARK_HOME"),
      List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))

    val data = sc.textFile(inputFileR).map(parseVector _).cache()
    val X = factory2D.make(sc.textFile(inputFileX).map(x => x.split(' ').map(_.toDouble)).toArray())

    println("getting dimensions")
    val w = data.map{case (k,v) => k(0)}.top(1).take(1)(0)
    val h = data.map{case (k,v) => k(1)}.top(1).take(1)(0)
    val d = data.map{case (k,v) => k(2)}.top(1).take(1)(0)

    println("initializing variables")
    val n = data.count().toInt
    val m = data.first()._2.size()
    val c = X.rows

    val k1 = 3

    // strip keys
    val R = data.map{case (k,v) => v}

    val Xinv = alg.inverse(alg.transpose(X))
    val Xpre = alg.transpose(X)
    val C1X = R.map(x => alg.mult(Xpre,alg.mult(Xinv,x)))

    val cov = C1X.map(x => outerProd(x,x)).reduce(_.assign(_,Functions.plus)).assign(Functions.div(n))

    println(cov)

    // compute OLS estimate of C for R = C * X
    //println("getting initial OLS estimate")
    //val C1X = R.map(r => X.mmul(Solve.solveLeastSquares(X,r)))

    //println(C1X.first())

    //val foo = C1X.first()
    //val duh = DoubleMatrix.ones(5)
    //val duh2 = DoubleMatrix.ones(5)
    //println(duh.mmul(duh2.transpose()))
    //println(foo.mmul(foo.transpose()))

    //val cov = C1X.map(x => outerProduct(x,x)).reduce(_.add(_)).rdiv(n)

    //println(cov)

    //println(Eigen.eigenvectors(cov))


//    // compute U using the SVD: [U S V] = svd(C * X)
//    println("computing SVD")
//    //val U = svd(C1X, k1, m,"basic")._1
    val eigs = new EigenvalueDecomposition(cov)
    println(eigs)
//
//    // project U back into C : C2 = U * U' * C
//    println("computing outer products")
//    val UC1 = U.zip(C1).map{case (x,y) => outerProd(x,y)}.reduce(_.assign(_,Functions.plus))
//    println("computing corrected estimate")
//    val C = U.map(x => alg.mult(alg.transpose(UC1),x))
//
//    // recompute U and V using the SVD: [U S V] = svd(C2)
//    println("computing SVD again")
//    val result = svd(C,k1, c,"basic")
//    val U2 = result._1
//    val V2 = result._2
//
//    // add keys back in
//    val out = data.map{case (k,v) => k}.zip(U2)
//
//    // print time series
//    printMatrix(V2, outputFileTxt)
//
//    // print images
//    for (ik <- 0 until k1) {
//      printToImage(out.map{case (k,v) => (k,v.get(ik)*100)}, w, h, d, outputFileImg + "-comp" + ik.toString() + ".png")
//    }

  }

}

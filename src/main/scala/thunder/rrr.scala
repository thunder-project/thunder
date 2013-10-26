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
import thunder.util.MatrixRDD
import cern.jet.math.Functions
import cern.colt.matrix.{DoubleMatrix1D, DoubleMatrix2D, DoubleFactory1D, DoubleFactory2D}
import cern.colt.matrix.linalg.Algebra
import scala.util.Random

object rrr {

  val factory2D = DoubleFactory2D.dense
  val factory1D = DoubleFactory1D.dense
  val alg = Algebra.DEFAULT

  def parseVector(line: String): ((Array[Int]), DoubleMatrix1D) = {
    var vec = line.split(' ').drop(3).map(_.toDouble)
    val inds = line.split(' ').take(3).map(_.toDouble.toInt) // xyz coords
    //val n = vec.length
    //val sortVals = vec.sorted
    //val mean = sortVals((n/4).toInt)
    val mean = vec.sum / vec.length
    vec = vec.map(x => (x - mean)/(mean + 0.1)) // time series
    vec = vec.map(x => x - (vec.sum / vec.length))
    return (inds,factory1D.make(vec))
  }

  def randomVector(index: Int, seed1: Int, k: Int) : DoubleMatrix1D ={
    val rand = new Random(index*seed1)
    return factory1D.make(Array.fill(k)(rand.nextDouble - 0.5))
  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try {
      op(p)
    } finally {
      p.close()
    }
  }

  def printToImage(rdd: RDD[(Array[Int],Double)], w: Int, h: Int, d: Array[Int], fileName: String): Unit = {
    for (id <- d) {
      val plane = rdd.filter(_._1(2) == id).map{case (k,v) => (k(0),k(1),v)}.toArray()
      val X = plane.map(_._1)
      val Y = plane.map(_._2)
      val RGB = plane.map(_._3)
      val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
      val raster = img.getRaster()
      (X,Y,RGB).zipped.foreach{case(x,y,rgb) => raster.setPixel(x-1, y-1, Array(rgb,rgb,rgb))}
      ImageIO.write(img, "png", new File(fileName + "-plane"+id.toString+".png"))
    }
  }

  def outerProd(vec1: DoubleMatrix1D, vec2: DoubleMatrix1D): DoubleMatrix2D = {
    val out = factory2D.make(vec1.size,vec2.size)
    alg.multOuter(vec1,vec2,out)
    return out
  }

  def printMatrix(data: DoubleMatrix2D, saveFile: String): Unit = {
    // print a DoubleMatrix2D to text by writing each row as a string
    val out = data.toArray()
    printToFile(new File(saveFile))(p => {
      out.foreach(x => p.println(x.mkString(" ")))
    })
  }

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
    val k = args(5).toInt

    System.setProperty("spark.executor.memory", "120g")
    val sc = new SparkContext(master, "rrr", System.getenv("SPARK_HOME"),
      List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))

    val data = sc.textFile(inputFileR).map(parseVector _).cache()
    val X = factory2D.make(sc.textFile(inputFileX).map(x => x.split(' ').map(_.toDouble)).toArray())

    println("getting dimensions")
    val w = data.map{case (k,v) => k(0)}.top(1).take(1)(0)
    val h = data.map{case (k,v) => k(1)}.top(1).take(1)(0)
    val d = data.filter{case (k,v) => (k(0) == 1) & (k(1) == 1)}.map{case (k,v) => k(2)}.toArray()

    println("initializing variables")
    val n = data.count().toInt
    val m = data.first()._2.size()
    val c = X.rows

    val k1 = 3
    var iter = 0
    val nIter = 10

    var u = factory2D.make(k,m)
    val seed1 = Random.nextInt*1000
    var v = data.map{case (k,v) => k(0) + (k(1)-1)*h}.map(randomVector(_,seed1,k))

    while (iter < nIter) {

      // goal is to solve R = C * X = V * U * X subject to rank(C) = k
      // by iteratively updating U and V with least squares

      println("starting" + iter.toString)

      // precompute inv(V' * V)
      val vinv = alg.inverse(v.map( x => outerProd(x,x)).reduce(_.assign(_,Functions.plus)))

      // update U using least squares row-wise as inv(V' * V) * V * R * pinv(X) (same as pinv(V) * R * pinv(X))
      u = alg.mult(data.map(_._2).zip(v.map (x => alg.mult(vinv,x))).map( x => outerProd(x._2,x._1)).reduce(_.assign(_,Functions.plus)),alg.transpose(alg.inverse(alg.transpose(X))))

      // clip negative values
      //u.assign(Functions.bindArg1(Functions.max,0))

      // precompute pinv(U * X)
      val ux = alg.mult(u,X)
      val uxinv = alg.transpose(alg.inverse(alg.transpose(ux)))

      // update V using least squares row-wise using R * pinv(U * X)
      v = data.map(_._2).map( x => alg.mult(alg.transpose(uxinv),x))

      // clip negative values
      //v = v.map(_.assign(Functions.bindArg1(Functions.max,0)))

      iter += 1

    }

    printMatrix(u,outputFileTxt + ".txt")

    for (i <- 0 until k) {
      val result = v.map(x => x.get(i))
      //val mx = result.top(1).take(1)(0)
      //printToImage(data.map(_._1).zip(result).map{case (k,v) => (k,(255*(v/mx)).toInt)}, w, h, d, outputFileImg + i.toString)
      val mx = result.top(1).take(1)(0)
      val mn = -result.map(x => -x).top(1).take(1)(0)
      printToImage(data.map(_._1).zip(result).map{case (k,v) => (k,(255*((v-mn)/(mn-mx))).toInt)},w,h,d,outputFileImg + i.toString)
    }

    System.exit(1)

  }

}

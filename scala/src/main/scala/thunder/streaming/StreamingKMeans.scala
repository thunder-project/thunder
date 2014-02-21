package thunder.streaming

import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.streaming._
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.util.Vector

import scala.util.Random.nextDouble
import scala.util.Random.nextGaussian

import thunder.util.Load

/**
 * K-means clustering on streaming data with support for
 * mini batch and forgetful algorithms
 */
class StreamingKMeans (
  var k: Int,
  var d: Int,
  var a: Double,
  var maxIterations: Int,
  var initializationMode: String)
  extends Serializable with Logging
{

  private type ClusterCentersAndCounts = Array[(Array[Double], Int)]

  def this() = this(2, 5, 1.0, 1, "gauss")

  /** Set the number of clusters to create (k). Default: 2. */
  private[streaming] def setK(k: Int): StreamingKMeans = {
    this.k = k
    this
  }

  /** Set the dimensionality of the data (d). Default: 5
    * TODO: if possible, set this automatically based on first data point
    */
  private[streaming] def setD(d: Int): StreamingKMeans = {
    this.d = d
    this
  }

  /**
   * Set the parameter alpha to determine the update rule.
   * If alpha = 1, perform "mini batch" KMeans, which treats all data
   * points equivalently. If alpha < 1, perform forgetful KMeans,
   * which uses a constant to weight old data
   * less strongly (with exponential weighting), e.g. 0.9 will
   * favor only recent data, whereas 0.1 will update slowly.
   * Weighting over time is per batch, so this algorithm implicitly
   * assumes an approximately constant number of data points per batch
   * Default: 1 (mini batch)
   */
  private[streaming] def setAlpha(a: Double): StreamingKMeans = {
    this.a = a
    this
  }

  /**
   * Set the number of iterations per batch of data
   */
  private[streaming] def setMaxIterations(maxIterations: Int): StreamingKMeans = {
    this.maxIterations = maxIterations
    this
  }

  /**
   * Set the initialization algorithm. Unlike batch KMeans, we
   * initialize randomly before we have seen any data. Options are "gauss"
   * for random Gaussian centers, and "pos" for random positive uniform centers.
   * Default: gauss
   */
  private[streaming] def setInitializationMode(initializationMode: String): StreamingKMeans = {
    if (initializationMode != "gauss" && initializationMode != "double") {
      throw new IllegalArgumentException("Invalid initialization mode: " + initializationMode)
    }
    this.initializationMode = initializationMode
    this
  }


  /**
   * Initialize random points for KMeans clustering
   */
  private[streaming] def initRandom(): StreamingKMeansModel = {

    val clusters = new Array[(Array[Double], Int)](k)
    for (ik <- 0 until k) {
      clusters(ik) = initializationMode match {
        case "gauss" => (Array.fill(d)(nextGaussian()), 0)
        case "pos" => (Array.fill(d)(nextDouble()), 0)
      }
    }
    new StreamingKMeansModel(clusters.map(_._1), clusters.map(_._2))
  }

  /**
   * Update KMeans clusters by doing training passes over data batch
   */
  private[streaming] def update(data: RDD[Array[Double]], model: StreamingKMeansModel): StreamingKMeansModel = {

    val centers = model.clusterCenters
    val counts = model.clusterCounts

    // do iterative KMeans updates on a batch of data
    for (i <- Range(0, maxIterations)) {
      // find nearest cluster to each point
      val closest = data.map(point => (model.predict(point), (point, 1)))

      // get sums and counts for updating each cluster
      val pointStats = closest.reduceByKey{
        case ((x1, y1), (x2, y2)) => (x1.zip(x2).map{case (x, y) => x + y}, y1 + y2)}
      val newPoints = pointStats.map{
        pair => (pair._1, (pair._2._1, pair._2._2))}.collectAsMap()

      a match {
        case 1 => for (newP <- newPoints) {
          // remove previous count scaling
          centers(newP._1) = centers(newP._1).map(x => x * counts(newP._1))
          // update sums
          centers(newP._1) = centers(newP._1).zip(newP._2._1).map{case (x, y) => x + y}
          // update counts
          counts(newP._1) += newP._2._2
          // rescale to compute new means (of both old and new points)
          centers(newP._1) = centers(newP._1).map(x => x / counts(newP._1))
        }
        case _ => for (newP <- newPoints) {
          // update centers with forgetting factor a
          centers(newP._1) = centers(newP._1).zip(newP._2._1.map(x => x / newP._2._2)).map{
            case (x, y) => x + a * (y - x)}
        }
      }
    }

    centers.foreach(x => print(Vector(x)))

    new StreamingKMeansModel(centers, counts)

  }

  private[streaming] def run(data: DStream[Array[Double]]): DStream[Int] = {
    var model = initRandom()
    data.foreachRDD(RDD => model = update(RDD, model))
    data.map(point => model.predict(point))
  }

}

/**
 * Top-level methods for calling Streaming KMeans clustering.
 */
object StreamingKMeans {

  private type ClusterCentersAndCounts = Array[(Array[Double], Int)]


  /**
   * Train a Streaming KMeans model
   *
   * @param k number of clusters to estimate
   * @param d data dimensionality
   * @param a update rule (1 mini batch, < 1 forgetful)
   * @param maxIterations maximum number of iterations per batch
   * @param initializationMode random initialization of points
   * @return a DStream of assignments of data points to clusters
   */
  def train(data: DStream[Array[Double]],
            k: Int,
            d: Int,
            a: Double,
            maxIterations: Int,
            initializationMode: String)
          : DStream[Int] =
  {
    new StreamingKMeans().setK(k)
                         .setD(d)
                         .setAlpha(a)
                         .setMaxIterations(maxIterations)
                         .setInitializationMode(initializationMode)
                         .run(data)
  }

  def main(args: Array[String]) {
    if (args.length < 4) {
      System.err.println("Usage: StreamingKMeans <master> <directory> <batchTime> <k> <d> <a> <maxIterations> <initializationMode>")
      System.exit(1)
    }

    val (master, directory, batchTime, k, d, a, maxIterations, initializationMode) = (
      args(0), args(1), args(2).toLong, args(3).toInt, args(4).toInt, args(5).toDouble, args(6).toInt, args(7))

    val ssc = new StreamingContext(master, "SimpleStreaming", Seconds(batchTime))

    ssc.checkpoint(System.getenv("CHECKPOINT"))

    val data = Load.loadStreamingData(ssc, directory)

    val assignments = StreamingKMeans.train(data, k, d, a, maxIterations, initializationMode)

    assignments.print()

    ssc.start()
  }

}

package thunder.streaming

import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.Logging

import org.scalatest.BeforeAndAfterAll
import org.scalatest.FunSuite
import java.io.File
import org.apache.commons.io.FileUtils
import com.google.common.io.Files
import scala.util.Random

import thunder.util.Load

class StreamingKMeansSuite extends FunSuite with BeforeAndAfterAll with Logging {

  import thunder.streaming.StreamingTestUtils._

  @transient private var ssc: StreamingContext = _

  override def beforeAll() {
    ssc = new StreamingContext("local", "test", Seconds(1))
  }

  override def afterAll() {
    ssc.stop()
    System.clearProperty("spark.driver.port")
  }

  test("single cluster") {

    // set parameters
    val k = 1 // number of clusters
    val d = 5 // number of dimensions
    val n = 100 // number of data points per batch
    val r = 0.05 // noise

    // create test directory and set up streaming data
    val testDir = Files.createTempDir()
    val data = Load.loadStreamingData(ssc, testDir.toString)

    // create and train KMeans model
    val KMeans = new StreamingKMeans().setK(k).setD(d)
    var model = KMeans.initRandom()
    data.foreachRDD(RDD => model = KMeans.update(RDD, model))
    ssc.start()

    // generate streaming data
    val rand = new Random(42)
    val centers = Array.fill(k)(Array.fill(d)(rand.nextGaussian()))

    Thread.sleep(200)
    for (i <- 0 until 10) {
      val samples = Array.tabulate(n)(i => Array.tabulate(d)(i => centers(i % k)(i) + rand.nextGaussian() * r).mkString(" "))
      val file = new File(testDir, i.toString)
      FileUtils.writeStringToFile(file, samples.mkString("\n") + "\n")
      Thread.sleep(Milliseconds(1000).milliseconds)
    }

    ssc.stop()

    // compare estimated center to actual
    assertSetsEqual(model.clusterCenters, centers, 0.1)
  }


}

import java.io.File
import spark.SparkContext
import spark.SparkContext._
import spark.util.Vector
import scala.collection.mutable.ArrayBuffer

/**
 * bisecting <master> <inputFile> <outputFile>
 *
 * bisecting k-means algorithm for divisive hierarchical clustering
 *
 */

object bisecting {

  def parseVector(line: String): (Int, (Double, Vector)) = {
    val nums = line.split(' ')
    //val k = nums(0).toDouble.toInt
    //val vec = nums.slice(1, nums.length).map(_.toDouble)
    val k3 = nums.slice(0,3).map(_.toDouble) // get xyz coordinates
    val k = (k3(0) + (k3(1) - 1)*2034 + (k3(2) - 1)*2034*1134).toInt // convert to linear index
    val vec = nums.slice(3,nums.length).map(_.toDouble)
    return (k, (1.toDouble, Vector(vec)))
  }

  def main(args: Array[String]) {

    if (args.length < 3) {
      System.err.println("Usage: bisecting <master> <inputFile> <outputFile> ")
      System.exit(1)
    }

    System.setProperty("spark.executor.memory", "120g")
    System.setProperty("spark.serializer", "spark.KryoSerializer")
    //System.setProperty("spark.default.parallelism", "50")
    val sc = new SparkContext(args(0), "hierarchical", System.getenv("SPARK_HOME"),
      List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))
    //sc.setCheckpointDir(System.getenv("CHECKPOINT"))

    val k = 20
    val data = sc.textFile(args(1)).map(parseVector _).cache()

    val clusters = ArrayBuffer(data)

    while (clusters.size < k) {
      val ind = 0
      val mid = (clusters(0).count() / 2).toInt
      val data1 = clusters(0).sample(false,0.5,1)
      val data2 = clusters(0).sample(false,0.5,2)
      clusters.remove(ind)
      clusters.append(data1)
      clusters.append(data2)
      println(clusters.map(_.count()))
      println(clusters.size)
    }

  }

 }

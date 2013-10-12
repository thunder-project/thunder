
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.util.Vector

object pls {

  def parseVector(line: String, mode: String): Vector = mode match {
    case "raw" => Vector(line.split(' ').map(_.toDouble))
    case "ca" => {
      Vector(line.split(' ').drop(3).map(_.toDouble))
    }
    case "dff" => {
      val vec = line.split(' ').drop(3).map(_.toDouble)
      val mean = vec.sum / vec.length
      Vector(vec.map(x => (x - mean)/(mean + 0.1)))
    }
    case _ => Vector(line.split(' ').map(_.toDouble))
  }

  def main(args: Array[String]) {

    if (args.length < 8) {
      System.err.println("Usage: pls <master> <inputFileX> <inputFileY> <outputFile> <inputMode> <k> <nSlices>")
      System.exit(1)
    }

    // collect arguments
    val master = args(0)
    val inputFileX = args(1)
    val inputFileY = args(2)
    val outputFile = args(3)
    val inputMode = args(4)
    val k = args(5).toDouble

    val sc = new SparkContext(master, "hierarchical", System.getenv("SPARK_HOME"),
      List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))

    val X = sc.textFile(inputFileX).map(parseVector (_,inputMode)).cache()
    val Y = Vector(sc.textFile(inputFileY).map(_.toDouble).collect().toArray)

    val startTime = System.nanoTime

    var w = X.map(x => x.dot(Y))
    val norm = scala.math.sqrt(w.map(w => w * w).reduce(_+_))
    w = w.map(w => w / norm)

    var t = X.zip(w).map{case (x,y) => x * y}.reduce(_+_)
    t = t / t.dot(t)

    var p = X.map(x => x.dot(t))

    var q = t.dot(Y)

    // update X

  }

}

/*** SimpleJob.scala ***/
import spark.SparkContext

object SimpleJob {
  def main(args: Array[String]) {
    val logFile = args(0) // Should be some file on your system
    val sc = new SparkContext("local", "Simple Job", System.getenv("SPARK_HOME"),
        List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))
    val logData = sc.textFile(logFile, 2).cache()
    val numAs = logData.filter(line => line.contains("a")).count()
    val numBs = logData.filter(line => line.contains("b")).count()
    println("Lines with a: %s, Lines with b: %s".format(numAs, numBs))
  }
}
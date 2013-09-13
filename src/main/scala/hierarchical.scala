import spark.SparkContext

object hierarchical {
  def main(args: Array[String]) {

    if (args.length < 2) {
      System.err.println("Usage: mantis <master> <directory> <batchTime> <nSlices>")
      System.exit(1)
    }
    
    val sc = new SparkContext("local", "hierarchical", System.getenv("SPARK_HOME"),
        List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))
    val logData = sc.textFile(logFile, 2).cache()
    val numAs = logData.filter(line => line.contains("a")).count()
    val numBs = logData.filter(line => line.contains("b")).count()
    println("Lines with a: %s, Lines with b: %s".format(numAs, numBs))
  }
}
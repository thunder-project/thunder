import spark.SparkContext._
import spark.streaming.{Seconds, StreamingContext}
import spark.streaming.StreamingContext._
import spark.util.Vector
import java.io._

object SimpleStreaming {

  def parseVector(line: String): (Int,Vector) = {
    val nums = line.split(' ') // split line into numbers
    val k = nums(0).toInt  // get index as key
    val v = Vector(nums.slice(1,nums.length).map(_ toDouble) :+ 1.toDouble) // get values, append 1 for counting
    return (k,v)
  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try { op(p) } finally { p.close() }
  }

  def printVector(rdd: spark.RDD[Vector]): Unit = {
    val data = rdd.collect().map(_.toString).map(x => x.slice(1,x.length-1)).map(_.replace(",",""))
    printToFile(new File("/groups/freeman/home/freemanj11/streamingresults/test.txt"))(p => {
      data.foreach(p.println)
    })
  }

  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: SimpleStreaming <master> <directory>")
      System.exit(1)
    }

    // create spark context
    val ssc = new StreamingContext(args(0), "SimpleStreaming", Seconds(2),
      System.getenv("SPARK_HOME"), List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))
    ssc.checkpoint("/groups/freeman/home/freemanj11/streamingresults/")

    // update state
    val updateFunc = (values: Seq[Vector], state: Option[Vector]) => {
      val currentState = values.foldLeft(Vector(0,0,0,0))(_+_) // ca, ephys, swim, count
      val previousState = state.getOrElse(Vector(0,0,0,0,0)) // baseLineTot, baseLine, diff1, diff2, count
      val count = currentState(3) + previousState(4) // update count
      var baseLineTot, baseLine, diff1, diff2 = 0.toDouble // initialize vars
      if (currentState(3) != 0.toDouble) { // if we have data, update states
        baseLineTot = currentState(0) + previousState(0)
        baseLine = baseLineTot / count
        if (currentState(2) == 0) { // if condition is 0
          diff1 = previousState(2) + ((currentState(0) - baseLine) / (baseLine + 0.1))
          diff2 = previousState(3)
        }
        else{
          diff2 = previousState(3) + ((currentState(0) - baseLine) / (baseLine + 0.1))
          diff1 = previousState(2)
        }
      }
      else { // otherwise use previous
        baseLineTot = previousState(0)
        baseLine = previousState(1)
        diff1 = previousState(2)
        diff2 = previousState(3)
      }
      Some(Vector(baseLineTot,baseLine,diff1,diff2,count))
    }

    // main streaming operations
    val lines = ssc.textFileStream(args(1)) // directory to monitor
    val dataStream = lines.map(parseVector _) // parse data
    val stateStream = dataStream.updateStateByKey(updateFunc)
    //dataStream.updateStateByKey(updateFunc).saveAsTextFiles("test") // update state
    //stateStream.print()
    val sortedStates = stateStream.transform(rdd => rdd.sortByKey(true)).map(x => Vector(x._2(2),x._2(3)))
    sortedStates.foreach(printVector _)
    sortedStates.print()
    //rdd => rdd.collect().foreach(println)
    //val output = stateStream.map{ x => (x._1,x._2(2)) }.transform(rdd => rdd.sortByKey(true)) // compute summary statistics
    //output.print()

    //wordDstream.reduceByKeyAndWindow(_+_,Seconds(10)).print()
    //lines.window(Seconds(10),Seconds(2)).map(parseVector _).map(x => x.sum).print()
    //println(partialSum)
    //val sums = lines.map(parseVector _).map(x => (1,1))
    //sums.print()
    //val counts = sums.updateStateByKey(updateFunc).map(_._2)
    //counts.print()
    ssc.start()
  }
}
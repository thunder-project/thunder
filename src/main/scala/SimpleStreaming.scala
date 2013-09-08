import spark.SparkContext._
import spark.streaming.{Seconds, StreamingContext}
import spark.streaming.StreamingContext._
import spark.util.Vector
import java.io._

object SimpleStreaming {

  def parseVector(line: String): (Int,Vector) = {
    val nums = line.split(' ') // split line into numbers
    val k = nums(0).toInt  // get index as key
    val vraw = Vector(nums.slice(1,nums.length).map(_ toDouble)) // ca, ephys, swim
    val v = Vector(vraw(0)*(1-vraw(2)),vraw(0)*vraw(2),1*(1-vraw(2)),1*vraw(2))
    return (k,v)
  }

  def getDiffs(vals: (Int,Vector)): (Int,Vector) = {
    val baseLine = (vals._2(0) + vals._2(1)) / (vals._2(2) + vals._2(3))
    val diff0 = ((vals._2(0) / vals._2(2)) - baseLine) / (baseLine + 0.1)
    val diff1 = ((vals._2(1) / vals._2(3)) - baseLine) / (baseLine + 0.1)
    return (vals._1, Vector(diff0,diff1))
  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try { op(p) } finally { p.close() }
  }

  def printVector(rdd: spark.RDD[Vector], saveFile: String): Unit = {
    val data = rdd.collect().map(_.toString).map(x => x.slice(1,x.length-1)).map(_.replace(",",""))
    printToFile(new File(saveFile))(p => {
      data.foreach(p.println)
    })
  }

  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: SimpleStreaming <master> <directory> <batchTime>")
      System.exit(1)
    }

    // create spark context
    System.setProperty("spark.executor.memory","120g")
    System.setProperty("spark.serializer", "spark.KryoSerializer")
    val ssc = new StreamingContext(args(0), "SimpleStreaming", Seconds(args(2).toLong),
      System.getenv("SPARK_HOME"), List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))
    ssc.checkpoint(System.getenv("CHECKPOINT"))

    // update state
    val updateFunc = (values: Seq[Vector], state: Option[Vector]) => {
      val currentState = values(0) // ca0, ca1, n0, n1
      val previousState = state.getOrElse(Vector(0,0,0,0))
      Some(currentState + previousState)
    }

    // main streaming operations
    val lines = ssc.textFileStream(args(1)) // directory to monitor
    val dataStream = lines.map(parseVector _) // parse data
    val stateStream = dataStream.reduceByKey(_+_,5).updateStateByKey(updateFunc)
    stateStream.print()
    //val sortedStates = stateStream.map(getDiffs _).transform(rdd => rdd.sortByKey(true)).map(x => Vector(x._2(0),x._2(1)))
    //sortedStates.print()

    //sortedStates.foreach(rdd => printVector(rdd,args(2)))

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
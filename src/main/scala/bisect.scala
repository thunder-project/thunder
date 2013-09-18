/**
 * bisect <master> <inputFile>
 *
 * efficient hierarchical clustering using the nearest neighbor chain algorithm
 * writes result to JSON for visualization in D3
 *
 */

 import java.io.File
 import spark.SparkContext
 import spark.SparkContext._
 import spark.util.Vector

 object bisect {

   def parseVector(line: String): (Int,(Double,Vector)) = {
     val nums = line.split(' ')
     val k = nums(0).toDouble.toInt
     val vec = nums.slice(1,nums.length).map(_.toDouble)
     //val k3 = nums.slice(0,3).map(_.toDouble) // get xyz coordinates
     //val k = (k3(0) + (k3(1) - 1)*2034 + (k3(2) - 1)*2034*1134).toInt // convert to linear index
     //val vec = nums.slice(3,nums.length).map(_.toDouble)
     return (k,(1.toDouble,Vector(vec)))
   }

   def main(args: Array[String]) {

     if (args.length < 3) {
        System.err.println("Usage: hierarchical <master> <inputFile> <outputFile> ")
        System.exit(1)
       }

       System.setProperty("spark.executor.memory","120g")
       System.setProperty("spark.serializer", "spark.KryoSerializer")
       //System.setProperty("spark.default.parallelism", "50")
       val sc = new SparkContext(args(0), "hierarchical", System.getenv("SPARK_HOME"),
           List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))
       sc.setCheckpointDir(System.getenv("CHECKPOINT"))
    }

}
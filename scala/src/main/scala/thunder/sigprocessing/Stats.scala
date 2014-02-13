package thunder.sigprocessing

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.util.Vector

import thunder.util.Load
import thunder.util.Load._


object Stats {

    def main(args: Array[String]) {
      if (args.length != 5) {
        println("Usage: Stats <master> <datafile> <outputdir> <mode>")
        System.exit(1)
      }
      val sc = new SparkContext(args(0), "Stats")
      val data = Load.loadData(sc, args(1))

      val dims = getDims(data)

      print(Vector(dims.map(_.toDouble)))

      val mode = args(3)
      if (mode == "mean") {
        data.mapValues(x => x.sum / x.length).keys.map(x => x(0)).foreach(println)
      }

    }

}

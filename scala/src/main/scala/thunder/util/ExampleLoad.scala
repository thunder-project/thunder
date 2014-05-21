package thunder.util

import org.apache.spark.SparkContext._
import org.apache.spark.{SparkContext, SparkConf}

object ExampleLoad {

  def main(args: Array[String]) {

    val master = args(0)

    val file = args(1)

    val conf = new SparkConf().setMaster(master).setAppName("ExampleLoad")

    val sc = new SparkContext(conf)

    val data = Load.fromBinaryWithKeys(sc, file, nKeys = 1)

    data.take(10).foreach{x =>
      println(x._1.mkString(" "))
      println(x._2.mkString(" "))
    }

    println(data.count())

  }
}

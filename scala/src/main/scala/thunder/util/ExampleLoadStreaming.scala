package thunder.util

import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}

object ExampleLoadStreaming {

  def main(args: Array[String]) {

    val master = args(0)

    val file = args(1)

    val batchTime = args(2).toLong

    val conf = new SparkConf().setMaster(master).setAppName("ExampleLoadStreaming")

    val ssc = new StreamingContext(conf, Seconds(batchTime))

    val data = LoadStreaming.fromBinary(ssc, file)

    data.map(v => v.mkString(" ")).print()

    data.foreachRDD(rdd => println(rdd.count()))

    ssc.start()

  }
}

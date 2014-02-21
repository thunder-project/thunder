/** Utilities for loading and parsing data */

package thunder.util

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.dstream.DStream

import scala.math.max
import scala.math.ceil

object Load {

  /**
   * Class for loading lines of a data file
   *
   * @param nKeys Number of keys per data point
   */
  case class DataLoader(nKeys: Int) {
    def get(line: String): Array[Double] = {
      val parts = line.split(' ')
      val value = parts.slice(nKeys, parts.length).map(_.toDouble)
      value

    }

    def getWithKeys(line: String): (Array[Int], Array[Double]) = {
      val parts = line.split(' ')
      val key = parts.slice(0,nKeys).map(_.toDouble.toInt)
      val value = parts.slice(nKeys, parts.length).map(_.toDouble)
      (key, value)

    }
  }

  /**
   * Class for pre processing data
   *
   * @param preProcessMethod Method of pre processing
   */
  case class DataPreProcessor(preProcessMethod: String) {
    def get(x: Array[Double]): Array[Double] = preProcessMethod match {
        case "raw" => x
        case "meanSubtract" =>
          val mean = x.sum / x.length
          x.map(_-mean)
        case "dff" =>
          val mean = x.sum / x.length
          return x.map{ x =>
            (x - mean) / (mean + 0.1) }
    }
  }

  /**
   * Get maximum dimensions of data based on keys in an RDD
   *
   * @param data: RDD of data points as key value pairs
   */
  def getDims(data: RDD[(Array[Int], Array[Double])]): Array[Int] = {
    Range(0, data.first()._1.length).map(x => data.map(_._1(x)).reduce(max)).toArray
  }

  /**
   * Convert subscript indexing to linear indexing in an RDD
   *
   * @param data RDD of data points as key value pairs
   * @param dims Dimensions for subscript indices
   * @return RDD of data points with linear indices as keys
   */
  def subToInd(data: RDD[(Array[Int], Array[Double])], dims: Array[Int]):
    RDD[(Int, Array[Double])] = dims.length match {
      case 1 => data.map{case (k, v) => (k(0), v)}
      case _ =>
        val dimprod = dims.map(x => Range(1, x + 1).reduce(_*_)).dropRight(1)
        data.map{case (k, v) =>
          (dimprod.zip(k.drop(1)).map{case (x, y) => (x - 1) * y}.sum + k(0), v)
      }
  }

  /**
   * Convert linear indexing to subscript indexing in an RDD
   *
   * @param data RDD of data points as key value pairs
   * @param dims Dimensions for subscript indices
   * @return RDD of data points with subscript indices as keys
   */
  def indToSub(data: RDD[(Int, Array[Double])], dims: Array[Int]):
    RDD[(Array[Int], Array[Double])] = dims.length match {
    case 1 => data.map{case (k, v) => (Array(k), v)}
    case _ =>
      val dimprod = dims.zip(Array.concat(Array(1), dims.map(x => Range(1, x + 1).reduce(_*_)).dropRight(1)))
      data.map{case (k, v) =>
        (dimprod.map{case (x, y) => ((ceil(k.toDouble / y) - 1) % x + 1).toInt}, v)
      }
  }

  /**
   * Convert subscript indexing to linear indexing in a DStream
   *
   * @param data DStream of data points as key value pairs
   * @param dims Dimensions for subscript indices
   * @return DStream of data points with linear indices as keys
   */
  def subToIndStreaming(data: DStream[(Array[Int], Array[Double])], dims: Array[Int]):
  DStream[(Int, Array[Double])] = dims.length match {
    case 1 => data.map{case (k, v) => (k(0), v)}
    case _ =>
      val dimprod = dims.map(x => Range(1, x + 1).reduce(_*_)).dropRight(1)
      data.map{case (k, v) =>
        (dimprod.zip(k.drop(1)).map{case (x, y) => (x - 1) * y}.sum + k(0), v)
      }
  }

  /**
   * Convert linear indexing to subscript indexing in a DStream
   *
   * @param data DStream of data points as key value pairs
   * @param dims Dimensions for subscript indices
   * @return DStream of data points with subscript indices as keys
   */
  def indToSubStreaming(data: DStream[(Int, Array[Double])], dims: Array[Int]):
  DStream[(Array[Int], Array[Double])] = dims.length match {
    case 1 => data.map{case (k, v) => (Array(k), v)}
    case _ =>
      val dimprod = dims.zip(Array.concat(Array(1), dims.map(x => Range(1, x + 1).reduce(_*_)).dropRight(1)))
      data.map{case (k, v) =>
        (dimprod.map{case (x, y) => ((ceil(k.toDouble / y) - 1) % x + 1).toInt}, v)
      }
  }

  /**
   * Load data from a text file with format
   * <t1> <t2> ...
   * where <t1> <t2> ... are the data values (Double)
   *
   * @param sc SparkContext
   * @param dir Directory to the input data files
   * @param PreProcessMethod Method for pre processing data (default = "raw")
   * @return An RDD of data, each with two elements: the first has a key (Array[Int]),
   *         and the second has the data values (an Array[Double])
   */
  def loadData(sc: SparkContext,
                       dir: String,
                       PreProcessMethod: String = "raw"): RDD[Array[Double]] = {
    val Loader = new DataLoader(0)

    if (PreProcessMethod != "raw") {
      val PreProcessor = new DataPreProcessor(PreProcessMethod)
      sc.textFile(dir).map(Loader.get).map(PreProcessor.get)
    } else {
      sc.textFile(dir).map(Loader.get)
    }
  }

  /**
   * Load keyed data from a text file with format
   * <k1> <k2> ... <t1> <t2> ...
   * where <k1> <k2> ... are keys (Int) and <t1> <t2> ... are the data values (Double)
   *
   * @param sc SparkContext
   * @param dir Directory to the input data files
   * @param PreProcessMethod Method for pre processing data (default = "raw")
   * @param nKeys Number of keys per data point (default = 3)
   * @return An RDD of data, each with two elements: the first has a key (Array[Int]),
   *         and the second has the data values (an Array[Double])
   */
  def loadDataWithKeys(sc: SparkContext,
               dir: String,
               PreProcessMethod: String = "raw",
               nKeys: Int = 3): RDD[(Array[Int], Array[Double])] = {
    val Loader = new DataLoader(nKeys)

    if (PreProcessMethod != "raw") {
      val PreProcessor = new DataPreProcessor(PreProcessMethod)
      sc.textFile(dir).map(Loader.getWithKeys).mapValues(PreProcessor.get)
    } else {
      sc.textFile(dir).map(Loader.getWithKeys)
    }
  }

  /**
   * Load streaming data from text files with format
   * <t1> <t2> ...
   * Where <t1> <t2> are the data values (Double)
   *
   * @param ssc StreamingContext
   * @param dir Directory to the input data files
   * @return A DStream of data, each with two elements: the first has the keys,
   *         and the second has the data values (an array of Double)
   */
  def loadStreamingData (ssc: StreamingContext,
                         dir: String): DStream[Array[Double]] = {
    val Loader = new DataLoader(0)
    ssc.textFileStream(dir).map(Loader.get)

  }

  /**
   * Load keyed streaming data from text files with format
   * <k1> <k2> ... <t1> <t2> ...
   * where <k1> <k2> ... are keys (Int) and <t1> <t2> ... are the data values (Double)
   * If multiple keys are provided (e.g. x, y, z), they are converted to linear indexing

   * @param ssc StreamingContext
   * @param dir Directory to the input data files
   * @param nKeys Number of keys per data point
   * @param dims Dimensions for converting subscript to linear indices
   * @return A DStream of data, each with two elements: the first has the keys,
   *         and the second has the data values (an array of Double)
   */
  def loadStreamingDataWithKeys (ssc: StreamingContext,
                         dir: String,
                         nKeys: Int = 1,
                         dims: Array[Int] = Array(1)): DStream[(Int, Array[Double])] = {
    val Loader = new DataLoader(nKeys)
    subToIndStreaming(ssc.textFileStream(dir).map(Loader.getWithKeys), dims)

  }

 }



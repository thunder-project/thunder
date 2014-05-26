package thunder.util.io

import org.apache.spark.mllib.regression.LabeledPoint
import java.nio.{ByteOrder, ByteBuffer}

// TODO add numerical format option to handle byte -> int vs byte -> float
/**
 * Class for loading lines of a data file
 * supporting a variety of formats
 *
 *
 * @param nKeys Number of integer keys per data point
 */
case class Parser(nKeys: Int = 0) {

  // TODO try to use 2 byte Int (unsigned Int)
  /**
   * Convert an Array[Byte] to Array[Int]
   * using a Java ByteBuffer
   */
  def byteArrayToIntArray(v: Array[Byte]): Array[Int] = {
    val buffer = ByteBuffer.wrap(v).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer()
    val intArray = new Array[Int](buffer.remaining())
    var t = 0
    while (buffer.remaining() > 0) {
      intArray(t) = buffer.get()
      t += 1
    }
    intArray
  }

  /** nKeys per string, but only keep the values */
  def get(line: String): Array[Double] = {
    val parts = line.split(' ')
    val value = parts.slice(nKeys, parts.length).map(_.toDouble)
    value
  }

  /** nKeys per Array[Int], but only keep the values */
  def get(line: Array[Int]): Array[Double] = {
    val value = line.slice(nKeys, line.length).map(_.toDouble)
    value
  }

  /** nKeys per Array[Byte], but only keep the values */
  def get(line: Array[Byte]): Array[Double] = {
    val value = byteArrayToIntArray(line).slice(nKeys, line.length).map(_.toDouble)
    value
  }

  /** nKeys per string, keep both */
  def getWithKeys(line: String): (Array[Int], Array[Double]) = {
    val parts = line.split(' ')
    val key = parts.slice(0,nKeys).map(_.toDouble.toInt)
    val value = parts.slice(nKeys, parts.length).map(_.toDouble)
    (key, value)
  }

  /** nKeys per Array[Int], keep both */
  def getWithKeys(line: Array[Int]): (Array[Int], Array[Double]) = {
    val key = line.slice(0,nKeys).map(_.toDouble.toInt)
    val value = line.slice(nKeys, line.length).map(_.toDouble)
    (key, value)
  }

  /** nKeys per Array[Byte], keep both */
  def getWithKeys(line: Array[Byte]): (Array[Int], Array[Double]) = {
    val tmp = byteArrayToIntArray(line)
    val key = tmp.slice(0,nKeys).map(_.toDouble.toInt)
    val value = tmp.slice(nKeys, line.length).map(_.toDouble)
    (key, value)
  }

  /** Single label per string, separated by a comma */
  def getWithLabels(line: String): LabeledPoint = {
    val parts = line.split(',')
    val label = parts(0).toDouble
    val features = parts(1).trim().split(' ').map(_.toDouble)
    LabeledPoint(label, features)
  }

  /** Single label per Array[Int], first value is the label */
  def getWithLabels(line: Array[Int]): LabeledPoint = {
    val label = line(0).toDouble
    val features = line.slice(1, line.length).map(_.toDouble)
    LabeledPoint(label, features)
  }

  /** Single label per Array[Byte], first value is the label */
  def getWithLabels(line: Array[Byte]): LabeledPoint = {
    val tmp = byteArrayToIntArray(line)
    val label = tmp(0).toDouble
    val features = tmp.slice(1, line.length).map(_.toDouble)
    LabeledPoint(label, features)
  }

}
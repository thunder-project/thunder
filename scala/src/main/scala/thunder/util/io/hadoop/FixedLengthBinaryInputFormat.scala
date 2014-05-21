package thunder.util.io.hadoop

import org.apache.hadoop.io.{LongWritable, BytesWritable}
import org.apache.hadoop.mapreduce.InputSplit
import org.apache.hadoop.mapreduce.RecordReader
import org.apache.hadoop.mapreduce.TaskAttemptContext
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat

object FixedLengthBinaryInputFormat {

  // TODO Find a way to pass this as an input argument
  def getRecordLength: Int = {
    8
  }

}

class FixedLengthBinaryInputFormat extends FileInputFormat[LongWritable, BytesWritable] {


  /**
   * This input format overrides computeSplitSize() to make sure that each split
   * only contains full records. Each InputSplit passed to FixedLengthBinaryRecordReader
   * will start at the first byte of a record, and the last byte will the last byte of a record.
   *
   */
  override def computeSplitSize(blockSize: Long, minSize: Long, maxSize: Long): Long = {

    val defaultSize = super.computeSplitSize(blockSize, minSize, maxSize)

    // If the default size is less than the length of a record, make it equal to it
    // Otherwise, make sure the split size is as close to possible as the default size,
    // but still contains a complete set of records, with the first record
    // starting at the first byte in the split and the last record ending with the last byte

    defaultSize match {
      case x if x < recordLength => recordLength.toLong
      case _ => (Math.floor(defaultSize / recordLength) * recordLength).toLong
    }
  }

  /**
   * Create a FixedLengthBinaryRecordReader
   *
   */
  override def createRecordReader(split: InputSplit, context: TaskAttemptContext):
    RecordReader[LongWritable, BytesWritable] = {
      new FixedLengthBinaryRecordReader
  }

  val recordLength = FixedLengthBinaryInputFormat.getRecordLength

}
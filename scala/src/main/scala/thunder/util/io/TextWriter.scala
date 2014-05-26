package thunder.util.io

import org.apache.spark.rdd.RDD
import java.util.Calendar
import java.io.File

case class TextWriter(directory: String) {

  def write(rdd: RDD[Double], fileName: String) = {
    val out = rdd.collect()
    val dateString = Calendar.getInstance().getTime.toString.replace(" ", "-").replace(":", "-")
    printToFile(new File(directory ++ File.separator ++ fileName ++ "-" ++ dateString ++ ".txt"))(p => {
      out.foreach(x => p.println("%.6f".format(x)))
    })
  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try {
      op(p)
    } finally {
      p.close()
    }
  }

}

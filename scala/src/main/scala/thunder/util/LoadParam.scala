package thunder.util

import scala.io.Source

object LoadParam {

  def fromText(file: String): ThunderParam = {
    val param = new ThunderParam()
    val lines = Source.fromFile(file).getLines()
    lines.foreach{l =>
      val parts = l.split(": ")
      val key = parts(0)
      val value = parts(1)
      param.set(key, value)
    }
    param
  }

}

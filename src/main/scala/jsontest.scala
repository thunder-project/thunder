import spark.SparkContext
import spark.SparkContext._
import spark.util.Vector
import scala.collection.mutable.ArrayBuffer
import cc.spray.json._
import cc.spray.json.DefaultJsonProtocol._

//

object jsontest {

  case class Cluster(var key: Int, var center: Array[Double], var children: Option[List[Cluster]])

  object MyJsonProtocol extends DefaultJsonProtocol {
    implicit val menuItemFormat: JsonFormat[Cluster] = lazyFormat(jsonFormat(Cluster, "key", "center", "children"))
  }

  import MyJsonProtocol._

  def main(args: Array[String]) {

    //val foo = new Item("j","test",Some(Item("j","test",Some(Item("j","test",None)))))
    //val foo = new Item("j","test",Some(Item("j","test",None)))

    val child1 = Cluster(1,Array(0.1,0.4),None)
    val child2 = Cluster(2,Array(0.25,0.25),None)
    val root = Cluster(0,Array(0.5,0.5),Some(List(child1,child2)))
    println(root.toJson.prettyPrint)

    val base = Cluster(0,Array(0.5,0.5),None)
    base.children = Some(List(child1,child2))
    println(base.toJson.prettyPrint)
  }


}




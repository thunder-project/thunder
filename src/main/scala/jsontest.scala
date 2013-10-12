import spray.json._

object jsontest {

  case class Cluster(var key: Int, var center: List[Map[String,Double]], var children: Option[List[Cluster]])

  object MyJsonProtocol extends DefaultJsonProtocol {
    implicit val menuItemFormat: JsonFormat[Cluster] = lazyFormat(jsonFormat(Cluster, "key", "center", "children"))
  }

  import MyJsonProtocol._

  def insert(node: Cluster, key: Int, children: List[Cluster]) {
    // recursively search cluster tree for desired key and insert children
    if (node.key == key) {
      node.children = Some(children)
    } else {
      if (node.children.getOrElse(0) != 0) {
        insert(node.children.get(0),key,children)
        insert(node.children.get(1),key,children)
      }
    }
  }

  def main(args: Array[String]) {

    val center1 = List(7.0,2.0,1.0)
    val child1 = Cluster(1,center1.zipWithIndex.map(x => Map("x"->x._2.toDouble,"y"->x._1)),None)
    val child2 = Cluster(2,center1.zipWithIndex.map(x => Map("x"->x._2.toDouble,"y"->x._1)),None)
    val child3 = Cluster(3,center1.zipWithIndex.map(x => Map("x"->x._2.toDouble,"y"->x._1)),None)
    val child4 = Cluster(4,center1.zipWithIndex.map(x => Map("x"->x._2.toDouble,"y"->x._1)),None)

    val base = Cluster(0,center1.zipWithIndex.map(x => Map("x"->x._2.toDouble,"y"->x._1)),None)
    insert(base,0,List(child1,child2))
    insert(base,1,List(child3,child4))

    println(base.toJson.prettyPrint)
  }


}




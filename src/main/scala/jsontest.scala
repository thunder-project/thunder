import cc.spray.json._

object jsontest {

  case class Cluster(var key: Int, var center: Array[Double], var children: Option[List[Cluster]])

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

    val child1 = Cluster(1,Array(0.1,0.4),None)
    val child2 = Cluster(2,Array(0.25,0.25),None)
    val child3 = Cluster(3,Array(0.3,0.6),None)
    val child4 = Cluster(4,Array(0.5,0.15),None)

    val base = Cluster(0,Array(0.5,0.5),None)
    insert(base,0,List(child1,child2))
    insert(base,1,List(child3,child4))

    println(base.toJson.prettyPrint)
  }


}




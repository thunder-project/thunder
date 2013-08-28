import com.typesafe.sbt.SbtStartScript

seq(SbtStartScript.startScriptForClassesSettings: _*)

name := "Thunder"

version := "1.0"

scalaVersion := "2.9.3"

libraryDependencies += "org.spark-project" %% "spark-core" % "0.7.3"

libraryDependencies += "org.spark-project" %% "spark-streaming" % "0.7.3"

resolvers ++= Seq(
  "Akka Repository" at "http://repo.akka.io/releases/",
  "Spray Repository" at "http://repo.spray.cc/")


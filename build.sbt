name := "Thunder"

version := "1.0"

scalaVersion := "2.9.3"

libraryDependencies += "org.spark-project" %% "spark-core" % "0.7.3"

libraryDependencies += "org.spark-project" %% "spark-streaming" % "0.7.3"

libraryDependencies ++= Seq(
    "org.eclipse.jetty.orbit"   % "javax.servlet"   % "3.0.0.v201112011016" % "provided"    artifacts Artifact("javax.servlet", "jar", "jar"),
    "org.eclipse.jetty"         % "jetty-webapp"    % "8.1.5.v20120716"     % "container"
)

resolvers ++= Seq(
  "Akka Repository" at "http://repo.akka.io/releases/",
  "Spray Repository" at "http://repo.spray.cc/")


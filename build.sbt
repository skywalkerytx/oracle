name := "oracle"

version := "0.1"

scalaVersion := "2.11.8"

scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature")

val doobieversion = "0.4.1"

resolvers += "Local Maven Repository" at "file:///home/nova/.m2/repository"

//doobie
libraryDependencies += "org.tpolecat" % "doobie-core_2.11" % doobieversion
libraryDependencies += "org.tpolecat" % "doobie-postgres_2.11" % doobieversion
libraryDependencies += "org.tpolecat" % "doobie-specs2_2.11" % doobieversion
libraryDependencies += "org.tpolecat" % "doobie-hikari_2.11" % doobieversion

libraryDependencies += "javax.mail" % "mail" % "1.4.7"
libraryDependencies += "joda-time" % "joda-time" % "2.9.6"

// https://mvnrepository.com/artifact/org.slf4j/slf4j-log4j12
//libraryDependencies += "org.slf4j" % "slf4j-log4j12" % "1.7.22"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "0.12",
  "org.scalanlp" %% "breeze-natives" % "0.12",

  "org.scalanlp" %% "breeze-viz" % "0.12"
)

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"

val dl4jversion = "0.7.2"
val nd4sversion = "0.7.2" // bit slower than dl4j

//dl4j
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % dl4jversion
libraryDependencies += "org.datavec" % "datavec-api" % dl4jversion
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % dl4jversion
//libraryDependencies += "org.nd4j" % "nd4j-cuda-8.0-platform" % dl4jversion
libraryDependencies += "org.nd4j" %% "nd4s" % nd4sversion

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs@_*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

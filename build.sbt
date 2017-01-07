name := "oracle"

version := "0.1"

scalaVersion := "2.11.8"

scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature")

//doobie
libraryDependencies += "org.tpolecat" % "doobie-core_2.11" % "0.3.0"
libraryDependencies += "org.tpolecat" % "doobie-contrib-postgresql_2.11" % "0.3.0"
libraryDependencies += "org.tpolecat" % "doobie-contrib-specs2_2.11"  % "0.3.0"
libraryDependencies += "org.tpolecat" % "doobie-contrib-hikari_2.11" % "0.3.0"

libraryDependencies += "javax.mail" % "mail" % "1.4.7"
libraryDependencies += "joda-time" % "joda-time" % "2.9.6"


// https://mvnrepository.com/artifact/org.slf4j/slf4j-log4j12
libraryDependencies += "org.slf4j" % "slf4j-log4j12" % "1.7.22"



//DL4j
//libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "0.7.2"
//libraryDependencies += "org.nd4j" % "nd4j-cuda-8.0" % "0.7.2"
//libraryDependencies += "org.datavec" % "datavec-api" % "0.7.2"

//resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository" //mxnet cache
//libraryDependencies += "ml.dmlc.mxnet" % "mxnet-full_2.11-linux-x86_64-gpu" % "0.1.2-SNAPSHOT" //seems it was forced to run on cpu, give up

assemblyMergeStrategy in assembly := {
 case PathList("META-INF", xs @ _*) => MergeStrategy.discard
 case x => MergeStrategy.first
}

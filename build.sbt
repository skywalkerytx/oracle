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
//libraryDependencies += "org.slf4j" % "slf4j-log4j12" % "1.7.22"

libraryDependencies  ++= Seq(
 "org.scalanlp" %% "breeze" % "0.12",
 "org.scalanlp" %% "breeze-natives" % "0.12",

 "org.scalanlp" %% "breeze-viz" % "0.12"
)

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"

resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository" //mxnet cache
libraryDependencies += "ml.dmlc.mxnet" % "mxnet-full_2.11-linux-x86_64-gpu" % "0.1.2-SNAPSHOT" //seems it was forced to run on cpu, give up

assemblyMergeStrategy in assembly := {
 case PathList("META-INF", xs @ _*) => MergeStrategy.discard
 case x => MergeStrategy.first
}

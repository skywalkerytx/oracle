name := "hello"

version := "0.1"

scalaVersion := "2.11.8"




libraryDependencies += "org.tpolecat" % "doobie-core_2.11" % "0.3.0"
libraryDependencies += "org.tpolecat" % "doobie-contrib-postgresql_2.11" % "0.3.0"
libraryDependencies += "org.tpolecat" % "doobie-contrib-specs2_2.11"  % "0.3.0"
libraryDependencies += "org.tpolecat" % "doobie-contrib-hikari_2.11" % "0.3.0"

libraryDependencies += "javax.mail" % "mail" % "1.4.7"
libraryDependencies += "joda-time" % "joda-time" % "2.9.6"


resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository"

libraryDependencies += "ml.dmlc.mxnet" % "mxnet-full_2.11-linux-x86_64-gpu" % "0.1.2-SNAPSHOT"


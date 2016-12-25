name := "hello"

version := "0.1"

scalaVersion := "2.11.8"



// https://mvnrepository.com/artifact/org.tpolecat/doobie-core_2.11
libraryDependencies += "org.tpolecat" % "doobie-core_2.11" % "0.3.0"

// https://mvnrepository.com/artifact/org.tpolecat/doobie-contrib-postgresql_2.11
libraryDependencies += "org.tpolecat" % "doobie-contrib-postgresql_2.11" % "0.3.0"
libraryDependencies += "org.tpolecat" % "doobie-contrib-specs2_2.11"  % "0.3.0"
libraryDependencies += "org.tpolecat" % "doobie-contrib-hikari_2.11" % "0.3.0"
// https://mvnrepository.com/artifact/javax.mail/mail
libraryDependencies += "javax.mail" % "mail" % "1.4.7"
libraryDependencies += "joda-time" % "joda-time" % "2.9.6"
// https://mvnrepository.com/artifact/ml.dmlc.mxnet/libmxnet-scala-linux-x86_64-gpu
libraryDependencies += "ml.dmlc.mxnet" % "libmxnet-scala-linux-x86_64-gpu" % "0.1.1"


// https://mvnrepository.com/artifact/org.apache.spark/spark-mllib_2.11
//libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.1.0"


import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}


/**
  * Created by nova on 16-12-19.
  */


object Main {

  def playground() = {
    import doobie.imports._

    import scalaz._
    import Scalaz._
    import scalaz.concurrent.Task

    val xa = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")

    import xa.yolo._

    sql"""select column_name, data_type
from INFORMATION_SCHEMA.COLUMNS where table_name = 'raw' and data_type = 'text' ;""".query[(String, String)].list.transact(xa).run.take(50000000).foreach(println)
  }

  def DailyUpdate() = {
    val mail = new EmailReader()
    mail.GetAttachments
    val zip = new ZipReader()
    zip.ReadAll
  }

  def main(args: Array[String]): Unit = {
    //DailyUpdate()
    val vec = new Vectorlize()
    //vec.GenMapping()
    //vec.GenVector()
    val res = vec.GenIndex()
    res.foreach {
      x =>
        print(x._1, ' ')
        x._2.foreach(x => print(x.toString + ' '))
        println()
    }

  }
}

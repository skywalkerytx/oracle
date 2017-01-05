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
    sql"""
       select
       industry,
       concept,
       area,
       macross,
       macdcross,
       kdjcross from raw


       """.query[(String, String, String, String, String, String)]
      .quick
      .unsafePerformSync


    val t = (1, 2, 3)


  }

  def main(args: Array[String]): Unit = {
    val vec = DailyUpdate

  }

  def DailyUpdate() = {
    val mail = new EmailReader()
    mail.GetAttachments
    val zip = new ZipReader()
    val vec = new Vectorlize()
    zip.ReadAll
    vec.GenMapping()
    vec.DataVector
  }
}

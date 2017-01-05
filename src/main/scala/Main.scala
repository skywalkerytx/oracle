import java.io._
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.zip.{ZipException, ZipFile}


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

    //val vec = DailyUpdate

  }

  def DailyUpdate() = {
    val today = Calendar.getInstance
    val ff = new SimpleDateFormat("yyyyMMdd")
    val filename = "data/holo/overview-push-" + ff.format(today.getTime) + ".zip"
    if (!new File(filename).exists && today.get(Calendar.HOUR_OF_DAY) >= 18) {
      //Today is not getted
      val mail = new EmailReader()
      mail.GetAttachments
      println("mail readed")
      val zip = new ZipReader()
      zip.ReadAll
      println("zip readed")
    }
    val vec = new Vectorlize()
    vec.GenMapping()
    vec.DataVector
  }
}

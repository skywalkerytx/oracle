import java.io._
import java.text.SimpleDateFormat
import java.util.Calendar


import doobie.imports._

import scalaz._
import Scalaz._
import scalaz.concurrent.Task
import scala.language.postfixOps
import doobie.contrib.postgresql.pgtypes._


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
    DailyUpdate(true)

  }

  def DailyUpdate(SavetoDatabase: Boolean = false) = {
    val today = Calendar.getInstance
    val ff = new SimpleDateFormat("yyyyMMdd")
    val filename = "data/holo/overview-push-" + ff.format(today.getTime) + ".zip"
    if (!new File(filename).exists && today.get(Calendar.HOUR_OF_DAY) >= 18) {
      println("update todays data")
      val mail = new EmailReader()
      mail.GetAttachments
      println("mail readed")
      val zip = new ZipReader()
      zip.ReadAll
      println("zip readed")
    }

    if (SavetoDatabase) {
      val xa = utils.GetDriverManagerTransactor
      val vec = new Vectorlize().GenMapping.DataBaseVector() // code date vector
      println(vec.toList.last.last.date)
      println("vector generated")
      vec.foreach {
        batch =>
          try {
            DailyQuery(batch, "vector").transact(xa).unsafePerformSync
          }
          catch {
            case ex: java.sql.BatchUpdateException => {
              val eex = ex.getNextException
              if (!eex.getMessage.contains("duplicate key")) {
                println("excetion when adding feature to database:")
                println(eex.getMessage)
                System.exit(2)
              }
            }
          }
      }
      val label = new Labels().DataBaseLabel()
        println("label generated")
      label.foreach {
        batch =>
          try {
            DailyQuery(batch, "label").transact(xa).unsafePerformSync
          }
          catch {
            case ex: java.sql.BatchUpdateException => {
              val eex = ex.getNextException
              if (!eex.getMessage.contains("duplicate key")) {
                println("excetion when adding label to database:")
                println(eex.getMessage)
                System.exit(2)
              }
            }
          }
      }
    }
  }

  def DailyQuery(data: List[utils.Features], table: String) = {
    val sql =
      """
        INSERT INTO """ +
        table +
        """
         (code,date,vector)
          VALUES(?,?,?)
      """
    Update[utils.Features](sql).updateMany(data)
  }
}

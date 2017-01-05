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
    val vec: Map[(String, String), Array[Float]] = new Vectorlize().GenMapping.DataVector // code date vector
    if (SavetoDatabase) {
      val xa = utils.GetDriverManagerTransactor
      try {

        DailyQuery(vec.map {
          vector =>
            utils.Features(vector._1._1, vector._1._2, vector._2)
        }.toList, "vector").transact(xa).unsafePerformSync
        println("here we go")
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
      try {
        val label = new Labels().DataBaseLabel
        DailyQuery(label, "label").transact(xa).unsafePerformSync
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
    (vec, label)
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

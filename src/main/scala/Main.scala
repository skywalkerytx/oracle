import java.io._
import java.text.SimpleDateFormat
import java.util.Calendar

import doobie.imports._

import scalaz._
import Scalaz._
import scalaz.concurrent.Task
import scala.language.postfixOps
import doobie.contrib.postgresql.pgtypes._
import doobie.syntax.string.Builder
import org.postgresql.util.PSQLException
import utils.Features

import scala.collection.immutable.Iterable


/**
  * Created by nova on 16-12-19.
  */


object Main {


  def main(args: Array[String]): Unit = {
    DailyUpdate(true)
  }

  def DailyQuery(tablename:String,Feature:Features) = {
    val query =
      s"""
        INSERT INTO
        ${tablename} (code,date,vector)
        VALUES(?,?,?)
      """
    Update[Features](query).toUpdate0(Feature)
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

      val vec = new Vectorlize().GenMapping.DataBaseVector().par
      vec.foreach{
        val xa = utils.GetDriverManagerTransactor
        vector=>
          try {
            DailyQuery("vector",vector).run.transact(xa).unsafePerformAsync _
          }
          catch {
            case ex:PSQLException =>
              if (!ex.getMessage.contains("duplicate key value violates unique constraint")) {
                println(ex.getMessage)
                println("Vector Error at:")
                println("    "+vector.code,vector.date)
              }
          }
      }
      val labels = new Labels().DataBaseLabel.par
      labels.foreach{
        label=>
          val xa = utils.GetDriverManagerTransactor
          try {
            DailyQuery("label",label).run.transact(xa).unsafePerformAsync _
          }
          catch {
            case ex:PSQLException =>
              if (!ex.getMessage.contains("duplicate key value violates unique constraint")) {
                println(ex.getMessage)
                println("Label Error at:")
                println("    "+label.code,label.date)
              }
          }
      }
    }

  }

}

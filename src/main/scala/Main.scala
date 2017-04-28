import java.io._
import java.text.SimpleDateFormat
import java.util.Calendar

import doobie.hikari.hikaritransactor.HikariTransactor
import doobie.imports.{ConnectionIO, _}

import scalaz._
import Scalaz._
import scalaz.concurrent.Task
import scala.language.postfixOps
import doobie.postgres.pgtypes._
import doobie.postgres.sqlstate.class23.UNIQUE_VIOLATION
//import org.apache.log4j.BasicConfigurator
import shapeless.HNil
import utils.Features
import org.joda.time._
import org.joda.time.format.{DateTimeFormat, DateTimeFormatter}


/**
  * Created by nova on 16-12-19.
  */

object Main {

  def main(args: Array[String]): Unit = {
    //BasicConfigurator.configure()

    //DailyUpdate(SavetoDatabase = true,SaveVector = false,SaveLabel = false)
    val SavetoDatabse = true
    val SaveVector = true
    val SaveLabel = true
    val UpdateAll = true

    DailyUpdate(SavetoDatabase = SavetoDatabse, SaveVector = SaveVector, SaveLabel = SaveLabel)
    //ValidationCheck.LabelCheck
    //new kdjpredict().mxnet
  }

  def DailyUpdate(SavetoDatabase: Boolean = false, SaveVector: Boolean = true, SaveLabel: Boolean = true, UpdateAll: Boolean = false) = {

    if (UpdateAll || ShouldDownload) {
      println("updating data from email: ")
      val mail = new EmailReader()
      mail.GetAttachments
      println("mail readed")
      val zip = new ZipReader()
      zip.ReadAll
      println("zip readed")
    }
    if (SavetoDatabase) {
      Cleanse(SaveVector = SaveVector, SaveLabel = SaveLabel)
      val xa: HikariTransactor[Task] = utils.GetHikariTransactor("daily-update-pool")
      val vec = new Vectorlize()
      val label = new Labels()

      if (SaveVector) {
        println("now inserting vector")
        vec.DataVector.foreach {
          feature =>
            //VectorQuery(feature)
            DailyQuery("vector", feature)
              .attemptSomeSqlState { case UNIQUE_VIOLATION => }.transact(xa).unsafePerformSync
        }
      }
      else println("note: Vector may not up-to-date")
      if (SaveLabel) {
        println("now inserting label")
        val labels = new Labels().DataBaseLabel
        labels.par.foreach {
          label =>
            DailyQuery("label", label)
              .attemptSomeSqlState { case UNIQUE_VIOLATION => }.transact(xa).unsafePerformSync
        }
      }
      else println("note: Label may not up-to-date")
    }
  }

  def Cleanse(SaveVector: Boolean = true, SaveLabel: Boolean = true) = {
    val xa = utils.GetDriverManagerTransactor
    val vector: Update0 = sql"delete from vector".update
    val label: Update0 = sql"delete from label".update
    if (SaveVector) {
      vector.run.transact(xa).unsafePerformIO
      println("note: Vector is cleaned")
    }
    if (SaveLabel) {
      label.run.transact(xa).unsafePerformIO
      println("note: Label is cleaned")
    }

  }

  def ShouldDownload: Boolean = {
    try {
      val today = new DateTime()
      var detectday = today
      val latest = utils.recursiveListFiles(new File("data/holo")).filter(_.getName.endsWith(".zip")).map(_.toString).sorted.last
      val fmt = DateTimeFormat.forPattern("yyyyMMdd")
      while (datetofile(detectday.toString(fmt)) != latest) {
        if (detectday == today) {
          if (today.hourOfDay.get >= 18)
            return true
        }
        else {
          if (detectday.dayOfWeek.get() < 6) {
            return true
          }
        }
        detectday = detectday.minusDays(1)
      }
      false
    }
    catch {
      case ex:Throwable => true
    }
  }

  def datetofile(s: String) = "data/holo/overview-push-" + s + ".zip"

  def DailyQuery(table: String, feature: Features): ConnectionIO[Int] = {
    val query = "INSERT INTO " + table + " (code,date,vector) VALUES(?,?,?)"

    Update[Features](query).toUpdate0(feature).run
  }

}

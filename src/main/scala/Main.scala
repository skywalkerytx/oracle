import java.io._
import java.text.SimpleDateFormat
import java.util.Calendar

import doobie.contrib.hikari.hikaritransactor.HikariTransactor
import doobie.imports.{ConnectionIO, _}

import scalaz._
import Scalaz._
import scalaz.concurrent.Task
import scala.language.postfixOps
import doobie.contrib.postgresql.pgtypes._
import doobie.contrib.postgresql.sqlstate.class23.UNIQUE_VIOLATION
import org.apache.log4j.BasicConfigurator
import shapeless.HNil
import utils.Features


/**
  * Created by nova on 16-12-19.
  */


object Main {

  def main(args: Array[String]): Unit = {
    BasicConfigurator.configure()
    DailyUpdate(SavetoDatabase = true)
  }

  def DailyUpdate(SavetoDatabase: Boolean = false, UpdateAll: Boolean = false) = {
    val today = Calendar.getInstance
    val ff = new SimpleDateFormat("yyyyMMdd")
    val filename = "data/holo/overview-push-" + ff.format(today.getTime) + ".zip"
    if ((UpdateAll) || (!utils.ZipValidation(filename) && today.get(Calendar.HOUR_OF_DAY) >= 17 && 0 < Calendar.DAY_OF_WEEK && Calendar.DAY_OF_WEEK < 7)) {
      println("update todays data: " + filename)
      val mail = new EmailReader()
      mail.GetAttachments
      println("mail readed")
      val zip = new ZipReader()
      zip.ReadAll
      println("zip readed")
    }
    val vec = new Vectorlize().DataBaseVector
    val labels = new Labels().DataBaseLabel
    if (SavetoDatabase) {
      val xa: HikariTransactor[Task] = utils.GetHikariTransactor
      xa.configure(hx => Task(hx setMaximumPoolSize 64)).unsafePerformSync
      println("now inserting vector")
      vec.par.foreach {
        feature =>
          //VectorQuery(feature)
          DailyQuery("vector", feature)
            .attemptSomeSqlState { case UNIQUE_VIOLATION => }.transact(xa).unsafePerformSync
      }
      println("now inserting label")
      labels.par.foreach {
        label =>
          DailyQuery("label", label)
            .attemptSomeSqlState { case UNIQUE_VIOLATION => }.transact(xa).unsafePerformSync
      }
      xa.shutdown.unsafePerformSync
    }
    (vec, labels)
  }


  def DailyQuery(table: String, feature: Features): ConnectionIO[Int] = {
    val query = "INSERT INTO " + table + " (code,date,vector) VALUES(?,?,?)"
    Update[Features](query).toUpdate0(feature).run
  }

}

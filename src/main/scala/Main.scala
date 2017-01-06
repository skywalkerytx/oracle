import java.io._
import java.text.SimpleDateFormat
import java.util.Calendar

import doobie.imports._

import scalaz._
import Scalaz._
import scalaz.concurrent.Task
import scala.language.postfixOps
import doobie.contrib.postgresql.pgtypes._
import doobie.contrib.postgresql.sqlstate.class23.UNIQUE_VIOLATION
import doobie.syntax.string.Builder
import org.apache.log4j.BasicConfigurator
import org.postgresql.util.PSQLException
import utils.Features

import scala.collection.immutable.Iterable
import scala.collection.parallel.ParIterable


/**
  * Created by nova on 16-12-19.
  */


object Main {


  def main(args: Array[String]): Unit = {
    utils.save(new Vectorlize().DataBaseVector,"data/vec.obj")
    utils.save(new Labels().DataBaseLabel,"data/label.obj")
    /*
    val vec = utils.load("data/vec.obj").asInstanceOf[scala.collection.parallel.ParIterable[Features]]
    val labels = utils.load("data/label.obj").asInstanceOf[ParIterable[Features]]
    vec.foreach {
      feature =>
        Insert("vector",feature)
    }
    labels.foreach{
      label =>
        Insert("label",label)
    }
    */
    //DailyUpdate(true)
    //Playground.DailyUpdate(true)
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
    }
  }

  def Insert(tablename: String, Feature: Features) = {
    val query = DailyQuery(tablename,Feature)
    val taskunit = for {
      xa <- utils.GetHikariTransactor
      _ <- xa.configure{
        hx =>
          Task.delay(hx.setMaximumPoolSize(65535))
      }
      a <- query.transact(xa).attemptSomeSqlState {
        case UNIQUE_VIOLATION => "Duplicate key, I really don't care about this"
      }.
      ensuring(xa.shutdown)
    } yield a
    taskunit.unsafePerformSync
  }
  def DailyQuery(tablename: String, Feature: Features): ConnectionIO[Features] = {
    val query =
      s"""
        INSERT INTO
        ${tablename} (code,date,vector)
        VALUES(?,?,?)
      """
    Update[Features](query).toUpdate0(Feature).withUniqueGeneratedKeys("code", "date")
  }

}

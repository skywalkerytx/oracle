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


/**
  * Created by nova on 16-12-19.
  */


object Main {


  def main(args: Array[String]): Unit = {
    BasicConfigurator.configure();
    DailyUpdate(true)
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
      println("sv")
      val vec = new Vectorlize()
        //.GenMapping
        .DataBaseVector()
      println("endsv")
      vec.foreach{
        vector=>
          Insert("vector", vector)
      }
      val labels = new Labels().DataBaseLabel
      labels.foreach{
        label=>
          Insert("label", label)
      }
    }
  }

  def Insert(tablename: String, Feature: Features) = {
    val query: ConnectionIO[Features] = DailyQuery(tablename, Feature)
    val xa = utils.GetDriverManagerTransactor
    query.attemptSomeSqlState {
      case UNIQUE_VIOLATION =>
    }.transact(xa).unsafePerformSync
    /*
    val taskunit = for {
      xa <- utils.GetHikariTransactor
      a <- query.transact(xa).attemptSomeSqlState {
        case UNIQUE_VIOLATION => "Duplicate key, I really don't care about this"
      }.
      ensuring(xa.shutdown)
    } yield a
    try {
      taskunit.unsafePerformSync
    }
    catch {
      case ex:Throwable =>
        println(ex.getMessage)
        println(tablename,Feature.code,Feature.date,Feature.vector.length)
        System.exit(5)
    }*/
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

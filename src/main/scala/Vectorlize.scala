/**
  * Created by nova on 16-12-25.
  */

import doobie.imports._

import scalaz._
import Scalaz._
import scalaz.concurrent.Task

class Vectorlize {

  val splitchar = '；'
  val toMap = List("industry", "concept", "area", "macross", "macdcross", "kdjcross")
  //.par
  val Queries: Map[String, Query0[String]] =
    Map(
      "concept" -> sql"select distinct concept from raw;".query[String],
      "industry" -> sql"select distinct industry from raw;".query[String],
      "area" -> sql"select distinct area from raw;".query[String],
      "macross" -> sql"select distinct macross from raw".query[String],
      "macdcross" -> sql"select distinct macdcross from raw".query[String],
      "kdjcross" -> sql"select distinct kdjcross from raw".query[String]
    )

  val all = 1000000000

  val codes: List[String] = {
    val xa = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")
    sql"select distinct code from raw ".query[String].list.transact(xa).unsafePerformSync.take(all)
  }

  val dates: List[String] = {
    val xa = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")
    sql"select distinct date from raw ".query[String].list.transact(xa).unsafePerformSync.take(all)
  }

  def GenMapping() = {
    toMap.foreach {
      col =>
        val xa = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")
        val result = Queries(col).list.transact(xa).unsafePerformSync.take(all).flatMap(_.split(splitchar)).distinct.zipWithIndex.map(tp2 => MappingClass(tp2._1, col, tp2._2))
        try {
          InsertInto(result).transact(xa).unsafePerformSync
        }
        catch {
          case ex: java.sql.BatchUpdateException => {
            val eex = ex.getNextException
            if (!eex.getMessage.contains("duplicate key"))
              System.exit(2)
          }
        }
    }
  }

  def InsertInto(data: List[MappingClass]) = {
    val query = "insert into Mapping(str,cat,id) values(?,?,?)"
    Update[MappingClass](query).updateMany(data)
  }

  def GetCount() = {
    val xa = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")
    val q1: Query0[(Int, String)] = sql"select count(1),cat from mapping group by cat ".query[(Int, String)]
    val r1 = q1.list.transact(xa).unsafePerformSync.take(toMap.length)
    val r2 = r1.map(_._1).sum
    (r2, r1)
  }

  def GenIndex(): Map[String, Array[Float]] = {
    dates.par.map {
      date =>
        val xa = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")
        (date,
          IndexByDate(date).list.
            transact(xa).unsafePerformSync.
            take(all).flatMap(x => x.toList).toArray)
    }.seq.toMap
  }

  def IndexByDate(date: String): Query0[(Float, Float, Float, Float, Float, Float, Float)] = {
    sql"select open,close,low,high,volume,money,delta from rawindex where index_date = $date order by index_code asc".query[(Float, Float, Float, Float, Float, Float, Float)]
  }

  def GetMapping()= {

  }

  def GenVector() = {

  }

  case class MappingClass(str: String, cat: String, id: Int)

  //case class IndexClass(open:Float,close:Float,low:Float,high:Float,volume:Float,money:Float,delta:Float)

}

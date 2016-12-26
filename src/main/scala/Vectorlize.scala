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
    sql"select distinct code from raw ".query[String].list.transact(xa).unsafePerformSync
  }

  val dates: List[String] = {
    val xa = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")
    sql"select distinct date from raw ".query[String].list.transact(xa).unsafePerformSync
  }

  def GenMapping() = {
    var gid = 0
    toMap.foreach {
      col =>
        val xa = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")
        import xa.yolo._
        val results: List[MappingClass] = Queries(col).list.transact(xa).unsafePerformSync.
          flatMap(_.split(splitchar)).distinct.map(str => MappingClass(str, col))
        try {
          InsertInto(results).transact(xa).unsafePerformSync
        }
        catch {
          case ex: java.sql.BatchUpdateException => {
            val eex = ex.getNextException
            if (!eex.getMessage.contains("duplicate key"))
              System.exit(2)
          }
        }
        for (result <- results) {
          UpdateGid(result.str, result.cat, gid).quick.unsafePerformSync
          gid = gid + 1
        }
    }
  }

  def UpdateGid(str: String, cat: String, gid: Int): Update0 = {
    sql"""
          UPDATE mapping
          SET
            gid = $gid
          WHERE
            str = $str AND cat = $cat
      """.update
  }

  def InsertInto(data: List[MappingClass]) = {
    val query = "insert into Mapping(str,cat) values(?,?)"
    Update[MappingClass](query).updateMany(data)
  }

  def GetCount() = {
    val xa = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")
    val q1: Query0[(Int, String)] = sql"select count(1),cat from mapping group by cat ".query[(Int, String)]
    val r1 = q1.list.transact(xa).unsafePerformSync.take(toMap.length)
    val r2 = r1.map(_._1).sum
    (r2, r1)
  }

  def GetIndex(): Map[String, Array[Float]] = {
    dates.par.map {
      date =>
        val xa = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")
        (date,
          IndexByDate(date).list.
            transact(xa).unsafePerformSync.
            flatMap(x => x.toList).toArray)
    }.seq.toMap
  }

  def IndexByDate(date: String): Query0[(Float, Float, Float, Float, Float, Float, Float)] = {
    sql"select open,close,low,high,volume,money,delta from rawindex where index_date = $date order by index_code asc".query[(Float, Float, Float, Float, Float, Float, Float)]
  }

  def GetMapping() = {
    val xa = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")
    val mappings: Map[(String, String), Int] = Mapping.list.transact(xa).unsafePerformSync.map { a => ((a._1, a._2), a._3) }.toMap
    RawMap.list.transact(xa).unsafePerformSync.map {
      rmc =>
        val key = (rmc.code, rmc.date)
        val mappingarray = new Array[Float](277)
        val tomap = rmc.productIterator.toList
        //val toMap = List("industry", "concept", "area", "macross", "macdcross", "kdjcross")
        for (i <- 2 until tomap.length) {
          val cat = toMap(i - 2)
          tomap(i).asInstanceOf[String].split(splitchar).foreach {
            str =>
              mappingarray(mappings(str, cat)) = 1
          }
        }
        (key, mappingarray)
    }
  }

  def Mapping: Query0[(String, String, Int)] = {
    sql"select str,cat,gid from mapping".query[(String, String, Int)]
  }

  def RawMap: Query0[RawMapClass] = {
    sql"select code,date,industry,concept,area,macross,macdcross,kdjcross from raw".query[RawMapClass]
  }

  def GenVector() = {

  }

  case class RawMapClass(code: String, date: String, industry: String, concept: String, area: String, macross: String, macdcross: String, kdjcross: String)

  case class MappingClass(str: String, cat: String)

  //case class IndexClass(open:Float,close:Float,low:Float,high:Float,volume:Float,money:Float,delta:Float)

}

/**
  * Created by nova on 16-12-25.
  */

import doobie.imports._

import scalaz._
import Scalaz._
import scalaz.concurrent.Task

import utils.Key, utils.codes, utils.dates

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
          UpdateGid(result.str, result.cat, gid).run.transact(xa).unsafePerformSync
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

  def GenVector(): Map[(String, String), Array[Float]] = {
    val xa = utils.GetDriverManagerTransactor
    val index: Map[String, Array[Float]] = GetIndex
    val mapping: Map[Key, Array[Float]] = GetMapping
    val concept: Map[String, Array[Float]] = GetConcept
    GetRaw.list.transact(xa).unsafePerformSync.par.map {
      raw =>
        (
          (raw.code, raw.date),
          Array(raw.op, raw.mx, raw.mn, raw.clse, raw.aft, raw.bfe, raw.amp, raw.vol,
            raw.market, raw.market_exchange, raw.on_board, raw.total, raw.zt, raw.dt, raw.shiyinlv, raw.shixiaolv, raw.shixianlv,
            raw.shijinglv, raw.ma5, raw.ma10, raw.ma20, raw.ma30, raw.ma60,
            raw.macddif, raw.macddea, raw.macdmacd, raw.k, raw.d, raw.j, raw.berlinmid, raw.berlinup, raw.berlindown,
            raw.psy, raw.psyma, raw.rsi1, raw.rsi2, raw.rsi3, raw.zhenfu, raw.volratio
          ) ++ index(raw.date) ++ concept(raw.date) ++ mapping(Key(raw.code, raw.date))
        )
    }.seq.toMap
  }

  def GetConcept: Map[String, Array[Float]] = {
    dates.par.map {
      date =>
        val xa = utils.GetDriverManagerTransactor
        (date,
          ConceptByDate(date).list.transact(xa).unsafePerformSync.
            flatMap(x => x.toList).toArray
        )
    }.seq.toMap
  }

  def ConceptByDate(date: String): Query0[(Float, Float, Float, Float, Float, Float, Float)] = {
    sql"select amount,uppercent,downpercent,drawpercent,amp,wamp,aprofit from rawconcept where date = $date order by concept asc".query[(Float, Float, Float, Float, Float, Float, Float)]
  }

  def GetIndex(): Map[String, Array[Float]] = {
    dates.par.map {
      date =>
        val xa = utils.GetDriverManagerTransactor
        (date,
          IndexByDate(date).list.
            transact(xa).unsafePerformSync.
            flatMap(x => x.toList).toArray)
    }.seq.toMap
  }

  def IndexByDate(date: String): Query0[(Float, Float, Float, Float, Float, Float, Float)] = {
    sql"select open,close,low,high,volume,money,delta from rawindex where index_date = $date order by index_code asc".query[(Float, Float, Float, Float, Float, Float, Float)]
  }

  def GetMapping(): Map[Key, Array[Float]] = {
    val xa = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")
    val mappings: Map[(String, String), Int] = Mapping.list.transact(xa).unsafePerformSync.map { a => ((a._1, a._2), a._3) }.toMap
    RawMap.list.transact(xa).unsafePerformSync.map {
      rmc =>
        val key = Key(rmc.code, rmc.date)
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
    }.toMap
  }

  def Mapping: Query0[(String, String, Int)] = {
    sql"select str,cat,gid from mapping".query[(String, String, Int)]
  }

  def RawMap: Query0[RawMapClass] = {
    sql"select code,date,industry,concept,area,macross,macdcross,kdjcross from raw".query[RawMapClass]
  }

  def GetRaw: Query0[Raw] = {
    sql"""
       select
        code,date,
                 op,mx,mn,clse,
                 aft,bfe,amp,vol,
                 market,market_exchange,
                 on_board,total,
                 zt,dt,
                 shiyinlv,shixiaolv,shixianlv,shijinglv,
                 ma5,ma10,ma20,ma30,ma60,
                 macddif,macddea,macdmacd,
                 k,d,j,
                 berlinmid,berlinup,berlindown,
                 psy,psyma,rsi1,rsi2,rsi3,zhenfu,volratio
       from
        raw
      """.query[Raw]
  }

  case class Raw(code: String, date: String, op: Float, mx: Float, mn: Float, clse: Float,
                 aft: Float, bfe: Float, amp: Float, vol: Float,
                 market: Float, market_exchange: Float, on_board: Float, total: Float,
                 zt: Float, dt: Float, shiyinlv: Float, shixiaolv: Float, shixianlv: Float, shijinglv: Float,
                 ma5: Float, ma10: Float, ma20: Float, ma30: Float, ma60: Float,
                 macddif: Float, macddea: Float, macdmacd: Float,
                 k: Float, d: Float, j: Float,
                 berlinmid: Float, berlinup: Float, berlindown: Float,
                 psy: Float, psyma: Float, rsi1: Float, rsi2: Float, rsi3: Float, zhenfu: Float, volratio: Float
                )

  case class RawMapClass(code: String, date: String, industry: String, concept: String, area: String, macross: String, macdcross: String, kdjcross: String)

  case class MappingClass(str: String, cat: String)


  //case class IndexClass(open:Float,close:Float,low:Float,high:Float,volume:Float,money:Float,delta:Float)

}

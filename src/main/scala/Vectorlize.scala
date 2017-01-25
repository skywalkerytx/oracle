/**
  * Created by nova on 16-12-25.
  */

import doobie.imports._

import scalaz._
import Scalaz._
import scalaz.concurrent.Task
import utils.{Features, Key, codes, dates}

import scala.collection.immutable.{Iterable, Seq}
import doobie.contrib.postgresql.sqlstate.class23.UNIQUE_VIOLATION

import scala.collection.parallel.immutable.ParSeq

import breeze.linalg._
import breeze.numerics._

class Vectorlize {

  val splitchar = '；'
  val toMap = List("industry", "concept", "area", "macross", "macdcross", "kdjcross")
  val xa = utils.GetHikariTransactor("vectorlize-pool")

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

  def dVector: ParSeq[Features] = {
    GenMapping
    val index = GetIndex
    val mapping = GetMapping
    val concept = GetConcept
    val raw= GetRaw.list.transact(xa).unsafePerformSync.par.map{
      raw =>
        (
          Key(raw.code, raw.date),
          DenseVector(raw.op, raw.mx, raw.mn, raw.clse, raw.aft, raw.bfe, raw.amp, raw.vol,
            raw.market, raw.market_exchange, raw.on_board, raw.total, raw.zt, raw.dt, raw.shiyinlv, raw.shixiaolv, raw.shixianlv,
            raw.shijinglv, raw.ma5, raw.ma10, raw.ma20, raw.ma30, raw.ma60,
            raw.macddif, raw.macddea, raw.macdmacd, raw.k, raw.d, raw.j, raw.berlinmid, raw.berlinup, raw.berlindown,
            raw.psy, raw.psyma, raw.rsi1, raw.rsi2, raw.rsi3, raw.zhenfu, raw.volratio
          )
        )
    }.toArray
    println(s"doing D now. :${raw.length}")
    (1 until raw.length).par.map(
      idx =>
        Features(raw(idx)._1.code,raw(idx)._1.date,
          (raw(idx)._2-raw(idx-1)._2).data
          ++index(raw(idx)._1.date)
          ++concept(raw(idx)._1.date)
          ++mapping(raw(idx)._1)
        )
    )
  }

  def DataVector: ParSeq[Features] = {
    val index: Map[String, Array[Float]] = GetIndex
    val mapping: Map[Key, Array[Float]] = GetMapping
    val concept: Map[String, Array[Float]] = GetConcept
    GetRaw.list.transact(xa).unsafePerformSync.par.map {
      raw =>
        Features(
          raw.code, raw.date,
          Array(raw.op, raw.mx, raw.mn, raw.clse, raw.aft, raw.bfe, raw.amp, raw.vol,
            raw.market, raw.market_exchange, raw.on_board, raw.total, raw.zt, raw.dt, raw.shiyinlv, raw.shixiaolv, raw.shixianlv,
            raw.shijinglv, raw.ma5, raw.ma10, raw.ma20, raw.ma30, raw.ma60,
            raw.macddif, raw.macddea, raw.macdmacd, raw.k, raw.d, raw.j, raw.berlinmid, raw.berlinup, raw.berlindown,
            raw.psy, raw.psyma, raw.rsi1, raw.rsi2, raw.rsi3, raw.zhenfu, raw.volratio
          )
            ++ index(raw.date)
            ++ concept(raw.date)
            ++ mapping(Key(raw.code, raw.date))
        )
    }
  }

  def GetConcept: Map[String, Array[Float]] = {
    dates.par.map {
      date =>
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
    GenMapping
    val mappings: Map[(String, String), Int] = Mapping.list.transact(xa).unsafePerformSync.map { a => ((a._1, a._2), a._3) }.toMap
    RawMap.list.transact(xa).unsafePerformSync.map {
      rmc =>
        val key = Key(rmc.code, rmc.date)
        val mappingarray = new Array[Float](mappings.length)
        val tomap = rmc.productIterator.toList
        //val toMap = List("industry", "concept", "area", "macross", "macdcross", "kdjcross")
        for (i <- 2 until tomap.length) {
          val cat = toMap(i - 2)
          tomap(i).asInstanceOf[String].split(splitchar).foreach {
            str =>
              mappingarray(mappings(str, cat) - 1) = 1
          }
        }
        (key, mappingarray)
    }.toMap
  }

  def GenMapping: Vectorlize = {
    var gid = 0
    toMap.foreach {
      col =>
        val results: List[MappingClass] = Queries(col).list.transact(xa).unsafePerformSync.
          flatMap(_.split(splitchar)).distinct.map(str => MappingClass(str, col))
        InsertInto(results).transact(xa)
          .attemptSomeSqlState { case UNIQUE_VIOLATION => }
          .unsafePerformSync
    }
    this
  }

  def InsertInto(data: List[MappingClass]) = {
    val query = "insert into Mapping(str,cat) values(?,?)"
    Update[MappingClass](query).updateMany(data)
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

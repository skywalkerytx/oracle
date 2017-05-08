import utils.Features
import GlobalConfig.DaystoPredict

import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.mutable.ParArray
import doobie.postgres.pgtypes._
import org.postgresql.util.PSQLException

/**
  * Created by nova on 16-12-27.
  */
class Labels {

  import doobie.imports._
  import utils.Key, utils.codes, utils.dates

  val amp = 1.03
  val rise = 1
  val fall = 0
  val xa = utils.GetHikariTransactor("label-pool")

  def CheckB(a:rawlabel,b:rawlabel) = {
    if(a.close*amp<=b.mx)
      true
    else
      false
  }

  def CheckA(a:rawlabel,b:rawlabel) = {
    if(a.op*amp<=b.mx)
      true
    else
      false
  }

  def Real(key: utils.Key): Query0[rawlabel] = {

    sql"SELECT op,mx,clse FROM raw WHERE code= ${key.code} AND date = ${key.date}".query[rawlabel]
  }

  def LabelA = GenLabel(CheckA)

  def LabelB = GenLabel(CheckB)

  def ResonaceLabel: ParArray[(String, String, Int)] = {
    val codes = CodeAvailable.list.transact(xa).unsafePerformSync
    codes.map {
      code =>
        val dates = DateAvailable(code).list.transact(xa).unsafePerformSync.toArray.par
        dates.map {
          date =>
            val tomorrow: Array[Int] = getResonance(code, date).list.transact(xa).unsafePerformSync.toArray
            (code, date, tomorrow)
        }.filter(_._3.length > 0).map(x => (x._1, x._2, x._3(0)))
    }.reduce {
      (x, y) =>
        x ++ y
    }
  }

  def CodeAvailable:Query0[String] =
    sql"""
       SELECT
         DISTINCT code
       FROM (
          SELECT
           code,
           count(1) as cc
          FROM
           raw
          GROUP BY
           code
          ) AS countbycode
       WHERE
           countbycode.cc >3;
       """.query[String]

  def DateAvailable(code:String):Query0[String] = {
    sql"select date from raw where code = $code order by date asc".query[String]
  }

  def getResonance(code: String, date: String): Query0[Int] = {
    sql"""
                 SELECT
         CASE
           WHEN macdcross = '金叉' AND kdjcross = '金叉' THEN 0
           WHEN macdcross = '金叉' AND kdjcross = '死叉' THEN 1
           WHEN macdcross = '金叉' AND kdjcross = '' THEN 2
           WHEN macdcross = '死叉' AND kdjcross ='金叉' THEN 3
           WHEN macdcross = '死叉' AND kdjcross = '死叉' THEN 4
           WHEN macdcross = '死叉' AND kdjcross = ''THEN 5
           WHEN macdcross = '' AND kdjcross = '金叉'THEN 6
           WHEN macdcross = '' AND kdjcross = '死叉' THEN 7
           WHEN macdcross = '' AND kdjcross = ''THEN 8
           ELSE 9 END
             AS Resonance
                 FROM
                  raw
                  WHERE code= $code AND date > $date
                  ORDER BY date ASC
                  LIMIT 1
              """.query[Int]
  }

  def GenLabel(checkfunc: (rawlabel, rawlabel) => Boolean): Map[Key, Int] = {
    val codes = CodeAvailable.list.transact(xa).unsafePerformSync
    codes.map{
      code=>
        val dates = DateAvailable(code).list.transact(xa).unsafePerformSync.toArray.par
        val reals= dates.map{
          date=>
              Real(Key(code,date)).unique.transact(xa).unsafePerformSync
        }
        val buffer = new ArrayBuffer[(String,String,Int)]()
        for ( i <- 0 until reals.length -(DaystoPredict+1)) {
          val flag = (0 until DaystoPredict).map{
            delta=>
              checkfunc(reals(i + 1), reals(i + delta + 2))
          }.reduce{
            (day1,day2)=>
              day1||day2
          }
          val value = if (flag) 1 else 0
          buffer.append((code,dates(i),value))
        }
        buffer
    }.reduce{
      (x,y) =>
      x++y
    }.map(row=>(Key(row._1,row._2),row._3)).toMap
  }

  def DataBaseLabel: Array[Features] = {
    val la = LabelA
    //val la = ResonaceLabel
    val lb = LabelB
    la.keys.map(key => utils.Features(key.code, key.date, Array(la(key), lb(key))))
  }.toArray

  case class rawlabel(op:Float,mx:Float,close:Float)

}

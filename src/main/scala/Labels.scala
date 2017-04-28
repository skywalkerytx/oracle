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

  def Real(key:utils.Key):Query0[rawlabel] = {

    sql"select op,mx,clse from raw where code= ${key.code} and date = ${key.date}".query[rawlabel]
  }

  def LabelA = GenLabel(CheckA)

  def LabelB = GenLabel(CheckB)

  def GenLabel(checkfunc:(rawlabel,rawlabel) => Boolean) = {
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
    val lb = LabelB
    la.keys.map(key => utils.Features(key.code, key.date, Array(la(key), lb(key))))
  }.toArray

  case class rawlabel(op:Float,mx:Float,close:Float)

}

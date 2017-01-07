import utils.Features

import scala.collection.immutable.IndexedSeq

/**
  * Created by nova on 16-12-27.
  */
class Labels {

  import doobie.imports._
  import utils.Key, utils.codes, utils.dates

  val amp = 1.03
  val rise = 1
  val fall = 0
  val xa = utils.GetHikariTransactor

  def LabelA(delta: Int = 2): Map[Key, Int] = GenLabel(this.checkA, delta)

  def GenLabel(CheckFunction: (Array[Float], Array[Float]) => Int, delta: Int): Map[Key, Int] = {
    val raw = query.list.transact(xa).unsafePerformSync.groupBy {
      _._1
    }
    codes.par.map {
      code =>
        val data = raw(code).map {
          case (code: String, date: String, op: Float, mx: Float, mn: Float, clse: Float) =>
            (date, Array(op, mx, mn, clse))
        }.sortBy(_._1)
        0 until data.length - delta map {
          i =>
            (Key(code, data(i)._1), CheckFunction(data(i + 1)._2, data(i + delta)._2))
        }
    }.reduce {
      (a, b) =>
        a ++ b
    }.toMap
  }

  def query: Query0[(String, String, Float, Float, Float, Float)] = sql"select code,date,op,mx,mn,clse from raw".query[(String, String, Float, Float, Float, Float)]

  def LabelB(delta: Int = 2): Map[Key, Int] = GenLabel(this.checkB, delta)

  def checkA(day1: Array[Float], day2: Array[Float]): Int = {
    //compare d+1 & d+2
    if (day1(0) * amp >= day2(1))
      rise
    else
      fall
  }

  def checkB(day1: Array[Float], day2: Array[Float]): Int = {
    if (day1(3) * amp >= day2(1))
      rise
    else
      fall
  }

  def DataBaseLabel: Array[Features] = {
    val la = LabelA()
    val lb = LabelB()
    la.keys.map(key => utils.Features(key.code, key.date, Array(la(key), lb(key))))
  }.toArray

  case class rawlabel(date: String, data: Array[Float])

}

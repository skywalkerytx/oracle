import doobie.hikari.hikaritransactor.HikariTransactor
import doobie.imports.{ConnectionIO, _}

import scalaz._
import Scalaz._
import scalaz.concurrent.Task
import scala.language.postfixOps
import doobie.postgres.pgtypes._
import doobie.postgres.sqlstate.class23.UNIQUE_VIOLATION
import shapeless.HNil


/**
  * Created by nova on 17-1-1.
  */
object ValidationCheck {

  def LabelCheck = {
    val xa = utils.GetHikariTransactor("LabelCheck-pool")
    val labels = lq.list.transact(xa).unsafePerformSync.toMap
    val codes = labels.keys.map(_.code).toList.distinct
    var count = 0
    codes.foreach {
      code =>
        val dates = dq(code).list.transact(xa).unsafePerformSync
        val realone = (0 until dates.length - 2).par.map {
          i =>
            (i, rq(utils.Key(code, dates(i))).unique.transact(xa).unsafePerformSync, code, dates(i))
        }
        count = count + realone.count {
          real =>
            try {
              val label = labels(utils.Key(code, dates(real._1)))
              val tom = realone(real._1 + 1)._2._1
              val dft = realone(real._1 + 2)._2._2
              if ((label == 0 && tom * 1.03 <= dft))
                true
              else
                false
              if ((label == 1 && tom * 1.03 > dft))
                true
              else
                false
              false
            }
            catch {
              case _: Throwable => false
            }
        }
    }
    println("mismatch label detected:", count)
  }

  def lq: Query0[(utils.Key, Int)] = {
    sql"select code,date,vector[1] from label".query[(utils.Key, Int)]
  }

  def rq(key: utils.Key): Query0[(Float, Float)] = {
    sql"select op,mx from raw where code= ${key.code} and date = ${key.date}".query[(Float, Float)]
  }

  def dq(code: String): Query0[String] = {
    sql"select date from raw where code = $code order by date asc".query[String]
  }
}

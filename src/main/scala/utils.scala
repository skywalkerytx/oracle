import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import java.util.zip.{ZipException, ZipFile}

import doobie.imports._

import scalaz._
import Scalaz._
import scalaz.concurrent.Task
/**
  * Created by nova on 16-12-20.
  */
object utils {

  import java.io.File

  val codes: List[String] = {
    val xa = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")
    sql"select distinct code from raw ".query[String].list.transact(xa).unsafePerformSync
  }

  val dates: List[String] = {
    val xa = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")
    sql"select distinct date from raw ".query[String].list.transact(xa).unsafePerformSync
  }

  def recursiveListFiles(f: File): Array[File] = {
    val these = f.listFiles
    these ++ these.filter(_.isDirectory).flatMap(recursiveListFiles)
  }

  def ZipValidation(filename: String): Boolean = {
    val zip = try {
      new ZipFile(filename)
    }
    catch {
      case ex: ZipException =>
        return false
    }
    zip.close
    return true
  }

  def GetDriverManagerTransactor = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")

  def save(obj: Any, path: String) = {
    val oos = new ObjectOutputStream(new FileOutputStream(path))
    oos.writeObject(obj)
  }

  def load(path: String) = {
    val ois = new ObjectInputStream(new FileInputStream(path))
    ois.readObject()
  }

  case class Raw(code: String, date: String,
                 industry: String, concept: String, area: String,
                 op: Float, mx: Float, mn: Float, clse: Float,
                 aft: Float, bfe: Float, amp: Float, vol: Float,
                 market: Float, market_exchange: Float,
                 on_board: Float, total: Float,
                 zt: Float, dt: Float,
                 shiyinlv: Float, shixiaolv: Float, shixianlv: Float, shijinglv: Float,
                 ma5: Float, ma10: Float, ma20: Float, ma30: Float, ma60: Float, macross: String,
                 macddif: Float, macddea: Float, macdmacd: Float, macdcross: String,
                 k: Float, d: Float, j: Float, kdjcross: String,
                 berlinmid: Float, berlinup: Float, berlindown: Float,
                 psy: Float, psyma: Float,
                 rsi1: Float, rsi2: Float, rsi3: Float,
                 zhenfu: Float, volratio: Float
                )

  case class RawIndex(index_code: String, index_date: String,
                      open: Float, close: Float, low: Float, high: Float,
                      volume: Float, money: Float, delta: Float
                     )

  case class RawConcept(concept: String, date: String,
                        amount: Int,
                        uppercent: Float, downpercent: Float, drawpercent: Float,
                        amp: Float, wamp: Float
                       )

  case class Key(code: String, date: String)

  case class Features(code: String, date: String, vector: Array[Float])
}

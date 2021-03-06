import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.zip.{ZipException, ZipFile}
import javax.mail.{Folder, Store}

import doobie.hikari.hikaritransactor.HikariTransactor
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
  val D2 = {
    val format = new SimpleDateFormat("yyyy-MM-dd")
    format.format(Calendar.getInstance().getTime)
  }
  val D1 = {
    val format = new SimpleDateFormat("yyyy-MM-dd")
    val cal = Calendar.getInstance()
    cal.add(Calendar.DATE, -1)
    format.format(cal.getTime)
  }
  val D0 = {
    val format = new SimpleDateFormat("yyyy-MM-dd")
    val cal = Calendar.getInstance()
    cal.add(Calendar.DATE, -2)
    format.format(cal.getTime)
  }

  def deleteifExists(path: String) = {
    val fileTemp = new File(path)
    if (fileTemp.exists) {
      fileTemp.delete()
    }
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
      case ex: Throwable =>
        return false
    }
    zip.close
    return true
  }

  def GetDriverManagerTransactor = DriverManagerTransactor[IOLite]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")


  def GetHikariTransactor(name:String): HikariTransactor[Task] = {

    val xa = HikariTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth").unsafePerformSync
    xa.configure(hx => Task(hx.setMaximumPoolSize(Runtime.getRuntime().availableProcessors()))).unsafePerformSync
    xa.configure(hx =>Task(hx.setPoolName(name))).unsafePerformSync
    xa
}

  def save(obj: Any, path: String) = {
    val oos = new ObjectOutputStream(new FileOutputStream(path))
    oos.writeObject(obj)
  }

  def load(path: String): Any = {
    val ois = new ObjectInputStream(new FileInputStream(path))
    val obj = ois.readObject()
    ois.close()
    return obj
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

  case class EmailProp(store: Store, inbox: Folder)

}

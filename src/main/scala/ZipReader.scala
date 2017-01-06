import java.io.{File, InputStream}
import java.util.zip.{ZipException, ZipFile}

import scala.util.matching.Regex
import doobie.imports._

import scalaz._
import Scalaz._
import scalaz.concurrent.Task
import scala.language.postfixOps

/**
  * Created by nova on 16-12-20.
  */
class ZipReader {

  def clear() = {
    val dmt = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")
    import dmt.yolo._
    sql"delete from raw;delete from rawindex;delete from rawconcept;".update.run.transact(dmt).unsafePerformSync
  }


  def ReadAll() = {

    val ziplist = utils.recursiveListFiles(new File("data/holo")).filter(_.getName.endsWith(".zip"))
    ziplist foreach {
      file =>
        val csvs = reader(file.getPath).toArray
        csvs.foreach {
          csv =>
            try {
              csv.FileName match {
                case "rawindex" => InsertIndexRaw(csv.csv)
                case "rawstock" => InsertStockRaw(csv.csv)
                case "rawconcept" => InsertConceptRaw(csv.csv)
              }
            }
            catch {
              case ex: java.sql.BatchUpdateException => {
                val eex = ex.getNextException
                if (!eex.getMessage.contains("duplicate key")) {
                  println(eex.getMessage)
                  System.exit(2)
                }
              }
            }
        }
    }


  }

  private def InsertStockRaw(csv: List[String]) = {
    val dmt = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")
    import dmt.yolo._
    val data = csv.map {
      line =>
        val cols = line.split(",")
        RawStock(cols(0), cols(1), cols(2), cols(3), cols(4), cols(5), cols(6).toFloat, cols(7).toFloat, cols(8).toFloat, cols(9).toFloat, cols(10).toFloat, cols(11).toFloat, cols(12).toFloat, cols(13).toFloat, cols(14).toFloat, cols(15).toFloat,
          cols(16).toFloat, cols(17).toFloat, cols(18).toFloat, cols(19).toFloat, cols(20).toFloat, cols(21).toFloat, cols(22).toFloat, cols(23).toFloat, cols(24).toFloat, cols(25).toFloat, cols(26).toFloat, cols(27).toFloat, cols(28).toFloat, cols(29),
          cols(30).toFloat, cols(31).toFloat, cols(32).toFloat,
          cols(33), cols(34).toFloat, cols(35).toFloat, cols(36).toFloat, cols(37),
          cols(38).toFloat, cols(39).toFloat, cols(40).toFloat, cols(41).toFloat, cols(42).toFloat,
          cols(43).toFloat, cols(44).toFloat, cols(45).toFloat, cols(46).toFloat, cols(47).toFloat)
    }
    RawStockBatch(data).transact(dmt).unsafePerformSync
  }

  private def RawStockBatch(data: List[RawStock]): ConnectionIO[Int] = {
    val sql =
      """
          INSERT INTO raw
          (code,name,date,industry,concept,area,
          op,mx,mn,clse,aft,bfe,amp,vol,market,market_exchange,on_board,total,zt,dt,shiyinlv,shixiaolv,shixianlv,shijinglv,
          ma5,ma10,ma20,ma30,ma60,macross,macddif,macddea,macdmacd,macdcross,k,d,j,kdjcross,berlinmid,berlinup,berlindown,
          psy,psyma,rsi1,rsi2,rsi3,zhenfu,volratio
          )
          VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      """
    Update[RawStock](sql).updateMany(data)
  }

  private def InsertIndexRaw(csv: List[String]) = {
    val dmt = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")
    import dmt.yolo._
    val data = csv.map {
      line =>
        val cols = line.split(",")
        RawIndex(cols(0), cols(1), cols(2).toFloat, cols(3).toFloat, cols(4).toFloat, cols(5).toFloat,
          cols(6).toFloat, cols(7).toFloat, cols(8).toFloat)
    }
    RawIndexBatch(data).transact(dmt).unsafePerformSync
  }

  private def RawIndexBatch(data: List[RawIndex]) = {
    val sql =
      """
          INSERT INTO rawindex
          (index_code,index_date,open,close,low,high,volume,money,delta)
          VALUES(?,?,?,?,?,?,?,?,?)
      """
    Update[RawIndex](sql).updateMany(data)
  }

  private def InsertConceptRaw(csv: List[String]) = {
    val dmt = DriverManagerTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "emeth")
    import dmt.yolo._
    val data = csv.map {
      line =>
        val cols = line.split(",")
        if (cols(7) == "")
          RawConcept(cols(0), cols(1).toFloat, cols(2).toFloat, cols(3).toFloat, cols(4).toFloat, cols(5).toFloat,
            cols(6).toFloat, null.asInstanceOf[Float], cols(8))
        else
          RawConcept(cols(0), cols(1).toFloat, cols(2).toFloat, cols(3).toFloat, cols(4).toFloat, cols(5).toFloat,
            cols(6).toFloat, cols(7).toFloat, cols(8))
    }
    RawConceptBatch(data).transact(dmt).unsafePerformSync
  }

  private def RawConceptBatch(data: List[RawConcept]) = {
    val sql =
      """
         INSERT INTO rawconcept
         (concept,amount,uppercent,downpercent,drawpercent,amp,wamp,aprofit,date)
         VALUES(?,?,?,?,?,?,?,?,?)
         """
    Update[RawConcept](sql).updateMany(data)
  }

  def reader(filename: String) = {
    val zip = try {
      new ZipFile(filename)
    }
    catch {
      case ex: ZipException => {
        println(ex.getMessage)
        println("zip error:" + filename)
        System.exit(3)
        new ZipFile(filename)
      }

    }
    import collection.JavaConverters._
    val entries = zip.entries().asScala.filter(_.getName.endsWith(".csv"))
    entries.map {
      file =>
        file.getName match {
          case name if name == "index data.csv" => CSV("rawindex", matching(zip.getInputStream(file), GlobalConfig.LegalIndexInformation))
          case name if name == "stock overview.csv" => CSV("rawstock", matching(zip.getInputStream(file), GlobalConfig.LegalStockInformation))
          case name if name == "industry overview.csv" => CSV("rawconcept", matching(zip.getInputStream(file), GlobalConfig.LegalConceptInformation))
        }
    }
  }

  def matching(stream: InputStream, re: Regex): List[String] = {
    scala.io.Source.fromInputStream(stream, "gbk").getLines().filter(s => re findFirstIn s isDefined).toList
  }

  case class CSV(FileName: String, csv: List[String])

  case class RawStock(code: String, name: String, date: String, industry: String, concept: String, area: String,
                      op: Float, mx: Float, mn: Float, clse: Float, aft: Float, bfe: Float, amp: Float, vol: Float, market: Float, market_exchange: Float,
                      on_board: Float, total: Float, zt: Float, dt: Float, shiyinlv: Float, shixiaolv: Float, shixianlv: Float, shijinglv: Float,
                      ma5: Float, ma10: Float, ma20: Float, ma30: Float, ma60: Float, macross: String,
                      macddif: Float, macddea: Float, macdmacd: Float, macdcross: String,
                      k: Float, d: Float, j: Float, kdjcross: String,
                      berlinmid: Float, berlinup: Float, berlindown: Float,
                      psy: Float, psyma: Float,
                      rsi1: Float, rsi2: Float, rsi3: Float,
                      zhenfu: Float, volratio: Float
                     )

  case class RawIndex(code: String, date: String, open: Float, close: Float, low: Float, high: Float, volume: Float, money: Float, delta: Float)

  case class RawConcept(concept: String, amount: Float, uppercent: Float, downpercent: Float, drawpercent: Float, amp: Float, wamp: Float, aprofit: Float, date: String)


}

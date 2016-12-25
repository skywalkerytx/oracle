/**
  * Created by nova on 16-12-25.
  */

import doobie.imports._

import scalaz._
import Scalaz._
import scalaz.concurrent.Task

class Vectorlize {

  case class MappingClass(str:String,cat:String,id:Int)

  val splitchar = '；'

  val toMap = List("industry","concept","area","macross","macdcross","kdjcross")//.par

  val Queries:Map[String,Query0[String]] =
    Map(
      "concept" -> sql"select distinct concept from raw;".query[String],
      "industry" -> sql"select distinct industry from raw;".query[String],
      "area" -> sql"select distinct area from raw;".query[String],
      "macross" -> sql"select distinct macross from raw".query[String],
      "macdcross" -> sql"select distinct macdcross from raw".query[String],
      "kdjcross" -> sql"select distinct kdjcross from raw".query[String]
    )

  def InsertInto(data:List[MappingClass]) = {
    val query = "insert into Mapping(str,cat,id) values(?,?,?)"
    Update[MappingClass](query).updateMany(data)
  }

  def GenMapping() = {
    toMap.foreach{
      col=>
        val xa = DriverManagerTransactor[Task]("org.postgresql.Driver","jdbc:postgresql:nova","nova","emeth")
        val result = Queries(col).list.transact(xa).unsafePerformSync.take(1000000).flatMap(_.split(splitchar)).distinct.zipWithIndex.map(tp2=> MappingClass(tp2._1,col,tp2._2))
        InsertInto(result).transact(xa).unsafePerformSync
    }
  }



}

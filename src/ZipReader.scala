import java.io.InputStream
import java.util.Scanner
import java.util.zip.ZipFile

import scala.util.matching.Regex



/**
  * Created by nova on 16-12-20.
  */
class ZipReader {

  def matching(stream:InputStream,re:Regex) = {
    scala.io.Source.fromInputStream(stream,"gbk").getLines().filter(s=> re findFirstIn s isDefined ).toArray
  }


  def reader(filename:String) = {
    val zip = new ZipFile(filename)
    import collection.JavaConverters._
    val entries = zip.entries().asScala.filter(_.getName.endsWith(".csv"))
    entries.map{
      file=>
        file.getName match {
          case name if (name == "index data.csv") => ("index",matching(zip.getInputStream(file),GlobalConfig.LegalIndexInformation))
          case name if (name == "stock overview.csv") => ("stock", matching(zip.getInputStream(file),GlobalConfig.LegalStockInformation))
          case name if (name == "industry overview.csv") => ("concept", matching(zip.getInputStream(file),GlobalConfig.LegalConceptInformation))
        }
    }.toMap
  }
}

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}


/**
  * Created by nova on 16-12-19.
  */


object Main {

  def DailyUpdate() = {
    val mail = new EmailReader()
    mail.GetAttachments
    val zip = new ZipReader()
    zip.ReadAll
  }

  def main(args: Array[String]): Unit = {

  }
}

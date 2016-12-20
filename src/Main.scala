import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import java.nio.file.Paths
import java.text.SimpleDateFormat
import java.time.format.DateTimeFormatter

import org.joda.time.DateTime
import org.joda.time.format.DateTimeFormat

/**
  * Created by nova on 16-12-19.
  */


object Main {

  def SaveMessage[A](obj: A) = {
    val oos = new ObjectOutputStream(new FileOutputStream("data/messages"))
    oos.writeObject(obj)
    oos.close
  }

  def LoadMessage() = {
    val ois = new ObjectInputStream(new FileInputStream("data/messages"))
    val obj = ois.readObject()
    ois.close
    obj
  }

  def main(args: Array[String]): Unit = {
    val mail = new EmailReader()
    //mail.Messages()
  }
}

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}




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

  def playground() = {
    import doobie.imports._
    import scalaz._, Scalaz._
    import scalaz.concurrent.Task
    import scalaz.stream.Process
    val xa = DriverManagerTransactor[Task] ("org.postgresql.Driver","jdbc:postgresql:nova","nova","emeth")
    import xa.yolo._

  }

  def main(args: Array[String]): Unit = {
    val mail = new EmailReader()
    //mail.Messages()
    playground()
    val zip = new ZipReader()
    zip.clear()
    zip.ReadAll()
  }
}

import java.io.{File, FileOutputStream}
import java.nio.file.Paths

/**
  * Created by nova on 16-12-19.
  */


class EmailReader(username: String = "908910385@qq.com", password: String = "ravigxtqyvmcbcjg") {

  import javax.mail._
  import javax.mail.search._

  val ImapServer = "imap.qq.com"
  val SmtpServer = "smtp.qq.com"

  def Messages() = {
    val prop = Prop()
    val store = prop._1
    val inbox = prop._2

    val From = new FromStringTerm("service@yucezhe.com")
    val Date = new SentDateTerm(ComparisonTerm.LE, new java.util.Date)
    val Subject = new SubjectTerm("市场全息数据推送")
    val Size = new SizeTerm(ComparisonTerm.GE, 1024)
    val Final = new AndTerm(Subject, Size)
    var messages = inbox.search(Final).map {
      message =>
        val date = trim(message.getSubject)
        (date, message.getMessageNumber)
    }.filter(message=>message._1.isDefined && !java.nio.file.Files.exists(Paths.get("data/holo/overview-push-"+message._1.get+".zip")))
    messages.foreach(message=>AttachmentByNumber(message._2,store,inbox))
    inbox.close(true)
    store.close
    messages
  }

  def trim(s: String): Option[String] = {
    val pattern = "(预测者网 - )([0-9]+)(市场全息数据推送)".r
    s match {
      case pattern(from, date, detail) => Some(date)
      case _ => Option(null)
    }
  }

  def Length() = {
    val prop = Prop()
    val store = prop._1
    val inbox = prop._2
    inbox.getMessageCount - inbox.getDeletedMessageCount
  }

  protected def Prop() = {
    val props = System.getProperties
    props.setProperty("mail.store.protocol", "imaps")
    val session = Session.getDefaultInstance(props, null)
    val store = session.getStore("imaps")
    store.connect(ImapServer, username, password)
    val inbox = store.getFolder("INBOX")
    inbox.open(Folder.READ_ONLY)
    (store, inbox)
  }

  def AttachmentByNumber(id: Int,s:Store = null,i:Folder = null):Boolean = {

    val prop: (Store, Folder) = if (s == null) Prop() else null
    val store = if (s == null) prop._1 else s
    val inbox = if (i == null) prop._2 else i
    val message = inbox.getMessage(id)
    val ContentType = message.getContentType
    val Content = message.getContent.asInstanceOf[Multipart]
    val attachment = (0 until Content.getCount).map(
      x => Content.getBodyPart(x)
    ).filterNot(part => part.getContentType.contains("TEXT")).head
    val IS = attachment.getInputStream
    val f = new File("data/holo/" + attachment.getFileName)
    val FOS = new FileOutputStream(f)
    val buffer = new Array[Byte](4096)
    var read = 0
    try {
      while ((read = IS.read(buffer)) != -1) {
        FOS.write(buffer, 0, read)
      }
    }
      catch {
        case _ =>
      }
    finally
    {
      FOS.close
      IS.close
    }
    return true
  }

  def playground() = {
    val subjectstr = "市场全息数据推送"
    val prop = Prop()
    val store = prop._1
    val inbox = prop._2
    val subject = new SubjectTerm(subjectstr)
    val Size = new SizeTerm(ComparisonTerm.GE, 1024)
    val Final = new AndTerm(subject,Size)
    val types = inbox.search(Final).map(_.getContent.asInstanceOf[Multipart]).flatMap{
      parts =>
        (0 until parts.getCount).map(x=>parts.getBodyPart(x)).map(_.getContentType.split(';').head)
    }
    inbox.close(true)
    store.close()
    types
  }

}

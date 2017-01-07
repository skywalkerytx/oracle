import java.io.{File, FileOutputStream}
import java.nio.file.Paths
import java.util.zip.{ZipException, ZipFile}
import utils.EmailProp

/**
  * Created by nova on 16-12-19.
  */


class EmailReader(username: String = "908910385@qq.com", password: String = "yavimvukjpavbbbc") {

  import javax.mail._
  import javax.mail.search._

  val ImapServer = "imap.qq.com"
  val SmtpServer = "smtp.qq.com"

  def GetAttachments() = {
    val prop = Prop()
    val store = prop.store
    val inbox = prop.inbox

    val From = new FromStringTerm("service@yucezhe.com")
    val Date = new SentDateTerm(ComparisonTerm.LE, new java.util.Date)
    val Subject = new SubjectTerm("市场全息数据推送")

    val Size = new SizeTerm(ComparisonTerm.GE, 1024)
    val Final = new AndTerm(Subject, Size)

    var messages = inbox.search(Final).map {
      message =>
        val date = trim(message.getSubject)
        (date, message.getMessageNumber)
    }.filter(
      message =>
        message._1.isDefined && !utils.ZipValidation("data/holo/overview-push-" + message._1.get + ".zip")
    )
    messages.foreach {
      message =>
        var success = false
        while (!success) {
          try {
            success = AttachmentByNumber(message._2, store, inbox)
          }
          catch {
            case ex: FolderClosedException => AttachmentByNumber(message._2, store, inbox)
          }
        }
    }
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

  def AttachmentByNumber(id: Int, s: Store = null, i: Folder = null): Boolean = {
    val prop: EmailProp = if (s == null) Prop() else null
    val store = if (s == null) prop.store else s
    val inbox = if (i == null) prop.inbox else i
    val message = inbox.getMessage(id)
    val ContentType = message.getContentType
    val Content = message.getContent.asInstanceOf[Multipart]
    val attachment = (0 until Content.getCount).map(
      x => Content.getBodyPart(x)
    ).filterNot(part => part.getContentType.contains("TEXT")).head
    val IS = attachment.getInputStream
    val filename = "data/holo/" + attachment.getFileName
    for (retry <- 1 until 10) {
      if (!utils.ZipValidation(filename)) {
        println("No valid Zip file " + filename + " found. retrying: " + retry.toString)
        println("writing: " + filename)
        val f = new File(filename)
        val FOS = new FileOutputStream(f)
        val buffer = new Array[Byte](4096)
        var read = 0
        try {
          read = IS.read(buffer)
          while (read != -1) {
            FOS.write(buffer, 0, read)
            read = IS.read(buffer)
          }
          read = IS.read(buffer)
        }
        catch {
          case ex: Throwable => println(ex.getMessage)
        }
        finally {
          FOS.close
          IS.close
        }
      }
      else {
        println("writing " + filename + " succeed.")
        return true
      }
    }
    return false
  }

  protected def Prop() = {
    val props = System.getProperties
    props.setProperty("mail.store.protocol", "imaps")
    val session = Session.getDefaultInstance(props, null)
    val store: Store = session.getStore("imaps")
    store.connect(ImapServer, username, password)
    val inbox: Folder = store.getFolder("INBOX")
    inbox.open(Folder.READ_ONLY)
    EmailProp(store, inbox)
  }

  def Length() = {
    val prop = Prop()
    val store = prop.store
    val inbox = prop.inbox
    inbox.getMessageCount - inbox.getDeletedMessageCount
  }


}

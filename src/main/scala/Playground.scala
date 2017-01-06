import ml.dmlc.mxnet._
import ml.dmlc.mxnet.optimizer.SGD


import java.io._
import java.text.SimpleDateFormat
import java.util.Calendar

import doobie.imports._

import scalaz._
import Scalaz._
import scalaz.concurrent.Task
import scala.language.postfixOps
import doobie.contrib.postgresql.pgtypes._
import org.postgresql.util.PSQLException
import utils.Features
import scala.language.postfixOps


import doobie.contrib.hikari.hikaritransactor._
import doobie.contrib.postgresql.sqlstate.class23.UNIQUE_VIOLATION
/**
  * Created by nova on 17-1-1.
  */
object Playground {
  def dl4jground = {
    val data = Symbol.Variable("data")
    val fc1 = Symbol.FullyConnected(name = "fc1")()(Map("data" -> data, "num_hidden" -> 128))
    val act1 = Symbol.Activation(name = "relu1")()(Map("data" -> fc1, "act_type" -> "relu"))
    val fc2 = Symbol.FullyConnected(name = "fc2")()(Map("data" -> act1, "num_hidden" -> 64))
    val act2 = Symbol.Activation(name = "relu2")()(Map("data" -> fc2, "act_type" -> "relu"))
    val fc3 = Symbol.FullyConnected(name = "fc3")()(Map("data" -> act2, "num_hidden" -> 10))
    val mlp = Symbol.SoftmaxOutput(name = "sm")()(Map("data" -> fc3))
    val trainDataIter = IO.MNISTIter(Map(
      "image" -> "data/train-images-idx3-ubyte",
      "label" -> "data/train-labels-idx1-ubyte",
      "data_shape" -> "(1, 28, 28)",
      "label_name" -> "sm_label",
      "batch_size" -> "50",
      "shuffle" -> "1",
      "flat" -> "0",
      "silent" -> "0",
      "seed" -> "10"))

    val valDataIter = IO.MNISTIter(Map(
      "image" -> "data/t10k-images-idx3-ubyte",
      "label" -> "data/t10k-labels-idx1-ubyte",
      "data_shape" -> "(1, 28, 28)",
      "label_name" -> "sm_label",
      "batch_size" -> "50",
      "shuffle" -> "1",
      "flat" -> "0", "silent" -> "0"))



    val model = FeedForward.newBuilder(mlp)
      .setContext(Context.gpu(0))
      .setNumEpoch(10)
      .setOptimizer(new SGD(learningRate = 0.1f, momentum = 0.9f, wd = 0.0001f))
      .setTrainData(trainDataIter)
      .setEvalData(valDataIter)
      .build()

    val probArrays = model.predict(valDataIter)
    // in this case, we do not have multiple outputs
    require(probArrays.length == 1)
    val prob = probArrays(0)

    // get real labels
    import scala.collection.mutable.ListBuffer
    valDataIter.reset()
    val labels = ListBuffer.empty[NDArray]
    while (valDataIter.hasNext) {
      val evalData = valDataIter.next()
      labels += evalData.label(0).copy()
    }
    val y = NDArray.concatenate(labels)

    // get predicted labels
    val predictedY = NDArray.argmax_channel(prob)

    require(y.shape == predictedY.shape)

    // calculate accuracy
    var numCorrect = 0
    var numTotal = 0
    for ((labelElem, predElem) <- y.toArray zip predictedY.toArray) {
      if (labelElem == predElem) {
        numCorrect += 1
      }
      numTotal += 1
    }
    val acc = numCorrect.toFloat / numTotal
    println(s"Final accuracy = $acc")

  }

  def DailyUpdate(SavetoDatabase: Boolean = false) = {
    if (SavetoDatabase) {
      val vec = new Vectorlize().GenMapping.DataBaseVector().par
      vec.foreach {
        vector =>
          Insert(DailyQuery("vector", vector))
      }
      val labels = new Labels().DataBaseLabel.par
      labels.foreach {
        label =>
          Insert(DailyQuery("label", label))
      }
    }
  }

  def DailyQuery(tablename: String, Feature: Features): ConnectionIO[Features] = {
    val query =
      s"""
        INSERT INTO
        ${tablename} (code,date,vector)
        VALUES(?,?,?)
      """
    Update[Features](query).toUpdate0(Feature).withUniqueGeneratedKeys("code", "date")
  }

  def Insert(query: ConnectionIO[Features]) = {
    val taskunit = for {
      xa <- utils.GetHikariTransactor
      a <- query.transact(xa).attemptSomeSqlState {
        case UNIQUE_VIOLATION => "Duplicate key, I really don't care about this"
      }.ensuring(xa.shutdown)
    } yield a
    taskunit.unsafePerformSync
  }

  def hikariground = {
    (0 to 2000000).par.foreach {
      blah =>
        val q = sql"select 42 from raw limit 1".query[Int].unique
        val p: Task[Int] = for {
          xa <- HikariTransactor[Task]("org.postgresql.Driver", "jdbc:postgresql:nova", "nova", "") //utils.GetHikariTransactor
          _ <- xa.configure(hx => Task.delay(/* do something with hx */ ()))
          a <- q.transact(xa) ensuring xa.shutdown
        } yield a
    }
  }
}

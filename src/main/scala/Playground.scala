import ml.dmlc.mxnet._
import ml.dmlc.mxnet.optimizer.SGD

import doobie.contrib.hikari.hikaritransactor.HikariTransactor
import doobie.imports.{ConnectionIO, _}

import scalaz._
import Scalaz._
import scalaz.concurrent.Task
import scala.language.postfixOps
import doobie.contrib.postgresql.pgtypes._
import doobie.contrib.postgresql.sqlstate.class23.UNIQUE_VIOLATION
import org.apache.log4j.BasicConfigurator
import shapeless.HNil


/**
  * Created by nova on 17-1-1.
  */
object Playground {

  def mxnetground = {

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


  def lq:Query0[(utils.Key,Int)] = {
    sql"select code,date,vector[1] from label".query[(utils.Key,Int)]
  }

  def rq(key:utils.Key):Query0[(Float,Float)] = {
    sql"select op,mx from raw where code= ${key.code} and date = ${key.date}".query[(Float,Float)]
  }

  def dq(code:String):Query0[String] = {
    sql"select date from raw where code = $code order by date asc".query[String]
  }

  def LabelCheck = {
    val xa = utils.GetHikariTransactor("LabelCheck-pool")
    val labels = lq.list.transact(xa).unsafePerformSync.toMap
    val codes = labels.keys.map(_.code).toList.distinct
    var count = 0
    codes.foreach{
      code=>
        val dates = dq(code).list.transact(xa).unsafePerformSync
        val realone = (0 until dates.length-2).par.map{
          i=>
            (i,rq(utils.Key(code,dates(i))).unique.transact(xa).unsafePerformSync,code,dates(i))
        }
        count = count + realone.count {
          real =>
            try {
              val label = labels(utils.Key(code, dates(real._1)))
              val tom = realone(real._1 + 1)._2._1
              val dft = realone(real._1 + 2)._2._2
              if ((label == 0 && tom * 1.03 <= dft))
                true
              else
                false
              if((label == 1 && tom * 1.03 > dft))
                true
              else
                false
              false
            }
            catch {
              case _: Throwable => false
            }
        }
    }
    println("mismatch label detected:",count)
  }
}

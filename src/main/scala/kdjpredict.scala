import java.io.{BufferedWriter, File, FileWriter}

import doobie.imports.Query0
import doobie.imports._
import ml.dmlc.mxnet.IO.IterCreateFunc
import utils.{codes, dates}
import utils.deleteifExists
import ml.dmlc.mxnet._
import ml.dmlc.mxnet.optimizer.SGD

/**
  * Created by nova on 17-3-12.
  */

//TODO: scale data with NormalizerStandardize, change it to shuang feng later
//

class kdjpredict {

  val xa = utils.GetHikariTransactor("kdjpredict")

  def datapreparation = {
    //grab k,d,j from csv and do some good stuff
    val rawkdj = codes.map(kdjquery).map(_.list.transact(xa).unsafePerformSync)
    codes.indices.foreach {
      idx =>
        val path = "data/dl4j/csv/" + codes(idx) + ".csv"
        deleteifExists(path)
        val csv = new File(path)
        val writer = new BufferedWriter(new FileWriter(csv))
        rawkdj(idx).foreach {
          kdj =>
            val bufferbuilder = new StringBuilder
            bufferbuilder ++= kdj.k.toString
            bufferbuilder += ','
            bufferbuilder ++= kdj.d.toString
            bufferbuilder += ','
            bufferbuilder ++= kdj.j.toString
            bufferbuilder += '\n'
            writer.write(bufferbuilder.result)
        }
        writer.close
    }
  }

  def kdjquery(code: String): Query0[kdj] = {
    sql"select k,d,j from raw where code = $code order by date asc".query[kdj]
  }

  def mxnet = {
    val data = Symbol.Variable("data")
    val fc1 = Symbol.FullyConnected(name = "fc1")()(Map("data" -> data, "num_hidden" -> 128))
    val act1 = Symbol.Activation(name = "relu1")()(Map("data" -> fc1, "act_type" -> "relu"))
    val fc2 = Symbol.FullyConnected(name = "fc2")()(Map("data" -> act1, "num_hidden" -> 64))
    val act2 = Symbol.Activation(name = "relu2")()(Map("data" -> fc2, "act_type" -> "relu"))
    val fc3 = Symbol.FullyConnected(name = "fc3")()(Map("data" -> act2, "num_hidden" -> 10))
    val mlp = Symbol.SoftmaxOutput(name = "sm")()(Map("data" -> fc3))
    val Data = 0 until 256 map (i => NDArray.ones(256, 256))
    val iter = new io.NDArrayIter(Data, dataBatchSize = 16)
    println(iter.length)
  }

  case class kdj(k: Int, d: Int, j: Int)

}

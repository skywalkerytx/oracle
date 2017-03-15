import doobie.postgres.pgtypes._
import doobie.imports.Query0
import org.postgresql.util.PSQLException
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import doobie.imports._
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader
import org.nd4j.linalg.api.ndarray.INDArray
import scala.collection.JavaConversions._
import utils.{codes, dates}

import scala.collection.immutable.Seq

/**
  * Created by nova on 17-3-12.
  */

//TODO: scale data with NormalizerStandardize, change it to shuang feng later
//

class kdjpredict {


  val xa = utils.GetHikariTransactor("for dl4j")

  def datapreparation = {
    codes.map { code =>
      FeaturesPercode(code)
    } //.head.foreach(println)

  }

  def FeaturesPercode(code: String) = {
    println(code)
    val rawset = query(code).list.transact(xa).unsafePerformSync
    //println(rawset.length)
    val label = rawset.map(_._4)
    val kdj = rawset.map {
      row =>
        Array(row._1, row._2, row._3).toNDArray
    }
    val finalfeature: java.util.List[INDArray] = (1 until kdj.length).map {
      idx =>
        Nd4j.concat(1, kdj(idx), kdj(idx) - kdj(idx - 1))
    }
    val shape: Array[Int] = Array(finalfeature.length, 6)

    println(Nd4j.create(finalfeature, shape).getClass)

    val reader = new CollectionRecordReader()


    System.exit(0)
  }

  def query(code: String): Query0[(Float, Float, Float, Float)] = {
    sql"""select k,d,j,
      case when kdjcross = '金叉' then 1 when kdjcross = '死叉' then 2 else 0 end
      from raw where code = $code order by date asc
      """
      .query[(Float, Float, Float, Float)]
  }


}

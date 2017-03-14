import doobie.postgres.pgtypes._
import doobie.imports.Query0
import org.postgresql.util.PSQLException
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.jita.conf.CudaEnvironment
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.factory.Nd4j
import doobie.imports._
import utils.{codes, dates}

import scala.collection.immutable.Seq

/**
  * Created by nova on 17-3-12.
  */

//TODO: scale data with NormalizerStandardize, change it to shuang feng later
//

class kdjpredict {


  val DMT = utils.GetHikariTransactor("for dl4j")

  def sequencepercode(code: String) = {
    val dataperday = query(code).list.transact(DMT).unsafePerformSync.toArray
    if (dataperday.Length < 5)
      return null

  }

  def query(code: String): Query0[(String, Float, Float, Float, String)] = {
    sql"select k,d,j,kdjcross from raw where code = $code order by date asc"
      .query[(String, Float, Float, Float, String)]
  }

  def datapreparation = {
    codes.map { code =>
      query(code).list.transact(DMT).unsafePerformSync.toArray
    }.head.foreach(println)
    val RecordReader = new CollectionRecordReader()
  }


}

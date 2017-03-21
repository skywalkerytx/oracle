import java.util

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

  //TODO: change plan, grab the data, save to csv, then use CsvRecordReader

  def DataPreparation = {

  }

  def query(code: String): Query0[(Float, Float, Float, Float)] = {
    sql"""select k,d,j,
      case when kdjcross = '金叉' then 1 when kdjcross = '死叉' then 2 else 0 end
      from raw where code = $code order by date asc
      """
      .query[(Float, Float, Float, Float)]
  }


}

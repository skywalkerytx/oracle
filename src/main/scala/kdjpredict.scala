

/**
  * Created by nova on 17-3-12.
  */

//TODO: scale data with NormalizerStandardize, change it to shuang feng later
//

class kdjpredict {

  val xa = utils.GetHikariTransactor("kdjpredict")



  case class kdj(k: Int, d: Int, j: Int)

}

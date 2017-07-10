
import java.io.Serializable

import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel


object TestFM extends App {
  def indiceChange(sc: SparkContext,path_in :String): RDD[String] ={
    """
    """.stripMargin
    val data = sc.textFile(path_in,minPartitions = 1000)
    val train: RDD[String] = data.map{
      line=>
        val segs: Array[String] = line.split(' ')
        val label = if(segs(0) == "1") "1" else "-1"
        val features = segs.drop(1)
        // add indices 1
        val features_process: Array[String] = features.map{
          elem =>
            val index = elem.split(":")(0).toInt
            val value = elem.split(":")(1)
            val new_index = index + 1 //index should be begin 1
            //val new_index = index
            new_index.toString + ":" +value
        }
        // sort index
        val features_sort: Array[String] = features_process.sortWith{
          (leftE, rightE) =>
            leftE.split(":")(0).toInt < rightE.split(":")(0).toInt
        }
        val line_arr: Array[String] = label +: features_sort
        // string line
        line_arr.mkString(" ")
    }
    train
  }

  def process_data(sc:SparkContext,path_in:String,ifSplit:Double):Array[RDD[LabeledPoint]]={

    val train: RDD[String] = indiceChange(sc,path_in)
    val util = new MyUtil
    val data: RDD[LabeledPoint] = util.loadLibSVMFile(sc, train,numFeatures = -1,minPartitions = 1000).persist(StorageLevel.MEMORY_AND_DISK)
    if (ifSplit > 0 && ifSplit < 1){
      val splitRdd: Array[RDD[LabeledPoint]] = data.randomSplit(Array(10*ifSplit,10*(1-ifSplit)),2017)
      return splitRdd
    }else{
      return Array(data)
    }
  }

  override def main(args: Array[String]): Unit = {

    val logger = Logger.getLogger("MY LOGGER")
    logger.info(s"===args===")
    args.foreach(elem=>logger.info(elem))
    logger.info(s"===args===")

    val intask = args(0).toInt
    val inallIterations = args(1).toInt
    val innumCorrections = args(2).toInt
    val intolerance = args(3).toDouble
    val indim = args(4).toInt
    val inreg1 = args(5).toDouble
    val inreg2 = args(6).toDouble
    val ininitStd = args(7).toDouble
    val instep = args(8).toInt
    val checkPointPath = args(9)
    val earlyStop = args(10).toInt

    val train_path_in = args(11)
    val test_path_in = args(12)
    val ifTestTrain = args(13).toInt
    val jobName = args(14)

    val localPath = args(15)
    val featureIDPath = args(16)

    val inregParam = (0,inreg1,inreg2)

    val ifSplit = args(17).toDouble
    val ifSaveweight = args(18).toInt


    // print warn
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val conf = new SparkConf().setAppName(jobName)
    conf.set("spark.hadoop.validateOutputSpecs","false")
    conf.set("spark.kryoserializer.buffer.max","2047m")

    val sc: SparkContext = new SparkContext(conf)
    sc.setCheckpointDir("/team/ad_wajue/chenlongzhen/checkpoint")


    logger.info("processing data")

    val useData: Array[RDD[LabeledPoint]] = if (test_path_in == "0") {
      val splitdata: Array[RDD[LabeledPoint]] = process_data(sc, train_path_in, ifSplit)
      splitdata
    }else{
      val train_data = process_data(sc, train_path_in, 0)(0)
      val test_data = process_data(sc, test_path_in, 0)(0)
      Array(train_data,test_data)
    }

    if (useData.length != 1) { // train and test
      val train_data = useData(0)
      val test_data = useData(1)

      val task = args(1).toInt
      val numIterations = args(2).toInt
      val stepSize = args(3).toDouble
      val miniBatchFraction = args(4).toDouble

      logger.info("train lbfgs")
      val fm2 = FMWithLBFGS.train(train_data, test_data, task = 1,
        numIterations = inallIterations, numCorrections = innumCorrections, tolerance = intolerance,
        dim = (true, true, indim), regParam = (0, inreg1, inreg2), initStd = ininitStd, step = instep,
        checkPointPath = checkPointPath, earlyStop = earlyStop, sc = sc, ifTestTrain = ifTestTrain,
        localPath = localPath, featureIDPath = featureIDPath, reload = 0,ifSaveWeight = 0)

      //save weight factor to local : no use! wrong version!
      //logger.info(s"save weight to local : $localPath")
      //FMModel.loadWeight2Local(sc,Modelpath = checkPointPath+s"/model",localPath = localPath,featureIDPath=featureIDPath)

      sc.stop()
    }else{ //train only

      val train_data = useData(0)

      val task = args(1).toInt
      val numIterations = args(2).toInt
      val stepSize = args(3).toDouble
      val miniBatchFraction = args(4).toDouble

      logger.info("train online  lbfgs")
      val fm2 = FMWithLBFGS.trainOnline(train_data, task = 1,
        numIterations = inallIterations, numCorrections = innumCorrections,tolerance=intolerance,
        dim = (true, true, indim), regParam = (0, inreg1, inreg2), initStd = ininitStd,
        sc = sc, ifTestTrain = ifTestTrain,
        localPath = localPath, featureIDPath = featureIDPath, reload = 0/*未实现*/,ifSaveWeight = 0)

      //save weight factor to local : no use! wrong version!
      //logger.info(s"save weight to local : $localPath")
      //FMModel.loadWeight2Local(sc,Modelpath = checkPointPath+s"/model",localPath = localPath,featureIDPath=featureIDPath)
      sc.stop()

    }
  }
}

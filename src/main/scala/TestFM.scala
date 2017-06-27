
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD




object TestFM extends App {
  def indiceChange(sc: SparkContext,path_in :String): RDD[String] ={
    """
      |indice base 0 to 1; label 0 to -1
    """.stripMargin
    val data = sc.textFile(path_in,minPartitions = 1000)
    val train: RDD[String] = data.map{
      line=>
        val segs: Array[String] = line.split('\t')
        val label = if(segs(0) == "1") "1" else "-1"
        val features = segs.drop(1)
        // add indices 1
        val features_process: Array[String] = features.map{
          elem =>
            val index = elem.split(":")(0).toInt
            val value = elem.split(":")(1)
            //val new_index = index + 1
            val new_index = index
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


    //print(train.take(2))

    //train.saveAsTextFile(path_out)

    train

  }

  def process_data(sc:SparkContext,path_in:String):RDD[LabeledPoint]={

    val train: RDD[String] = indiceChange(sc,path_in)
    val util = new MyUtil
    val data: RDD[LabeledPoint] = util.loadLibSVMFile(sc, train,numFeatures = -1,minPartitions = 1000).cache()
    data
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

    val inregParam = (0,inreg1,inreg2)


    /*
    val intask = 1
    val inallIterations = 20
    val innumCorrections = 5
    val intolerance = 1e-7
    val indim = 5
    val inregParam = (0,0.01,0.01)
    val ininitStd = 0.1
    val instep = 1
    val checkPointPath = "/team/ad_wajue/chenlongzhen/checkPoint"
    val earlyStop = 10
    */

    // print warn
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val conf = new SparkConf().setAppName("sparkFM")
    conf.set("spark.hadoop.validateOutputSpecs","false")
    conf.set("spark.kryoserializer.buffer.max","2047m")

    val sc: SparkContext = new SparkContext(conf)
    sc.setCheckpointDir("/team/ad_wajue/chenlongzhen/checkpoint")


/*    val train_path_in = "/team/ad_wajue/dw/rec_ml_dev6/rec_ml_dev6/model_dataSet/training"
    val train_path_out = "/team/ad_wajue/chenlongzhen/model_dataSet/training_processed"
    val test_path_in = "/team/ad_wajue/dw/rec_ml_dev6/rec_ml_dev6/model_dataSet/training"
    val test_path_out = "/team/ad_wajue/chenlongzhen/model_dataSet/training_processed"*/


    // process lines


    logger.info("processing data")
    val train_data = process_data(sc,train_path_in)
    val test_data = process_data(sc,test_path_in)

    //    val task = args(1).toInt
    //    val numIterations = args(2).toInt
    //    val stepSize = args(3).toDouble
    //    val miniBatchFraction = args(4).toDouble

    //print("train SGD")
    //val fm1 = FMWithSGD.train(training, task = 1, numIterations = 100, stepSize = 0.15, miniBatchFraction = 1.0, dim = (true, true, 4), regParam = (0, 0, 0), initStd = 0.1)


    logger.info("train lbfgs")
    val fm2 = FMWithLBFGS.train(train_data, test_data, task = 1,
      numIterations = inallIterations, numCorrections = innumCorrections, tolerance = intolerance,
      dim = (true,true,indim), regParam = (0,0.01,0.01), initStd =ininitStd,step = instep,
      checkPointPath = checkPointPath,earlyStop = earlyStop,sc = sc, ifTestTrain=ifTestTrain)
    //fm2.save(sc, s"/team/ad_wajue/chenlongzhen/fmmodel_save/fmmodel_end")


    //predict
    //val path_test_in = "/team/ad_wajue/dw/rec_ml_test/rec_ml_test/model_dataSet/testing"
    //val path_test_out = "/team/ad_wajue/dw/rec_ml_test/rec_ml_test/model_dataSet/testing_processed"
    sc.stop()
  }
}


import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD




object TestFM extends App {
  def indiceChange(sc: SparkContext,path_in :String,path_out:String): Unit ={
    """
      |indice base 0 to 1; label 0 to -1
    """.stripMargin
    val data = sc.textFile(path_in).repartition(1000).cache()
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
            val new_index = index + 1
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

    train.saveAsTextFile(path_out)

  }

  def process_data(sc:SparkContext,path_in:String,path_out:String):RDD[LabeledPoint]={

    indiceChange(sc,path_in,path_out)
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, path_out)
    data
  }


  override def main(args: Array[String]): Unit = {

    val intask = 1
    val inallIterations = 3
    val innumCorrections = 20
    val intolerance = 1e-7
    val indim = 5
    val inregParam = (0,0.01,0.01)
    val ininitStd = 0.1
    val instep = 1
    val checkPointPath = "/team/ad_wajue/chenlongzhen/checkPoint"
    val earlyStop = 10

    // print warn
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val conf = new SparkConf().setAppName("sparkFM")
    conf.set("spark.hadoop.validateOutputSpecs","false")
    conf.set("spark.kryoserializer.buffer.max","2047m")

    val sc: SparkContext = new SparkContext(conf)
    sc.setCheckpointDir("/team/ad_wajue/chenlongzhen/checkpoint")

    val train_path_in = "/team/ad_wajue/dw/rec_ml_dev6/rec_ml_dev6/model_dataSet/training"
    val train_path_out = "/team/ad_wajue/chenlongzhen/model_dataSet/training_processed"
    val test_path_in = "/team/ad_wajue/dw/rec_ml_dev6/rec_ml_dev6/model_dataSet/training"
    val test_path_out = "/team/ad_wajue/chenlongzhen/model_dataSet/training_processed"


    // process lines
    val logger = Logger.getLogger("MY LOGGER")


    logger.info("processing data")
    val train_data = process_data(sc,train_path_in,train_path_out)
    val test_data = process_data(sc,test_path_in,test_path_out)

    //    val task = args(1).toInt
    //    val numIterations = args(2).toInt
    //    val stepSize = args(3).toDouble
    //    val miniBatchFraction = args(4).toDouble

    //print("train SGD")
    //val fm1 = FMWithSGD.train(training, task = 1, numIterations = 100, stepSize = 0.15, miniBatchFraction = 1.0, dim = (true, true, 4), regParam = (0, 0, 0), initStd = 0.1)


    logger.info("train lbfgs")
    val fm2 = FMWithLBFGS.train(train_data, test_data, task = 1,
      numIterations = 5, numCorrections = innumCorrections, tolerance = intolerance,
      dim = (true,true,indim), regParam = (0,0.01,0.01), initStd =ininitStd,step = instep,
      checkPointPath = checkPointPath,earlyStop = earlyStop,sc = sc)
    fm2.save(sc, s"/team/ad_wajue/chenlongzhen/fmmodel_save/fmmodel_end")


    //predict
    //val path_test_in = "/team/ad_wajue/dw/rec_ml_test/rec_ml_test/model_dataSet/testing"
    //val path_test_out = "/team/ad_wajue/dw/rec_ml_test/rec_ml_test/model_dataSet/testing_processed"
    sc.stop()
  }
}

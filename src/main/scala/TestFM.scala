
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD


/**
 * Created by zrf on 4/18/15.
 */


object TestFM extends App {
  def indiceChange(sc: SparkContext,path_in :String,path_out:String): Unit ={
    """
      |indice base 0 to 1; label 0 to -1
    """.stripMargin
    val data = sc.textFile(path_in)
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


  override def main(args: Array[String]): Unit = {

    // print warn
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val conf = new SparkConf().setAppName("sparkFM")
    conf.set("spark.hadoop.validateOutputSpecs","false")
    conf.set("spark.kryoserializer.buffer.max","2047m")

    val sc: SparkContext = new SparkContext(conf)
    sc.setCheckpointDir("/team/ad_wajue/chenlongzhen/checkpoint")

    val path_in = "/team/ad_wajue/dw/rec_ml_test/rec_ml_test/model_dataSet/training"
    val path_out = "/team/ad_wajue/dw/rec_ml_test/rec_ml_test/model_dataSet/training_processed"

    // process lines
    print("indeicChange")
    //indiceChange(sc,path_in,path_out)

    //    "hdfs://ns1/whale-tmp/url_combined"
    print("load svm file")
    val training = MLUtils.loadLibSVMFile(sc, path_out).cache()

    //    val task = args(1).toInt
    //    val numIterations = args(2).toInt
    //    val stepSize = args(3).toDouble
    //    val miniBatchFraction = args(4).toDouble

    print("train SGD")
    val fm1 = FMWithSGD.train(training, task = 1, numIterations = 100, stepSize = 0.15, miniBatchFraction = 1.0, dim = (true, true, 4), regParam = (0, 0, 0), initStd = 0.1)


    print("train lbfgs")
    val fm2 = FMWithLBFGS.train(training, task = 1, numIterations = 20, numCorrections = 5, dim = (true, true, 8), regParam = (0, 0.01, 0.01), initStd = 0.1)
  }
}

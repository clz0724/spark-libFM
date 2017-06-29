package org.apache.spark.mllib.regression

import org.apache.log4j.Logger
import java.io.{File, PrintWriter}
import java.text.NumberFormat

import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import scala.util.Random
import org.apache.spark.{Logging, SparkContext}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.optimization.{Gradient, Updater}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.util.Loader._
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.sql.{DataFrame, SQLContext}

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

/**
  * Created by zrf on 4/13/15.
  */

/**
  * Factorization Machine model.
  */
class FMModel(val task: Int,
              val factorMatrix: Matrix,
              val weightVector: Option[Vector],
              val intercept: Double,
              val min: Double,
              val max: Double) extends Serializable with Saveable {

  val numFeatures = factorMatrix.numCols
  val numFactors = factorMatrix.numRows

  require(numFeatures > 0 && numFactors > 0)
  require(task == 0 || task == 1)

  def predict(testData: Vector): Double = {
    //require(testData.size == numFeatures)

    var pred = intercept
    if (weightVector.isDefined) {
      testData.foreachActive {
        case (i, v) =>
          if (i < numFeatures) { //有可能预测集合的feature数量比测试集合的数量多
            pred += weightVector.get(i) * v
          }
      }
    }

    for (f <- 0 until numFactors) {
      var sum = 0.0
      var sumSqr = 0.0
      testData.foreachActive {
        case (i, v) =>
          if (i < numFeatures) {
            val d = factorMatrix(f, i) * v
            sum += d
            sumSqr += d * d
          }
      }
      pred += (sum * sum - sumSqr) / 2
    }

    task match {
      case 0 =>
        Math.min(Math.max(pred, min), max)
      case 1 =>
        1.0 / (1.0 + Math.exp(-pred))
    }
  }

  def predict(testData: RDD[Vector]): RDD[Double] = {
    testData.mapPartitions {
      _.map {
        vec =>
          predict(vec)
      }
    }
  }

  override protected def formatVersion: String = "1.0"

  override def save(sc: SparkContext, path: String): Unit = {
    val data = FMModel.SaveLoadV1_0.Data(factorMatrix, weightVector, intercept, min, max, task)
    FMModel.SaveLoadV1_0.save(sc, path, data)
  }
}

object FMModel extends Loader[FMModel] {

  private object SaveLoadV1_0 {

    def thisFormatVersion = "1.0"

    def thisClassName = "org.apache.spark.mllib.regression.FMModel"

    /** Model data for model import/export */
    case class Data(factorMatrix: Matrix, weightVector: Option[Vector], intercept: Double,
                    min: Double, max: Double, task: Int)

    def save(sc: SparkContext, path: String, data: Data): Unit = {
      val sqlContext = new SQLContext(sc)
      import sqlContext.implicits._
      // Create JSON metadata.
      val metadata = compact(render(
        ("class" -> this.getClass.getName) ~ ("version" -> thisFormatVersion) ~
          ("numFeatures" -> data.factorMatrix.numCols) ~ ("numFactors" -> data.factorMatrix.numRows)
          ~ ("min" -> data.min) ~ ("max" -> data.max) ~ ("task" -> data.task)))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(metadataPath(path))

      // Create Parquet data.
      val dataRDD: DataFrame = sc.parallelize(Seq(data), 1).toDF()
      dataRDD.saveAsParquetFile(dataPath(path))
    }


    def load(sc: SparkContext, path: String): FMModel = {
      val sqlContext = new SQLContext(sc)
      // Load Parquet data.
      val dataRDD = sqlContext.parquetFile(dataPath(path))
      // Check schema explicitly since erasure makes it hard to use match-case for checking.
      checkSchema[Data](dataRDD.schema)
      val dataArray = dataRDD.select("task", "factorMatrix", "weightVector", "intercept", "min", "max").take(1)
      assert(dataArray.length == 1, s"Unable to load FMModel data from: ${dataPath(path)}")
      val data = dataArray(0)
      val task = data.getInt(0)
      val factorMatrix = data.getAs[Matrix](1)
      val weightVector:Option[Vector] = data.getAs[Option[Vector]](2)
      val intercept = data.getDouble(3)
      val min = data.getDouble(4)
      val max = data.getDouble(5)
      new FMModel(task, factorMatrix, weightVector, intercept, min, max)
    }

    def loadWeight2Local(sc: SparkContext, Modelpath: String,localPath:String,featureIDPath:String):Unit = {
      """
        Modelpath: hdfs modelpath
        localPath: local save path
        featureIDPath: local id path
      """.stripMargin
      val sqlContext = new SQLContext(sc)
      // Load Parquet data.
      val dataRDD = sqlContext.parquetFile(dataPath(Modelpath))
      // Check schema explicitly since erasure makes it hard to use match-case for checking.
      checkSchema[Data](dataRDD.schema)
      val dataArray = dataRDD.select("task", "factorMatrix", "weightVector", "intercept", "min", "max").take(1)
      assert(dataArray.length == 1, s"Unable to load FMModel data from: ${dataPath(Modelpath)}")
      val data = dataArray(0)
      val task = data.getInt(0)
      val factorMatrix: Matrix = data.getAs[Matrix](1)
      val weightVector: DenseVector = data.getAs[DenseVector](2)
      val intercept: Double = data.getDouble(3)
      val min = data.getDouble(4)
      val max = data.getDouble(5)

      // get feature Map
      val file = Source.fromFile(featureIDPath)

      var IDFeatureMap:Map[String,String] = Map()
      for (line <- file.getLines){
        val segs = line.split('\t')
        val len = segs.length
        require(len == 6, s"$featureIDPath file has $len columns, need 6, and col0 is id , col2 is id name!")
        val ID = segs(0)
        val NAME = segs(2)
        IDFeatureMap += (ID -> NAME)
      }
      // 保留小数
      val format = NumberFormat.getInstance()
      format.setMinimumFractionDigits(1)
      format.setMinimumIntegerDigits(1)
      format.setMaximumFractionDigits(8)
      format.setMaximumIntegerDigits(8)
      //System.out.println(format.format(2132323213.23266666666));

      // get info
      val numFeatures = factorMatrix.numCols
      val numFactors = factorMatrix.numRows
      val weightLen = weightVector.toArray.length
      val logger =  Logger.getLogger("MY LOG")
      logger.info(s"In load, $numFeatures $numFactors $weightLen ")
      require(numFeatures == weightLen, s"factorMatrix len $numFeatures, weightLen $weightLen, not euqal!")

      val writer = new PrintWriter(new File(localPath))
      writer.write(s"bias\t$intercept\n")
      writer.write(s"nfactor\t$numFactors\n")

      for (i <- 0 until numFeatures){
        val arrBuffer = ArrayBuffer[String]()

        val idName: String = IDFeatureMap.getOrElse((i+1).toString,"NULL")
        arrBuffer += idName
        val weight: Double = weightVector.apply(i)
        arrBuffer += format.format(weight)

        for (f <- 0 until numFactors){
          val elem = factorMatrix(f,i)
          arrBuffer += format.format(elem)
        }
        writer.write(arrBuffer.toArray.mkString("\t") + "\n")
      }
      writer.close()
    }
  }


  def loadWeight2Local(sc: SparkContext, Modelpath: String,localPath:String,featureIDPath:String):Unit = {

    SaveLoadV1_0.loadWeight2Local(sc,Modelpath,localPath,featureIDPath)

  }

  override def load(sc: SparkContext, path: String): FMModel = {
    SaveLoadV1_0.load(sc, path)
    /*
    implicit val formats = DefaultFormats

    val (loadedClassName, version, metadata) = loadMetadata(sc, path)
    val classNameV1_0 = SaveLoadV1_0.thisClassName

    (loadedClassName, version) match {
      case (className, "1.0") if className == classNameV1_0 =>
        val numFeatures = (metadata \ "numFeatures").extract[Int]
        val numFactors = (metadata \ "numFactors").extract[Int]
        val model = SaveLoadV1_0.load(sc, path)
        assert(model.factorMatrix.numCols == numFeatures,
          s"FMModel.load expected $numFeatures features," +
            s" but factorMatrix had columns of size:" +
            s" ${model.factorMatrix.numCols}")
        assert(model.factorMatrix.numRows == numFactors,
          s"FMModel.load expected $numFactors factors," +
            s" but factorMatrix had rows of size:" +
            s" ${model.factorMatrix.numRows}")
        model

      case _ => throw new Exception(
        s"FMModel.load did not recognize model with (className, format version):" +
          s"($loadedClassName, $version).  Supported:\n" +
          s"  ($classNameV1_0, 1.0)")
    }*/
  }
}


/**
  * :: DeveloperApi ::
  * Compute gradient and loss for a Least-squared loss function, as used in linear regression.
  * For the detailed mathematical derivation, see the reference at
  * http://doi.acm.org/10.1145/2168752.2168771
  */
class FMGradient(val task: Int, val k0: Boolean, val k1: Boolean, val k2: Int,
                 val numFeatures: Int, val min: Double, val max: Double) extends Gradient {

  private def predict(data: Vector, weights: Vector): (Double, Array[Double]) = {

    var pred = if (k0) weights(weights.size - 1) else 0.0

    if (k1) {
      val pos = numFeatures * k2
      data.foreachActive {
        case (i, v) =>
          pred += weights(pos + i) * v
      }
    }

    val sum = Array.fill(k2)(0.0)
    for (f <- 0 until k2) {
      var sumSqr = 0.0
      data.foreachActive {
        case (i, v) =>
          val d = weights(i * k2 + f) * v
          sum(f) += d
          sumSqr += d * d
      }
      pred += (sum(f) * sum(f) - sumSqr) / 2
    }

    if (task == 0) {
      pred = Math.min(Math.max(pred, min), max)
    }

    (pred, sum)
  }


  private def cumulateGradient(data: Vector, weights: Vector,
                               pred: Double, label: Double,
                               sum: Array[Double], cumGrad: Vector): Unit = {

    val mult = task match {
      case 0 =>
        pred - label
      case 1 =>
        -label * (1.0 - 1.0 / (1.0 + Math.exp(-label * pred)))
    }

    cumGrad match {
      case vec: DenseVector =>
        val cumValues = vec.values

        if (k0) {
          cumValues(cumValues.length - 1) += mult
        }

        if (k1) {
          val pos = numFeatures * k2
          data.foreachActive {
            case (i, v) =>
              cumValues(pos + i) += v * mult
          }
        }

        data.foreachActive {
          case (i, v) =>
            val pos = i * k2
            for (f <- 0 until k2) {
              cumValues(pos + f) += (sum(f) * v - weights(pos + f) * v * v) * mult
            }
        }

      case _ =>
        throw new IllegalArgumentException(
          s"cumulateGradient only supports adding to a dense vector but got type ${cumGrad.getClass}.")
    }
  }


  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val cumGradient = Vectors.dense(Array.fill(weights.size)(0.0))
    val loss = compute(data, label, weights, cumGradient)
    (cumGradient, loss)
  }

  override def compute(data: Vector, label: Double, weights: Vector, cumGradient: Vector): Double = {
    require(data.size == numFeatures)
    val (pred, sum) = predict(data, weights)
    cumulateGradient(data, weights, pred, label, sum, cumGradient)

    task match {
      case 0 =>
        (pred - label) * (pred - label)
      case 1 =>
        1 - Math.signum(pred * label)
    }
  }
}

/**
  * :: DeveloperApi ::
  * Updater for L2 regularized problems.
  * Uses a step-size decreasing with the square root of the number of iterations.
  */
class FMUpdater(val k0: Boolean, val k1: Boolean, val k2: Int,
                val r0: Double, val r1: Double, val r2: Double,
                val numFeatures: Int) extends Updater {

  override def compute(weightsOld: Vector, gradient: Vector,
                       stepSize: Double, iter: Int, regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val len = weightsOld.size

    val weightsNew = Array.fill(len)(0.0)
    var regVal = 0.0

    if (k0) {
      weightsNew(len - 1) = weightsOld(len - 1) - thisIterStepSize * (gradient(len - 1) + r0 * weightsOld(len - 1))
      regVal += r0 * weightsNew(len - 1) * weightsNew(len - 1)
    }

    if (k1) {
      for (i <- numFeatures * k2 until numFeatures * k2 + numFeatures) {
        weightsNew(i) = weightsOld(i) - thisIterStepSize * (gradient(i) + r1 * weightsOld(i))
        regVal += r1 * weightsNew(i) * weightsNew(i)
      }
    }

    for (i <- 0 until numFeatures * k2) {
      weightsNew(i) = weightsOld(i) - thisIterStepSize * (gradient(i) + r2 * weightsOld(i))
      regVal += r2 * weightsNew(i) * weightsNew(i)
    }

    (Vectors.dense(weightsNew), regVal / 2)
  }
}

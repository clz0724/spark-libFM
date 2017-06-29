package org.apache.spark.mllib.regression

import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
  * Created by clz on 2017/6/27.
  */
class MyUtil {

  val logger = Logger.getLogger("MY LOGGER")


  def rmHDFS(path: String) {
    val hadoopConf = new org.apache.hadoop.conf.Configuration()
    val hdfs = org.apache.hadoop.fs.FileSystem.get(new java.net.URI("hdfs://ns2"), hadoopConf)
    try {
      hdfs.delete(new org.apache.hadoop.fs.Path(path), true)
    } catch {
      case _: Throwable => {
        logger.warn(s"rm hdfs wrong $path")
      }

    }
  }


  def dirExists(path: String):Boolean= {
    val hadoopConf = new org.apache.hadoop.conf.Configuration()
    val hdfs = org.apache.hadoop.fs.FileSystem.get(new java.net.URI("hdfs://ns2"), hadoopConf)
    val exists = hdfs.exists(new org.apache.hadoop.fs.Path(path))
    exists
    }

  /**
    * Loads labeled data in the LIBSVM format into an RDD[LabeledPoint].
    * The LIBSVM format is a text-based format used by LIBSVM and LIBLINEAR.
    * Each line represents a labeled sparse feature vector using the following format:
    * {{{label index1:value1 index2:value2 ...}}}
    * where the indices are one-based and in ascending order.
    * This method parses each line into a [[org.apache.spark.mllib.regression.LabeledPoint]],
    * where the feature indices are converted to zero-based.
    *
    * @param sc            Spark context
    * //@param path          file or directory path in any Hadoop-supported file system URI
    * @param train          train rdd
    * @param numFeatures   number of features, which will be determined from the input data if a
    *                      nonpositive value is given. This is useful when the dataset is already split
    *                      into multiple files and you want to load them separately, because some
    *                      features may not present in certain files, which leads to inconsistent
    *                      feature dimensions.
    * @param minPartitions min number of partitions
    * @return labeled data stored as an RDD[LabeledPoint]
    */
  def loadLibSVMFile(
                      sc: SparkContext,
                      train:RDD[String],
                      numFeatures: Int,
                      minPartitions: Int): RDD[LabeledPoint] = {
    val parsed = train
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map { line =>
        val items = line.split(' ')
        val label = items.head.toDouble
        val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
          val indexAndValue = item.split(':')
          val index = indexAndValue(0).toInt - 1
          // Convert 1-based indices to 0-based.
          val value = indexAndValue(1).toDouble
          (index, value)
        }.unzip

        // check if indices are one-based and in ascending order
        var previous = -1
        var i = 0
        val indicesLength = indices.length
        while (i < indicesLength) {
          val current = indices(i)
          require(current > previous, "indices should be one-based and in ascending order")
          previous = current
          i += 1
        }

        (label, indices.toArray, values.toArray)
      }

    // Determine number of features.
    val d = if (numFeatures > 0) {
      numFeatures
    } else {
      parsed.persist(StorageLevel.MEMORY_ONLY)
      parsed.map { case (label, indices, values) =>
        indices.lastOption.getOrElse(0)
      }.reduce(math.max) + 1
    }

    parsed.map { case (label, indices, values) =>
      LabeledPoint(label, Vectors.sparse(d, indices, values))
    }
  }

  def evaluate(model:FMModel,data:RDD[LabeledPoint]): Double ={

    // evaluate
    val predictionAndLabels = data.map{ case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    // Instantiate metrics object
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val auROC = metrics.areaUnderROC
    auROC
  }


}


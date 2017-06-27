package org.apache.spark.mllib.regression

import org.apache.log4j.Logger

/**
  * Created by clz on 2017/6/27.
  */
class Util {

  val logger = Logger.getLogger("MY LOGGER")


  def rmHDFS(path:String) {
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
}




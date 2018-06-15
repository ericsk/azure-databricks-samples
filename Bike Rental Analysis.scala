// Databricks notebook source
// MAGIC %md #Bike Rental Analysis
// MAGIC 
// MAGIC **資料**: 資料來自於 [Capital Bikeshare](https://www.capitalbikeshare.com/) 系統 2011 - 2012 年的資料，並且加上了相關的欄位，例如：天氣。這份資料集由 Fanaee-T 與 Gama (2013) 整理，並且由 [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) 營運提供。
// MAGIC 
// MAGIC **目標**: 預測出每個時段單車的需求量。

// COMMAND ----------

// MAGIC %md ## 把資料從 Azure Storage 讀取進來
// MAGIC 
// MAGIC 首先，我們把從 UCI 下載來的資料，上傳 _hour.csv_ 檔案到 Azure Storage Blob 中。

// COMMAND ----------

dbutils.fs.mount(
  source = "wasbs://bikesharing@skhol.blob.core.windows.net/",
  mountPoint = "/mnt/bike",
  extraConfigs = Map("fs.azure.account.key.skhol.blob.core.windows.net" -> ""))

// COMMAND ----------

val dfRaw = spark.read.format("csv").option("header", "true").load("/mnt/bike/hour.csv")
dfRaw.show()

// COMMAND ----------

println("資料筆數: " + dfRaw.count())

// COMMAND ----------

val dfProcessed = dfRaw.drop("instant").drop("dteday").drop("casual").drop("registered")
dfProcessed.show()

// COMMAND ----------

dfProcessed.printSchema()

// COMMAND ----------

val dfTransformed = dfProcessed.select(dfProcessed.columns.map(c => dfProcessed.col(c).cast("double")): _*)
dfTransformed.printSchema()

// COMMAND ----------

val Array(dataTrain, dataTest) = dfTransformed.randomSplit(Array(0.7, 0.3))
println(s"接下來會用 ${dataTrain.count()} 筆資料做訓練，而用 ${dataTest.count()} 筆資料作為測試")

// COMMAND ----------

display(dataTrain.select("hr", "cnt"))

// COMMAND ----------

val cols = dfTransformed.columns
val featureCols = cols.filter(!_.contains("cnt"))



// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorIndexer

val va = new VectorAssembler()
            .setInputCols(featureCols)
            .setOutputCol("rawFeatures");

val vi = new VectorIndexer()
              .setInputCol("rawFeatures")
              .setOutputCol("features")
              .setMaxCategories(4)

// COMMAND ----------

import org.apache.spark.ml.regression.GBTRegressor

val gbt = new GBTRegressor().setLabelCol("cnt")

// COMMAND ----------

import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.RegressionEvaluator

// Define a grid of hyperparameters to test:
//  - maxDepth: max depth of each decision tree in the GBT ensemble
//  - maxIter: iterations, i.e., number of trees in each GBT ensemble
// In this example notebook, we keep these values small.  In practice, to get the highest accuracy, you would likely want to try deeper trees (10 or higher) and more trees in the ensemble (>100).
val paramGrid = new ParamGridBuilder()
                      .addGrid(gbt.maxDepth, Array(2, 5))
                      .addGrid(gbt.maxIter, Array(10, 100))
                      .build()
// We define an evaluation metric.  This tells CrossValidator how well we are doing by comparing the true labels with predictions.
val evaluator = new RegressionEvaluator()
                    .setMetricName("rmse")
                    .setLabelCol(gbt.getLabelCol)
                    .setPredictionCol(gbt.getPredictionCol)
// Declare the CrossValidator, which runs model tuning for us.
val cv = new CrossValidator()
              .setEstimator(gbt)
              .setEvaluator(evaluator)
              .setEstimatorParamMaps(paramGrid)


// COMMAND ----------

import org.apache.spark.ml.Pipeline

val pipeline = new Pipeline().setStages(Array(va, vi, cv))

// COMMAND ----------

val pipelineModel = pipeline.fit(dataTrain)

// COMMAND ----------

val predictions = pipelineModel.transform(dataTest)

// COMMAND ----------

val a = featureCols.toSeq


// COMMAND ----------

val tailCols = "prediction" +: featureCols
display(predictions.select("cnt",  tailCols:_*))

// COMMAND ----------

val rmse = evaluator.evaluate(predictions)
println(s"Root-Mean-Square-Error on our test set: ${rmse}")

// COMMAND ----------

display(dataTrain.select("hr", "cnt"))
display(predictions.select("hr", "prediction"))

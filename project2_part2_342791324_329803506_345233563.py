
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
import os
from pyspark.ml.linalg import Vectors

def init_spark(app_name: str):
  spark = SparkSession.builder.appName(app_name).config("spark.kryoserializer.buffer.max","512m").config("spark.driver.memory","9g").getOrCreate()
  sc = spark.sparkContext
  return spark, sc

spark, sc = init_spark('demo')

denseVec = Vectors.dense(0, 2.0, 3.0)
size = 3
idx = [1, 2]
values = [2.0, 3.0]
sparseVec = Vectors.sparse(size, idx, values)
df = spark.read.json("data.json").filter(f.col("gt") != 'null')
df.dropna()

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(
    inputCols=["User", "gt","Creation_Time"],
    outputCols=["User_col", "label","Ct_lab"])

model = indexer.fit(df)
output = model.transform(df)

from pyspark.ml.feature import OneHotEncoder

encoder = OneHotEncoder(inputCol="User_col", outputCol="User_vec")

model = encoder.fit(output)
output = model.transform(output)

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=["User_vec", "x","y","z","Creation_Time"],
    outputCol="features")

output = assembler.transform(output)

from pyspark.ml import Pipeline

transformer_pipeline = Pipeline(stages=[
    StringIndexer(inputCols=["User", "gt"], outputCols=["User_col", "label"]),
    OneHotEncoder(inputCol="User_col", outputCol="User_vec"),
    VectorAssembler(inputCols=["User_vec", "x","y","z","Creation_Time"], outputCol="features")
])

model = transformer_pipeline.fit(df)
prepared_df = model.transform(df)

train, test = prepared_df.randomSplit([0.7, 0.3])
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="label",featuresCol="features")
from pyspark.sql import SQLContext
from pyspark.ml.linalg import Vectors, VectorUDT
fittedLR = lr.fit(train)
fittedLR.transform(test).select('gt', 'label', 'probability', 'prediction')
lrModel = fittedLR.transform(test).select('features','label')
train, test = lrModel.randomSplit([0.7, 0.3])

#Decision Tree Model

from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dtModel = dt.fit(train)
dt_results = dtModel.transform(test)
dt_accuracy = dt_results.filter(dt_results.label == dt_results.prediction).count() / dt_results.count()
print("Decision Tree Accuracy:\n", dt_accuracy)


#Random Forest Regressor Model

from pyspark.ml.classification import RandomForestClassifier

rfClassifier = RandomForestClassifier()
trainedModel = rfClassifier.fit(train)
rf_results= trainedModel.transform(test)
rf_accuracy = rf_results.filter(rf_results.label == rf_results.prediction).count() / rf_results.count()
print("Random Forest Accuracy:\n", rf_accuracy)

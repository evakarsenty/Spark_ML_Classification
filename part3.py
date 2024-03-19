from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import *
import os
import time
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.functions import expr
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import col, when
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import copy


SCHEMA = StructType([StructField("Arrival_Time", LongType(), True),
                     StructField("Creation_Time", LongType(), True),
                     StructField("Device", StringType(), True),
                     StructField("Index", LongType(), True),
                     StructField("Model", StringType(), True),
                     StructField("User", StringType(), True),
                     StructField("gt", StringType(), True),
                     StructField("x", DoubleType(), True),
                     StructField("y", DoubleType(), True),
                     StructField("z", DoubleType(), True)])

spark = SparkSession.builder.appName('demo_app') \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .getOrCreate()

os.environ['PYSPARK_SUBMIT_ARGS'] = \
    "--packages=org.apache.spark:spark-sql-kafka-0-10_2.12:2.4.8,com.microsoft.azure:spark-mssql-connector:1.0.1"
kafka_server = 'dds2020s-kafka.eastus.cloudapp.azure.com:9092'
topic = "activities"


streaming = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_server) \
    .option("subscribe", topic) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", False) \
    .option("maxOffsetsPerTrigger", 432) \
    .load() \
    .select(f.from_json(f.decode("value", "US-ASCII"), schema=SCHEMA).alias("value")).select("value.*")

simpleTransform = streaming \
    .where("gt is not null") \
    .select("gt", "model", "User", "x", "y", "z", "Creation_Time", "Index") \
    .writeStream \
    .queryName("simple_transform") \
    .format("memory") \
    .outputMode("append") \
    .start()

static = spark.read \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_server) \
    .option("subscribe", topic) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", False) \
    .option("maxOffsetsPerTrigger", 432) \
    .load() \
    .select(f.from_json(f.decode("value", "US-ASCII"), schema=SCHEMA).alias("value")).select("value.*")


def data_organization(data):
    data = data.filter(data.User != 'null')
    data = data.withColumn("user_id", when(col("User") == 'a', 1).when(col("User") == 'b', 2)
                           .when(col("User") == 'c', 3).when(col("User") == 'd', 4).
                           when(col("User") == 'e', 5).when(col("User") == 'f', 6) \
                           .when(col("User") == 'g', 7).when(col("User") == 'h', 8) \
                           .when(col("User") == 'i', 9))

    data_col = static.columns
    assembler = VectorAssembler(
        inputCols=["user_id", "x", "y", "z", "Index", "Creation_Time"],
        outputCol="features")
    output = assembler.transform(data)
    output = output.filter(output.gt != 'null')
    output = output.withColumn("label", when(col("gt") == 'stand', 1).when(col("gt") == 'sit', 2)
                               .when(col("gt") == 'walk', 3).when(col("gt") == 'stairsup', 4).
                               when(col("gt") == 'stairsdown', 5).when(col("gt") == 'bike', 6))
    output = output.drop(*data_col)
    return output
    

time.sleep(25)
sample = static.sample(False, 0.005)
sample = data_organization(sample)

rfClassifier = RandomForestClassifier(labelCol="label", featuresCol="features", maxBins=42, maxDepth=10, numTrees=50)
fittedModel = rfClassifier.fit(sample)

train_nb = 0
length = 20
total_count = 0

for i in range(length):
    df = spark.sql("SELECT * FROM simple_transform")
    total_count = df.count()

    if total_count == 0:
        trainedModel = fittedModel.transform(data_organization(df))
        MCEvaluator = MulticlassClassificationEvaluator(metricName="accuracy")
        print("\taccuracy: {}".format(MCEvaluator.evaluate(trainedModel)))
        time.sleep(5)
        train_nb = df.count()
    else:
        fittedModel_after = rfClassifier.fit(data_organization(df))
        total_count = df.count()
        test_nb = total_count - train_nb

        inv_df = df.withColumn('index_y', f.monotonically_increasing_id()).orderBy(f.col('index_y').desc()).drop(
            'index_y')
     
        test_df = inv_df.limit(test_nb)
        print("\n\tNb of train elements: {}\t Iteration number: {}".format(train_nb, i))
        print("\tNb of test elements: {}\t\t Iteration number: {}\n".format(test_nb, i))
        trainedModel = fittedModel.transform(data_organization(test_df))
        MCEvaluator = MulticlassClassificationEvaluator(metricName="accuracy")
        print("\tAccuracy result: {}".format(MCEvaluator.evaluate(trainedModel)))

    if total_count == 0:
        fittedModel = rfClassifier.fit(data_organization(df))
    else:
        fittedModel = fittedModel_after
        train_nb = total_count
    time.sleep(5)


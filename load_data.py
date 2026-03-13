from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("LoadLogistics").getOrCreate()

# Example logistics dataset (replace with your real file)
df = spark.read.csv("logistics.csv", header=True, inferSchema=True)

df.show(5)
df.printSchema()

spark.stop()
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("LogisticsDataProcessing") \
    .master("local[*]") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

print("Spark Session Created Successfully!")

spark.stop()
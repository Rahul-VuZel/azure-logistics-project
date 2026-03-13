from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col

spark = SparkSession.builder.appName("TransformLogistics").getOrCreate()

df = spark.read.csv("logistics.csv", header=True, inferSchema=True)

# Delivery speed category
df = df.withColumn(
    "delivery_category",
    when(col("delivery_time") <= 3, "Fast")
    .when(col("delivery_time") <= 7, "Medium")
    .otherwise("Slow")
)

# Delay flag
df = df.withColumn(
    "is_delayed",
    when(col("status") == "Delayed", 1).otherwise(0)
)

df.show()

spark.stop()

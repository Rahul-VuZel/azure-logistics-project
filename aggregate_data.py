from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, sum

spark = SparkSession.builder.appName("AggregateLogistics").getOrCreate()

df = spark.read.csv("logistics.csv", header=True, inferSchema=True)

# Recreate delay flag (needed for aggregation)
from pyspark.sql.functions import when, col
df = df.withColumn("is_delayed", when(col("status") == "Delayed", 1).otherwise(0))

agg_df = df.groupBy("warehouse").agg(
    count("*").alias("total_shipments"),
    avg("delivery_time").alias("avg_delivery_time"),
    sum("is_delayed").alias("total_delays")
)

agg_df.show()

spark.stop()
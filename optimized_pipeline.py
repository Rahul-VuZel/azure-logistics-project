from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, avg, count, sum

spark = SparkSession.builder \
    .appName("OptimizedLogisticsPipeline") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

df = spark.read.csv("logistics.csv", header=True, inferSchema=True)

# Cache dataset for repeated operations
df.cache()

# Cleaning
df = df.filter(col("shipment_id").isNotNull())
df = df.dropDuplicates(["shipment_id"])

# Transformation
df = df.withColumn("is_delayed", when(col("status") == "Delayed", 1).otherwise(0))

# Aggregation
agg_df = df.groupBy("warehouse").agg(
    count("*").alias("total_shipments"),
    avg("delivery_time").alias("avg_delivery_time"),
    sum("is_delayed").alias("total_delays")
)

# Save results (optimized format)
agg_df.write.mode("overwrite").parquet("output_parquet")

agg_df.show()

spark.stop()
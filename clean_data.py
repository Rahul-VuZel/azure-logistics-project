from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("CleanLogistics").getOrCreate()

df = spark.read.csv("logistics.csv", header=True, inferSchema=True)

# Remove rows with missing shipment_id
df = df.filter(col("shipment_id").isNotNull())

# Remove duplicates
df = df.dropDuplicates(["shipment_id"])

# Remove invalid delivery times (negative values)
df = df.filter(col("delivery_time") >= 0)

df.show()

spark.stop()
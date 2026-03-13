from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator

# Start Spark session
spark = SparkSession.builder \
    .appName("Logistics ML") \
    .getOrCreate()

# Load CSV dataset
df = spark.read.csv("logistics.csv", header=True, inferSchema=True)

df.show(5)

# Encode categorical columns
warehouse_indexer = StringIndexer(inputCol="warehouse", outputCol="warehouse_index")
status_indexer = StringIndexer(inputCol="status", outputCol="label")

df = warehouse_indexer.fit(df).transform(df)
df = status_indexer.fit(df).transform(df)

# Features
assembler = VectorAssembler(
    inputCols=["warehouse_index", "delivery_time"],
    outputCol="features"
)

df = assembler.transform(df)

# ✅ ADD THIS LINE HERE
df.select("warehouse", "warehouse_index", "features").show(5)

train, test = df.randomSplit([0.8, 0.2])

# Model 1: Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train)
lr_pred = lr_model.transform(test)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
lr_accuracy = evaluator.evaluate(lr_pred)

print("Logistic Regression Accuracy:", lr_accuracy)

# Model 2: Decision Tree 
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
dt_model = dt.fit(train)
dt_pred = dt_model.transform(test)

dt_accuracy = evaluator.evaluate(dt_pred)
print("Decision Tree Accuracy:", dt_accuracy)

# Model 3: Linear Regression 
lr_reg = LinearRegression(featuresCol="features", labelCol="delivery_time")

reg_model = lr_reg.fit(train)
reg_pred = reg_model.transform(test)

reg_eval = RegressionEvaluator(
    labelCol="delivery_time",
    predictionCol="prediction",
    metricName="rmse"
)

rmse = reg_eval.evaluate(reg_pred)

print("Linear Regression RMSE:", rmse)
input("Press Enter to stop Spark...")
spark.stop()
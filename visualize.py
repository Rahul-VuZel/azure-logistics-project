from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt

# Connect to local MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["logistics_db"]
collection = db["shipments"]

# Load data into pandas
data = list(collection.find({}, {"_id": 0}))
df = pd.DataFrame(data)

print(df.head()) 


# Visualization 1: Avg delivery time by warehouse
avg_delivery = df.groupby("warehouse")["delivery_time"].mean()

avg_delivery.plot(kind="bar")
plt.title("Average Delivery Time by Warehouse")
plt.xlabel("Warehouse")
plt.ylabel("Average Delivery Time")
plt.show()


# Visualization 2: Shipment status count
status_count = df["status"].value_counts()

status_count.plot(kind="bar")
plt.title("Shipment Status Distribution")
plt.xlabel("Status")
plt.ylabel("Count")
plt.show()


# Visualization 3: Delivery time distribution
df["delivery_time"].plot(kind="hist", bins=10)
plt.title("Delivery Time Distribution")
plt.xlabel("Delivery Time")
plt.ylabel("Frequency")
plt.show()
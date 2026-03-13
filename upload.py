import csv
from pymongo import MongoClient

# Connect to local MongoDB
client = MongoClient("mongodb://localhost:27017")  

# Access database and collection
db = client["logistics_db"]
collection = db["shipments"]

# Read CSV file
csv_file_path = "logistics.csv"  # My CSV is in the same folder
data = []

with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Convert numeric fields to integers
        row["shipment_id"] = int(row["shipment_id"])
        row["delivery_time"] = int(row["delivery_time"])
        data.append(row)

# Insert data into MongoDB
if data:
    result = collection.insert_many(data)
    print(f"Inserted {len(result.inserted_ids)} documents into MongoDB.")
else:
    print("No data found.")

# Verify insertion
for doc in collection.find():
    print(doc)
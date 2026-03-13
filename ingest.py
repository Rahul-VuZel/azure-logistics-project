from azure.storage.blob import BlobServiceClient
import os
import logging

# Azure connection string
connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
container_name = "raw-data"
folder_path = "./datasets"

# logging setup
logging.basicConfig(filename='ingestion.log', level=logging.INFO)

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# loop through dataset folder
for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)

    # VALIDATION STEP (this is the line you asked about)
    if os.path.isfile(file_path):

        with open(file_path, "rb") as data:
            container_client.upload_blob(name=file, data=data, overwrite=True)

            print(f"Uploaded: {file}")
            logging.info(f"{file} uploaded successfully")

print("All datasets uploaded successfully!")

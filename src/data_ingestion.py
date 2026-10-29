# data_ingestion.py
# This file is responsible for LOADING data into our project.
import os
import sys
import urllib.request
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import Azure AND Snowflake settings from our config file
from config import (
    AZURE_CONN_STR, AZURE_CONTAINER, 
    SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_ACCOUNT,
    SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA
)

# Where to save all raw (unmodified) data
RAW_DIR = "data/01_raw"

SAMPLE_DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/wdbc.data"
)

# Column names that match the UCI dataset format
SAMPLE_DATASET_COLS = ["id", "diagnosis"] + ["feature_" + str(i) for i in range(1, 31)]


def load_local_file(filepath):
    """
    Load a CSV or Excel file that is already on your computer.
    """
    print("\n[Ingestion] Loading local file: " + filepath)

    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith(".xlsx") or filepath.endswith(".xls"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file type. Please use .csv or .xlsx")

    filename = os.path.basename(filepath)
    dest = os.path.join(RAW_DIR, filename)
    
    # Ensure the directory exists before saving
    os.makedirs(RAW_DIR, exist_ok=True)
    df.to_csv(dest, index=False)

    print("[Ingestion] Saved raw copy to: " + dest)
    print("[Ingestion] Rows loaded: " + str(len(df)))
    return df


def download_sample_dataset():
    """
    Download the sample (UCI) dataset automatically from the internet.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    dest = os.path.join(RAW_DIR, "sample_raw.csv")

    if os.path.exists(dest):
        print("[Ingestion] Sample file already downloaded, skipping: " + dest)
    else:
        print("[Ingestion] Downloading sample dataset from UCI repository")
        urllib.request.urlretrieve(SAMPLE_DATASET_URL, dest)
        print("[Ingestion] Download complete! Saved to: " + dest)

    df = pd.read_csv(dest, header=None, names=SAMPLE_DATASET_COLS)
    print("[Ingestion] Loaded " + str(len(df)) + " rows with " + str(df.shape[1]) + " columns")
    return df


def load_from_azure(blob_name):
    """
    Download a CSV file from your Azure Blob Storage container.
    """
    if AZURE_CONN_STR == "CONNECTION_STRING":
        raise ValueError(
            "Please open config.py and replace 'CONNECTION_STRING' "
        )

    print("\n[Ingestion] Connecting to Azure")
    print("[Ingestion] Container: " + AZURE_CONTAINER + " | File: " + blob_name)

    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        raise ImportError(
            "Azure library not found! Run this in your terminal:\n"
            "    pip install azure-storage-blob azure-ai-ml azure-identity"
        )

    client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
    blob_client = client.get_blob_client(container=AZURE_CONTAINER, blob=blob_name)

    os.makedirs(RAW_DIR, exist_ok=True)
    dest = os.path.join(RAW_DIR, blob_name)
    with open(dest, "wb") as f:
        f.write(blob_client.download_blob().readall())

    print("[Ingestion] Azure download complete! Saved to: " + dest)

    df = pd.read_csv(dest)
    print("[Ingestion] Loaded " + str(len(df)) + " rows from Azure")
    return df


def load_dataframe_to_snowflake(df, table_name="RAW_LOGISTICS"):
    """
    Uploads a Pandas DataFrame directly into Snowflake.
    """
    print(f"\n[Ingestion] Connecting to Snowflake to upload {table_name}...")
    
    try:
        # 1. Connect WITHOUT specifying the database initially
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT
        )
        
        # 2. Force Snowflake to wake up our Warehouse, Database, and Schema
        print(f"[Ingestion] Waking up warehouse and ensuring database '{SNOWFLAKE_DATABASE}' exists...")
        
        # Create the warehouse if it's missing, and turn it on
        conn.cursor().execute(f"CREATE WAREHOUSE IF NOT EXISTS {SNOWFLAKE_WAREHOUSE} WITH WAREHOUSE_SIZE='X-SMALL' AUTO_SUSPEND=60 AUTO_RESUME=TRUE")
        conn.cursor().execute(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}")
        
        # Create the database and schema
        conn.cursor().execute(f"CREATE DATABASE IF NOT EXISTS {SNOWFLAKE_DATABASE}")
        conn.cursor().execute(f"USE DATABASE {SNOWFLAKE_DATABASE}")
        
        conn.cursor().execute(f"CREATE SCHEMA IF NOT EXISTS {SNOWFLAKE_SCHEMA}")
        conn.cursor().execute(f"USE SCHEMA {SNOWFLAKE_SCHEMA}")
        
        # 3. Write the pandas DataFrame to Snowflake
        success, num_chunks, num_rows, output = write_pandas(conn, df, table_name, auto_create_table=True)
        
        if success:
            print(f"[Ingestion] SUCCESS: Uploaded {num_rows} rows to Snowflake table {table_name}.")
        else:
            print("[Ingestion] Failed to upload to Snowflake.")
            
    except Exception as e:
        print(f"[Ingestion] ERROR connecting to Snowflake: {e}")
        
    finally:
        if 'conn' in locals():
            conn.close()
            print("[Ingestion] Snowflake connection closed.")


if __name__ == "__main__":
    # 1. Load the local healthcare logistics dataset
    df = load_local_file("Raw_Data.csv")
    
    # 2. Push it securely to the cloud
    load_dataframe_to_snowflake(df, table_name="RAW_LOGISTICS")
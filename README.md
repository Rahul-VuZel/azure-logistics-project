# 💊 Healthcare Supply Chain Intelligence

[![Live Demo](https://img.shields.io/badge/Live_Dashboard-Azure_App_Service-blue?logo=microsoftazure)](https://logistics-dashboard-chaudgb2fnevcfbf.westus3-01.azurewebsites.net)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python)](#)
[![Snowflake](https://img.shields.io/badge/Snowflake-Data_Warehouse-29B5E8?logo=snowflake)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?logo=streamlit)](#)

An end-to-end Analytics Engineering data pipeline designed to ingest, transform, and visualize global healthcare logistics data. This project demonstrates full-stack cloud data engineering capabilities, transitioning raw supply chain metrics into an interactive, cloud-hosted executive dashboard.

## 🏗️ Cloud Architecture & Tech Stack

1. **Data Ingestion (Python & Pandas):** Raw supply chain CSV data is processed and ingested programmatically using a custom Python pipeline (`src/data_ingestion.py`).
2. **Cloud Data Warehouse (Snowflake):** Data is loaded into a Snowflake database (`LOGISTICS_DB`), where it is cleaned and modeled into a production-ready view (`CLEAN_SUPPLY_CHAIN`).
3. **Data Visualization (Streamlit):** An interactive front-end dashboard (`app.py`) queries the Snowflake warehouse directly using Snowflake's Python connector, generating real-time insights on total shipments, freight costs, and delivery modes.
4. **Cloud Deployment (Microsoft Azure):** The dashboard is continuously deployed on a Linux-based Azure App Service, providing public internet access with zero-downtime integration from GitHub.

## 📊 Key Dashboard Metrics
* **Total Freight Costs:** Aggregated global shipping expenditures, automatically handling nulls and string-to-numeric casting.
* **Top Destinations:** Bar chart analysis of the highest-cost delivery countries.
* **Shipment Modes:** Distribution analysis between Air, Ocean, and Truck freight.

## 📂 Project Structure
```text
├── app.py                  # Main Streamlit dashboard application
├── requirements.txt        # Production Python dependencies for Azure
├── src/
│   └── data_ingestion.py   # Pipeline script connecting local data to Snowflake
└── README.md               # Project documentation
import streamlit as st
import snowflake.connector
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Healthcare Logistics Dashboard", page_icon="💊", layout="wide")

st.title("💊 Healthcare Supply Chain Intelligence Dashboard")
st.markdown("End-to-end data pipeline powered by **Snowflake** and visualized via **Streamlit & Azure**.")

# Initialize Snowflake Connection
@st.cache_resource
def init_connection():
    from config import SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_ACCOUNT, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA
    return snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )

try:
    conn = init_connection()
    
    # Query data using the safer cursor method
    @st.cache_data(ttl=600)
    def load_data():
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM LOGISTICS_DB.PUBLIC.CLEAN_SUPPLY_CHAIN LIMIT 1000;")
        df_data = cursor.fetch_pandas_all()
        cursor.close()
        return df_data

    df = load_data()

    # FORCE the freight cost column to be numbers, turning text notes into blanks (NaN)
    df['FREIGHT_COST_USD'] = pd.to_numeric(df['FREIGHT_COST_USD'], errors='coerce')

    # --- TOP METRICS ROW ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Shipments Tracked", f"{len(df):,}")
    
    # Calculate total cost, ignoring the blanks (NaNs)
    total_cost = df['FREIGHT_COST_USD'].sum(skipna=True)
    col2.metric("Total Freight Cost", f"${total_cost:,.2f}")
    
    col3.metric("Destinations Served", df['DESTINATION_COUNTRY'].nunique())

    st.markdown("---")

    # --- CHARTS SECTION ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top Freight Costs by Destination")
        cost_by_country = df.groupby('DESTINATION_COUNTRY')['FREIGHT_COST_USD'].sum().nlargest(10)
        st.bar_chart(cost_by_country)

    with col_b:
        st.subheader("Shipment Modes Breakdown")
        mode_counts = df['SHIPMENT_MODE'].value_counts()
        st.bar_chart(mode_counts)

    # --- DATA TABLE PREVIEW ---
    st.subheader("📋 Cleaned Supply Chain Data Preview")
    st.dataframe(df.head(50))

except Exception as e:
    st.error(f"Error connecting to Snowflake or loading data: {e}")
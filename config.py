import os
from dotenv import load_dotenv

# --- SNOWFLAKE SETUP ---
# Load the hidden variables from our .env file
load_dotenv()

SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")

# The default compute and storage settings Snowflake creates for new accounts
SNOWFLAKE_WAREHOUSE = "COMPUTE_WH"
SNOWFLAKE_DATABASE = "LOGISTICS_DB"
SNOWFLAKE_SCHEMA = "PUBLIC"

AZURE_CONN_STR = "CONNECTION_STRING"
AZURE_CONTAINER = "healthcare-logistics"
# Logistics Label settings
# These are the class names we care about in our logistics dataset.
# "Delayed" means the shipment did NOT arrive on time.
# "On-Time" means the shipment arrived as scheduled.
# (In the Week 6 breast cancer project these were "M" and "B" - same idea!)
LABEL_DELAYED = "Delayed"
LABEL_ON_TIME = "On-Time"

# ML Model settings 
# 70% of data for training, 30% for testing - same ratio used in Week 6
TEST_SIZE = 0.30

# Fixed random seed - this ensures we get the same split and results every run.
# Any number works, 42 is just a convention in the ML world.
RANDOM_STATE = 42

# Reinforcement Learning settings 
# How many episodes to train the Q-Learning agent (from Week 4)
# More episodes = smarter agent, but slower training
RL_EPISODES = 500

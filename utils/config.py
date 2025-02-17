import os

# Configuration file for setting up ticker and data parameters

CONFIG = {
    "ticker": "SPY",  # Default ticker, can be changed
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
    "data_dir": "data/raw/",  # Directory to save market data
    "processed_data_dir": "data/processed/",  # Directory for processed data
}

def get_config():
    """Returns the configuration settings."""
    return CONFIG

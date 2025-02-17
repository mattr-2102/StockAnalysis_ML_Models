import pandas as pd
import yfinance as yf
import os
from utils.config import get_config

# Load configuration settings
config = get_config()
ticker = config["ticker"]
start_date = config["start_date"]
end_date = config["end_date"]
data_dir = config["data_dir"]

def fetch_market_data(ticker=ticker, start_date=start_date, end_date=end_date, save_dir=data_dir):
    """
    Fetches OHLCV market data for a given ticker from Yahoo Finance and saves it to a CSV file.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{ticker}_market_data.csv")
    
    print(f"Fetching market data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError("No market data fetched. Check ticker and date range.")
    
    df.to_csv(save_path)
    print(f"Market data saved to {save_path}")
    return df

def load_market_data(ticker, load_dir="data/raw/"):
    """Loads market data for a given ticker from CSV and returns a Pandas DataFrame."""
    load_path = os.path.join(load_dir, f"{ticker}_market_data.csv")

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Market data file not found: {load_path}")

    # Read CSV while skipping the first three rows (Price, Ticker, and Date row)
    df = pd.read_csv(load_path, skiprows=2)

    # Debugging: Print columns before renaming
    print(f"Columns before renaming: {df.columns.tolist()}")

    # Rename columns to match expected structure
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

    # Debugging: Print columns after renaming
    print(f"Columns after renaming: {df.columns.tolist()}")

    # Ensure 'Date' column exists
    if 'Date' not in df.columns:
        raise ValueError(f"Expected 'Date' column not found. Available columns: {df.columns.tolist()}")

    # Set 'Date' as index and convert to datetime
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)

    print(f"Market data loaded successfully from {load_path}")
    return df
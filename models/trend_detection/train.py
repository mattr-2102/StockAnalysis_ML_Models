import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from utils.data_loader import load_market_data
from utils.data_loader import fetch_market_data
from utils.feature_engineering import compute_trend_features
from models.trend_detection.model import TrendDetectionModel
from utils.config import get_config

# Load configuration settings
config = get_config()
ticker = config["ticker"]
data_dir = config["data_dir"]

def prepare_data(ticker):
    """Loads, processes, and prepares the data for training."""
    data_path = os.path.join(data_dir, f"{ticker}_market_data.csv")
    
    if not os.path.exists(data_path):
        print(f"Market data for {ticker} not found. Fetching new data...")
        fetch_market_data(ticker)

    df = load_market_data(ticker)
    df = compute_trend_features(df)  # Compute trend-related features
    df.dropna(inplace=True)  # Remove NaNs after feature engineering
    
    print("Trend Label Distribution:")
    print(df['Trend_Label'].value_counts())
    
    # Define feature columns (excluding date & target labels)
    feature_cols = [col for col in df.columns if col not in ['Trend_Label']]
    X = df[feature_cols].values
    
    # Define target labels (Uptrend / Sideways / Downtrend)
    y = df['Trend_Label'].values.reshape(-1, 1)
    
    # One-Hot Encode Labels
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)
    
    # Normalize Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_model():
    """Trains the CNN-LSTM model for trend detection."""
    print("📊 Loading and preparing data...")
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(ticker)
    input_shape = (X_train.shape[1], 1)  # Required shape for LSTM

    # Reshape data for LSTM input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Initialize Model
    print("🚀 Initializing Trend Detection Model...")
    model_instance = TrendDetectionModel(input_shape=input_shape, num_classes=3)
    model = model_instance.get_model()

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)

    # Train Model
    print("📈 Training Model...")
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, lr_schedule]
    )

    # Save Model
    model.save("models/trend_detection/trend_model.keras")
    print("✅ Model training complete. Saved to models/trend_detection/trend_model.keras")

if __name__ == "__main__":
    train_model()

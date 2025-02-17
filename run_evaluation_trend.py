import numpy as np
import tensorflow as tf
from utils.data_loader import load_market_data
from utils.feature_engineering import compute_trend_features  # Import trend-specific features
from utils.config import get_config
from models.trend_detection.utils import load_model
from utils.evaluation import evaluate_model  # Import evaluation function

# Load configuration settings
config = get_config()
ticker = config["ticker"]

# Load test data
df = load_market_data(ticker)
df = compute_trend_features(df)  # Use trend-specific feature computation
df.dropna(inplace=True)  # Remove NaNs

# Prepare features and labels
feature_cols = [col for col in df.columns if col not in ['Trend_Label']]
X_test = df[feature_cols].values
y_test = df['Trend_Label'].values.reshape(-1, 1)

# One-Hot Encode Labels
from sklearn.preprocessing import OneHotEncoder, StandardScaler
encoder = OneHotEncoder(sparse_output=False)
y_test_encoded = encoder.fit_transform(y_test)

# Normalize Features
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Reshape for LSTM
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Load trained model
model = load_model("models/trend_detection/trend_model.keras")  # Update path to trend model

# Run evaluation using the function from utils.evaluation
evaluate_model(model, X_test_scaled, y_test_encoded)

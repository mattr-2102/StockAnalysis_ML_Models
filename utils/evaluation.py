import numpy as np
import tensorflow as tf
from utils.data_loader import load_market_data
from utils.feature_engineering import compute_volatility_features
from models.volatility.utils import load_model, evaluate_model
from utils.config import get_config

# Load configuration settings
config = get_config()
ticker = config["ticker"]

def evaluate_volatility_model():
    """Loads the trained model, runs evaluation on test data, and prints metrics."""
    # Load test data
    df = load_market_data(ticker)
    df = compute_volatility_features(df)
    df.dropna(inplace=True)  # Remove NaNs
    
    # Prepare features and labels
    feature_cols = [col for col in df.columns if col not in ['Volatility_Label']]
    X_test = df[feature_cols].values
    y_test = df['Volatility_Label'].values.reshape(-1, 1)
    
    # One-Hot Encode Labels
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    y_test_encoded = encoder.fit_transform(y_test)
    
    # Normalize Features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    # Reshape for LSTM
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
    
    # Load trained model
    model = load_model("models/volatility/volatility_model.keras")
    
    # Evaluate model
    evaluate_model(model, X_test_scaled, y_test_encoded)

if __name__ == "__main__":
    evaluate_volatility_model()

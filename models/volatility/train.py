import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
import shap
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from utils.data_loader import load_market_data, fetch_market_data
from utils.feature_engineering import compute_volatility_features
from models.volatility.model import VolatilityModel
from utils.config import get_config

# Load configuration settings
config = get_config()
ticker = config["ticker"]
data_dir = config["data_dir"]

def prepare_data(ticker):
    """Loads, processes, and prepares the data for training."""
    data_path = os.path.join(data_dir, f"{ticker}_market_data.csv")
    
    # Check if the CSV exists, if not, fetch the data
    if not os.path.exists(data_path):
        print(f"Market data for {ticker} not found. Fetching new data...")
        fetch_market_data(ticker)
    
    df = load_market_data(ticker)
    df = compute_volatility_features(df)
    df.dropna(inplace=True)  # Remove NaNs after feature engineering
    
    # Define feature columns (excluding date, target label)
    feature_cols = [col for col in df.columns if col not in ['Volatility_Label']]
    X = df[feature_cols].values
    
    # Define target labels (Volatility Classification: Low, Moderate, High)
    y = df['Volatility_Label'].values.reshape(-1, 1)
    
    # One-Hot Encode Labels
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)
    
    # Normalize Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, feature_cols

def objective(trial):
    """Objective function for Optuna hyperparameter tuning."""
    print(f"\n🚀 Starting Trial {trial.number + 1} 🚀")

    X_train, X_test, y_train, y_test, feature_cols = prepare_data(ticker)
    input_shape = (X_train.shape[1], 1)

    # Hyperparameter tuning
    conv1_filters = trial.suggest_categorical("conv1_filters", [32, 64, 128])
    conv2_filters = trial.suggest_categorical("conv2_filters", [64, 128, 256])
    lstm_units = trial.suggest_categorical("lstm_units", [64, 128, 256])
    dense_units = trial.suggest_categorical("dense_units", [32, 64, 128])
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    print(f"📌 Trial {trial.number} Hyperparameters: "
          f"conv1={conv1_filters}, conv2={conv2_filters}, lstm={lstm_units}, "
          f"dense={dense_units}, dropout={dropout_rate}, lr={learning_rate}, batch={batch_size}")

    # Initialize Model
    model_instance = VolatilityModel(
        input_shape=input_shape,
        num_classes=3,
        conv1_filters=conv1_filters,
        conv2_filters=conv2_filters,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    model = model_instance.get_model()

    print(f"🛠️ Model Compiled for Trial {trial.number}")

    # Reduce LR if validation loss plateaus
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=0.0001)

    # Train Model
    history = model.fit(X_train, y_train, epochs=5, batch_size=batch_size, validation_data=(X_test, y_test), 
                        callbacks=[lr_schedule], verbose=1)  # <-- Enable verbose mode here!

    val_accuracy = max(history.history['val_accuracy'])

    print(f"✅ Trial {trial.number} Finished: Best Validation Accuracy = {val_accuracy:.4f}")

    return val_accuracy


def train_with_optuna():
    """Runs Optuna optimization to find best hyperparameters and trains final model with best params."""
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    print("Best hyperparameters:", study.best_params)

    # Retrieve the best hyperparameters
    best_params = study.best_params

    # Prepare data
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(ticker)
    input_shape = (X_train.shape[1], 1)

    # Train model with best params
    best_model_instance = VolatilityModel(
        input_shape=input_shape,
        num_classes=3,
        **best_params
    )
    best_model = best_model_instance.get_model()

    # Reduce LR if validation loss plateaus
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=0.0001)

    # Train with best hyperparameters
    best_model.fit(X_train, y_train, epochs=30, batch_size=best_params["batch_size"], 
                   validation_data=(X_test, y_test), callbacks=[lr_schedule])

    # Save the best model
    best_model.save("models/volatility/volatility_model_best.keras")
    print("Training complete with best hyperparameters. Model saved as models/volatility/volatility_model_best.keras.")

if __name__ == "__main__":
    train_with_optuna()

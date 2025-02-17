import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import classification_report, confusion_matrix

# Label mapping for volatility classification
VOLATILITY_LABELS = {
    0: "Low Volatility",
    1: "Moderate Volatility",
    2: "High Volatility"
}

def encode_labels(y):
    """Encodes categorical volatility labels into numerical values."""
    label_map = {"Low Volatility": 0, "Moderate Volatility": 1, "High Volatility": 2}
    return np.array([label_map[label] for label in y])

def decode_labels(y_pred):
    """Decodes numerical predictions into human-readable labels."""
    return [VOLATILITY_LABELS[np.argmax(pred)] for pred in y_pred]

def save_model(model, save_path="models/volatility/volatility_model.keras"):
    """Saves the trained model as a Keras file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved successfully to {save_path}")

def load_model(load_path="models/volatility/volatility_model.keras"):
    """Loads a trained Keras model from file."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file not found: {load_path}")
    model = tf.keras.models.load_model(load_path)
    print(f"Model loaded from {load_path}")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and prints classification metrics."""
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    print("Classification Report:")
    print(classification_report(y_test_labels, y_pred_labels, target_names=list(VOLATILITY_LABELS.values())))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_labels, y_pred_labels))

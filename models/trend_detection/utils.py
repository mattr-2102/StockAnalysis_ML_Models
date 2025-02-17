import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import classification_report, confusion_matrix

# Label mapping for trend classification
TREND_LABELS = {
    0: "Downtrend",
    1: "Sideways",
    2: "Uptrend"
}

def encode_labels(y):
    """Encodes categorical trend labels into numerical values."""
    label_map = {"Downtrend": 0, "Sideways": 1, "Uptrend": 2}
    return np.array([label_map[label] for label in y])

def decode_labels(y_pred):
    """Decodes numerical predictions into human-readable labels."""
    return [TREND_LABELS[np.argmax(pred)] for pred in y_pred]

def save_model(model, save_path="models/trend_detection/trend_model.keras"):
    """Saves the trained model as a Keras file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"✅ Model saved successfully to {save_path}")

def load_model(load_path="models/trend_detection/trend_model.keras"):
    """Loads a trained Keras model from file."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"❌ Model file not found: {load_path}")
    model = tf.keras.models.load_model(load_path)
    print(f"📂 Model loaded from {load_path}")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and prints classification metrics."""
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    print("📊 Classification Report:")
    print(classification_report(y_test_labels, y_pred_labels, target_names=list(TREND_LABELS.values())))
    
    print("📌 Confusion Matrix:")
    print(confusion_matrix(y_test_labels, y_pred_labels))

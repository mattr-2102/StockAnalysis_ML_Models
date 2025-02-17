import os
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

def save_model(model, save_path):
    """Saves the trained model."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"✅ Model saved successfully to {save_path}")

def load_model(load_path):
    """Loads a trained model from file."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file not found: {load_path}")
    model = tf.keras.models.load_model(load_path)
    print(f"📂 Model loaded from {load_path}")
    return model

def evaluate_model(model, X_test, y_test, class_labels):
    """Evaluates the trained model and prints classification metrics."""
    y_pred = model.predict(X_test)
    y_pred_labels = y_pred.argmax(axis=1)
    y_test_labels = y_test.argmax(axis=1)

    print("\n📊 Classification Report:")
    print(classification_report(y_test_labels, y_pred_labels, target_names=class_labels))

    print("\n📌 Confusion Matrix:")
    print(confusion_matrix(y_test_labels, y_pred_labels))

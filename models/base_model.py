import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, input_shape, num_classes=3, config=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.config = config or {}

        # Extract model parameters from config (with defaults)
        self.conv1_filters = self.config.get("conv1_filters", 64)
        self.conv2_filters = self.config.get("conv2_filters", 128)
        self.lstm_units = self.config.get("lstm_units", 128)
        self.dense_units = self.config.get("dense_units", 64)
        self.dropout_rate = self.config.get("dropout_rate", 0.4)
        self.learning_rate = self.config.get("learning_rate", 0.0005)
        self.batch_size = self.config.get("batch_size", 64)

        self.model = self.build_model()

    @abstractmethod
    def build_model(self):
        """To be implemented by specific models (Trend/Volatility)"""
        pass

    def compile_model(self):
        """Compile the model with standard optimizer and loss function."""
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train(self, X_train, y_train, X_test, y_test, epochs=30):
        """Train the model with common callbacks."""
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-4)

        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr]
        )

    def get_model(self):
        """Returns the compiled model."""
        return self.model

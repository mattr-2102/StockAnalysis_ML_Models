import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class VolatilityModel:
    def __init__(self, input_shape, num_classes=3, conv1_filters=64, conv2_filters=128, 
                 lstm_units=128, dense_units=64, dropout_rate=0.4, learning_rate=0.0005, batch_size=64):
        """
        Initializes the VolatilityModel with hyperparameters that can be optimized by Optuna.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = self.build_model()

    def build_model(self):
        """Builds the CNN-LSTM model for volatility classification."""
        inputs = Input(shape=self.input_shape)

        # CNN Layers (Feature Extraction)
        x = Conv1D(filters=self.conv1_filters, kernel_size=3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(filters=self.conv2_filters, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)

        # LSTM Layers (Time-Series Memory)
        x = tf.keras.layers.Reshape((-1, 1))(x)  # Reshape for LSTM
        x = LSTM(self.lstm_units, return_sequences=True)(x)
        x = LSTM(self.lstm_units // 2, return_sequences=False)(x)
        
        # Fully Connected Layers
        x = Dense(self.dense_units, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(self.dense_units // 2, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)

        # Output Layer (Softmax for Classification)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        
        return model

    def get_model(self):
        """Returns the compiled model."""
        return self.model

    def train(self, X_train, y_train, X_test, y_test, epochs=50):
        """Trains the model with early stopping and learning rate reduction."""
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-4)
        
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=self.batch_size, 
                       validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

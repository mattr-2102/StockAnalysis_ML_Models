from models.base_model import BaseModel
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Flatten, BatchNormalization

class TrendDetectionModel(BaseModel):
    def build_model(self):
        inputs = Input(shape=self.input_shape)

        # CNN Feature Extraction
        x = Conv1D(filters=self.conv1_filters, kernel_size=3, activation="relu", padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(filters=self.conv2_filters, kernel_size=3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)

        # LSTM Time-Series Processing
        x = LSTM(self.lstm_units, return_sequences=True)(x)
        x = LSTM(self.lstm_units // 2, return_sequences=False)(x)

        # Fully Connected Layers
        x = Dense(self.dense_units, activation="relu")(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(self.dense_units // 2, activation="relu")(x)
        x = Dropout(self.dropout_rate)(x)

        outputs = Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.models.Model(inputs, outputs)
        self.compile_model()
        return model

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

class LstmModelHelper:
    """Class to prepare dataset."""

    def create_lstm_dataset(self, data, look_back=1):
        """Creates a dataset for LSTM based on the look_back period."""
        X, y = [], []
        for i in range(look_back, len(data)):
            X.append(data[i-look_back:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def normalize_data(self, data, feature_range=(0, 1)):
        """Normalizes the data using MinMaxScaler."""
        scaler = MinMaxScaler(feature_range=feature_range)
        return scaler.fit_transform(data), scaler


    def build_lstm_model(self, input_shape):
        """Builds and compiles an LSTM model."""
        model = Sequential()
        model.add(Input(shape=input_shape))  # Two features: price and timestamp
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Predict price based on the current date-time timestamp
    def predict_next_price(self, model, current_timestamp, prices, timestamps, price_scaler, timestamp_scaler, look_back=60):
        """Predict price based on the current timestamp."""
        timestamp_numeric = current_timestamp.timestamp()
        timestamp_numeric = np.array([[timestamp_numeric]])

        # Normalize the timestamp
        timestamp_numeric = timestamp_scaler.transform(timestamp_numeric)

        # Prepare the input data (last known prices and the timestamp feature)
        latest_prices = prices[-look_back:].reshape(1, -1, 1)
        latest_timestamps = timestamps[-look_back:].reshape(1, -1, 1)

        # Combine the last prices and the new timestamp
        combined_data = np.concatenate((latest_prices, latest_timestamps), axis=-1)

        # Add the new timestamp to the combined data
        new_data = np.array([[[latest_prices[-1, 0, 0], timestamp_numeric[0, 0]]]])
        combined_data = np.append(combined_data[:, 1:, :], new_data, axis=1)

        prediction = model.predict(combined_data)
        prediction = price_scaler.inverse_transform(prediction)

        return prediction
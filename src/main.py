import os
# Disable OneDNN optimizations for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from lstm_model_helper import LstmModelHelper
import numpy as np
import pandas as pd
from historical_data_generator import HistoricalDataGenerator
from sklearn.model_selection import train_test_split


# Generate historical data with prices and timestamps
historical_data_generator = HistoricalDataGenerator()
historical_data = historical_data_generator.generate_historical_data()

# Display last 5 rows of the data
print('----------------------------------')
print(historical_data.tail())
print('----------------------------------')

model_helper = LstmModelHelper()

# Normalize the price data
prices = historical_data['price'].values.reshape(-1, 1)
prices, price_scaler = model_helper.normalize_data(prices)

# Normalize the timestamp data
timestamps_numeric = historical_data['timestamp'].astype('int64') / 10**9
timestamps = timestamps_numeric.values.reshape(-1, 1)
timestamps, timestamp_scaler = model_helper.normalize_data(timestamps)

# Combine normalized prices and timestamps
combined_data = np.hstack((prices, timestamps))
look_back = min(60, len(combined_data) - 1)
if look_back <= 0:
    raise ValueError('Not enough historical data for the look_back period.')

X, y = model_helper.create_lstm_dataset(combined_data, look_back=look_back)

# Reshape for LSTM input (keeping both features)
X = np.reshape(X, (X.shape[0], X.shape[1], 2))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = model_helper.build_lstm_model((look_back, 2))

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=7, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Predict price for the current timestamp
current_timestamp = pd.Timestamp.now()
next_price = model_helper.predict_next_price(model, current_timestamp, prices, timestamps, price_scaler, timestamp_scaler, look_back)

print('----------------------------------')
print(f'Estimated price for {current_timestamp}: {next_price[0, 0]:.2f}')



# Incremental training
recent_data = HistoricalDataGenerator().generate_recent_data()

# Display last 5 rows of the data
print('----------------------------------')
print(recent_data.tail())
print('----------------------------------')

# Normalize the price data
recent_prices = recent_data['price'].values.reshape(-1, 1)
recent_prices = price_scaler.transform(recent_prices)  # Use the existing price_scaler

# Normalize the timestamp data
recent_timestamps_numeric = recent_data['timestamp'].astype('int64') / 10**9
recent_timestamps = recent_timestamps_numeric.values.reshape(-1, 1)
recent_timestamps = timestamp_scaler.transform(recent_timestamps)  # Use the existing timestamp_scaler

# Combine normalized prices and timestamps
recent_combined_data = np.hstack((recent_prices, recent_timestamps))
look_back = min(60, len(recent_combined_data) - 1)
if look_back <= 0:
    raise ValueError('Not enough recent data for the look_back period.')

X_recent, y_recent = model_helper.create_lstm_dataset(recent_combined_data, look_back=look_back)
X_recent = np.reshape(X_recent, (X_recent.shape[0], X_recent.shape[1], 2))

model.fit(X_recent, y_recent, batch_size=32, epochs=3, verbose=1)

# Predict price using updated model
next_price_updated = model_helper.predict_next_price(model, current_timestamp, recent_prices, recent_timestamps, price_scaler, timestamp_scaler, look_back)

print('----------------------------------')
print(f'Estimated price after incremental training for {current_timestamp}: {next_price_updated[0, 0]:.2f}')
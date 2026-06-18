import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Step 1: Load stock data
stock = "AAPL"  # Change to any stock symbol
data = yf.download(stock, start="2015-01-01", end="2024-01-01")

# Use only 'Close' price
data = data[['Close']]

# Step 2: Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Step 3: Create training dataset
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Step 4: Build LSTM model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Step 6: Prepare test data
test_data = scaled_data[train_size - 60:]

x_test = []
y_test = scaled_data[train_size:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Step 7: Predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Step 8: Plot results
train = data[:train_size]
valid = data[train_size:].copy()
valid['Predictions'] = predictions

plt.figure(figsize=(12,6))
plt.title(f"{stock} Stock Price Prediction")
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Actual','Predicted'])
plt.show()

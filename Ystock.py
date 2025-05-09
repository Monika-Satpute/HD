import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Download Google stock price data
data = yf.download("GOOG", start="2015-01-01", end="2023-12-31")
print(data.head())  # Show first few rows

# Step 2: Visualize the closing prices
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='GOOG Close Price')
plt.title('Google Stock Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Step 3: Preprocess (Use only 'Close' price for prediction)
close_data = data[['Close']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_data)

# Create sequences
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 60
X, y = create_sequences(scaled_data, window_size)

# Split into train and test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 4: Build RNN model
model = Sequential([
    SimpleRNN(50, activation='tanh', input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Step 5: Train with EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, validation_split=0.1, callbacks=[early_stop])

# Step 6: Evaluate
y_pred = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Plot prediction vs actual
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual Price')
plt.plot(y_pred_inv, label='Predicted Price')
plt.title('GOOG Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Step 7: Save model
model.save("google_stock_rnn_model.h5")

# Show one prediction
print("Sample prediction:", y_pred_inv[0][0])

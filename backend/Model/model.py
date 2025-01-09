
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess data
df_stock = pd.read_csv("NULBdata.csv")
df_stock = df_stock.rename(columns={'% Change': 'Change'})
df_stock['Date'] = pd.to_datetime(df_stock['Date'], format='%d/%m/%Y')
df_stock['Ltp'] = df_stock['Ltp'].str.replace(',', '').astype(float)

# Sort data and reset index
data = df_stock.sort_values(by='Date').reset_index(drop=True)

# Extract LTP values for training
ltp_data = data['Ltp'].values.reshape(-1, 1)

# Scale data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(ltp_data)

# Create training and test datasets
training_size = int(len(scaled_data) * 0.70)
train_data = scaled_data[:training_size]
test_data = scaled_data[training_size:]

# Create datasets for time-series prediction
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step, 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape data for LSTM input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=1)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE
train_rmse = np.sqrt(np.mean((train_predict - scaler.inverse_transform(y_train.reshape(-1, 1)))**2))
test_rmse = np.sqrt(np.mean((test_predict - scaler.inverse_transform(y_test.reshape(-1, 1)))**2))
print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")

# Visualize results
train_index = range(time_step, len(train_predict) + time_step)
test_index = range(len(train_data) + time_step, len(data))

plt.figure(figsize=(15, 8))
plt.plot(data['Ltp'], label="Actual LTP")
plt.plot(train_index, train_predict.flatten(), label="Train Predictions")
plt.plot(test_index, test_predict.flatten(), label="Test Predictions")
plt.xlabel("Time")
plt.ylabel("LTP Value")
plt.title("LTP Predictions vs Actual")
plt.legend()
plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import datetime
import os

# Step 1: Load historical data
def load_historical_data(file_path):
    # data = pd.read_csv(file_path)
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True, dayfirst=True)
    data['Date'] = pd.to_datetime(data.index, dayfirst=True) 
    data=data.rename(columns={'% Change':'Change'})
    data['Open'] = data['Open'].astype(str)
    data['High'] = data['High'].astype(str)
    data['Low'] = data['Low'].astype(str)
    data['Ltp'] = data['Ltp'].astype(str)
    data['Qty'] = data['Qty'].astype(str)
    data['Turnover'] = data['Turnover'].astype(str)


    data['Open'] = data['Open'].str.replace(',', '').astype(float)
    data['High'] = data['High'].str.replace(',', '').astype(float)
    data['Low'] = data['Low'].str.replace(',', '').astype(float)
    data['Ltp'] = data['Ltp'].str.replace(',', '').astype(float)
    data['Qty'] = data['Qty'].str.replace(',', '').astype(float)
    data['Turnover'] = data['Turnover'].str.replace(',', '').astype(float)
    data['Change'] = data['Change'].astype(float)
    return data

# Step 2: Preprocess data
def preprocess_data(data):
    """Preprocess the data for model training."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Ltp']])
    return scaled_data, scaler

# Step 3: Prepare data for training
def create_sequences(data, sequence_length):
    """Create sequences for training and target data."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :-1])
        y.append(data[i + sequence_length, -1])
    return np.array(X), np.array(y)

# Step 4: Build the LSTM model
def build_lstm_model(input_shape):
    """Build an LSTM model."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Step 5: Train the model
def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    """Train the LSTM model."""
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

# Step 6: Make predictions
def predict_next_day(model, scaled_data, scaler, sequence_length):
    last_sequence = scaled_data[-sequence_length:]  # Extract the last sequence
    last_sequence = np.array(last_sequence).reshape(1, sequence_length, 4)  # Reshape to (1, 60, 4)
    
    print("Input shape for prediction:", last_sequence.shape)  # Debugging
    
    prediction = model.predict(last_sequence)
    prediction = scaler.inverse_transform(prediction)
    return prediction



# Step 7: Append predicted value to the dataset
def append_prediction(data, prediction, scraped_data):
    """Append the predicted value to the dataset."""
    next_date = data['Date'].iloc[-1] + pd.Timedelta(days=1)
    new_entry = {
        'Date': next_date,
        'Open': np.nan,  # Fill with scraped data if available
        'High': np.nan,  # Fill with scraped data if available
        'Low': np.nan,   # Fill with scraped data if available
        'Ltp': prediction,
        'Change': np.nan,
        'Qty': np.nan,
        'Turnover': np.nan
    }
    # Merge with scraped data if fields are available
    data = data.append(new_entry, ignore_index=True)
    return data

# Step 8: Plot actual vs predicted values
def plot_predictions(actual, predicted):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual Values')
    plt.plot(predicted, label='Predicted Values')
    plt.legend()
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Load historical data
    historical_data = load_historical_data("/Users/pragyanadhikari/Desktop/majorProjectUI/backend/data/NULB.csv")

    # Preprocess data
    scaled_data, scaler = preprocess_data(historical_data)

    # Prepare training data
    sequence_length = 60
    X, y = create_sequences(scaled_data, sequence_length)

    # Split data into training and testing sets
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Build and train the model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model = train_model(model, X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform([[0, 0, 0, p[0]] for p in predictions])[:, -1]

    scrapped_data_path="/Users/pragyanadhikari/Desktop/majorProjectUI/ShareSansarDataScrape/Data"
    todays_date=datetime.datetime.now()
    file_name = todays_date.strftime("%Y_%m_%d.csv")

    
    todays_data=os.path.join(scrapped_data_path,file_name)
    # Append the next prediction
    next_day_prediction = predict_next_day(model, scaled_data, scaler, sequence_length)
    historical_data = append_prediction(historical_data, next_day_prediction, todays_data)

    # Plot results
    plot_predictions(historical_data['Ltp'].iloc[-len(predictions):], predictions)

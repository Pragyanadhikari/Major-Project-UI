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
    """Load and preprocess historical data from a CSV file."""
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True, dayfirst=True)
    data['Date'] = pd.to_datetime(data.index, dayfirst=True)
    data = data.rename(columns={'% Change': 'Change'})

    # Convert columns to numeric, handling commas and missing values
    numeric_columns = ['Open', 'High', 'Low', 'Ltp', 'Qty', 'Turnover']
    for col in numeric_columns:
        data[col] = data[col].astype(str).str.replace(',', '').astype(float)
    data['Change'] = data['Change'].astype(float)

    return data

# Step 2: Preprocess data
def preprocess_data(data):
    """Scale the data for model training."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Ltp']])
    return scaled_data, scaler

# Step 3: Prepare data for training
def create_sequences(data, sequence_length):
    """Create sequences for training and target data."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :])
        y.append(data[i + sequence_length, -1])  # Predict the Ltp
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
    """Predict the next day's LTP value."""
    last_sequence = scaled_data[-sequence_length:]  # Extract the last sequence
    last_sequence = np.array(last_sequence).reshape(1, sequence_length, scaled_data.shape[1])  # Reshape to (1, 60, 4)
    prediction = model.predict(last_sequence)
    return scaler.inverse_transform([[0, 0, 0, prediction[0][0]]])[:, -1]

# Step 7: Append predicted value to the dataset
def append_prediction(data, prediction, scraped_data_path):
    """Append the predicted value to the dataset."""
    next_date = data.index[-1] + pd.Timedelta(days=1)
    prediction_value = prediction[0]  # Extract the scalar prediction value

    # Load scraped data for the current day
    scraped_data = pd.read_csv(scraped_data_path)
    scraped_data['Symbol'] = scraped_data['Symbol'].astype(str)

    # Find the row corresponding to "NULB"
    nulb_row = scraped_data[scraped_data['Symbol'] == 'NULB']

    # Extract relevant fields if found, else use NaN
    open_value = nulb_row['Open'].iloc[0] if not nulb_row.empty else np.nan
    high_value = nulb_row['High'].iloc[0] if not nulb_row.empty else np.nan
    low_value = nulb_row['Low'].iloc[0] if not nulb_row.empty else np.nan
    turnover_value = nulb_row['Turnover'].iloc[0] if not nulb_row.empty else np.nan

    # Create a new entry with the predicted LTP value
    new_entry = pd.DataFrame([{
        'Date': next_date,
        'Open': open_value,
        'High': high_value,
        'Low': low_value,
        'Ltp': prediction_value,
        'Change': np.nan,
        'Qty': np.nan,
        'Turnover': turnover_value,
    }])

    # Concatenate the new entry to the existing DataFrame
    data = pd.concat([data, new_entry.set_index('Date')])
    return data


# Step 8: Plot actual vs predicted values
def plot_predictions(actual, predicted):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual Values')
    plt.plot(predicted, label='Predicted Values')
    plt.legend()
    plt.show()

# Main execution
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

    # Make predictions for the test set
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform([[0, 0, 0, p[0]] for p in predictions])[:, -1]

    # Determine today's scraped data file
    scraped_data_dir = "/Users/pragyanadhikari/Desktop/majorProjectUI/ShareSansarDataScrape/Data"
    todays_date = datetime.datetime.now()
    file_name = todays_date.strftime("%Y_%m_%d.csv")
    todays_data_path = os.path.join(scraped_data_dir, file_name)

    # Append the next prediction
    next_day_prediction = predict_next_day(model, scaled_data, scaler, sequence_length)
    historical_data = append_prediction(historical_data, next_day_prediction, todays_data_path)

    # Plot results
    plot_predictions(historical_data['Ltp'].iloc[-len(predictions):], predictions)

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('./Model/stock_lstm_model2.h5')  # Adjust the path accordingly


scaler = joblib.load("./Model/scaler.pkl")  # Update with your scaler path

class DataProcessor:
    def __init__(self, scaler=None):
        self.scaler = scaler

    def preprocess_data(self, data, time_steps=60):
        """
        Preprocess the input data for prediction.
        
        Args:
            data (pd.Series, pd.DataFrame, or np.ndarray): The LTP values to preprocess.
            time_steps (int): Number of time steps to look back.
        
        Returns:
            np.ndarray: Scaled and reshaped data.
        """
        # Clean the data by removing commas and converting to float
        if isinstance(data, pd.Series):
            data = data.replace({',': ''}, regex=True)  # Remove commas
            data = data.astype(float)  # Convert to float
        
        # Convert to NumPy array if input is a Pandas DataFrame or Series
        if isinstance(data, (pd.Series, pd.DataFrame)):
            data = data.to_numpy()  # Convert to NumPy array
    
        data = data.reshape(-1, 1)  # Reshape to (n_samples, 1)
    
        if self.scaler:
            data = self.scaler.transform(data)  # Scale if scaler is provided
    
        # Prepare the last `time_steps` data points
        data = data[-time_steps:]
        return data.reshape(1, time_steps, 1)

# Initialize Flask app and data processor
app = Flask(__name__)
data_processor = DataProcessor(scaler=scaler)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input CSV file path from the request (in JSON format)
    data = request.json
    input_data = data["input_data"]  # Expecting CSV file path
    
    # Load and preprocess the input data (CSV file)
    data_df = pd.read_csv(input_data)
    
    # Assuming LTP column is the one we need to use for prediction
    ltp_data = data_df['Ltp']  # Adjust based on your data

    # Preprocess data using the DataProcessor class
    preprocessed_data = data_processor.preprocess_data(ltp_data)
    
    # Make prediction
    prediction = model.predict(preprocessed_data)
    
    # Return the prediction as a JSON response
    return jsonify({"prediction": prediction[0].tolist()})  # Convert to list if needed

if __name__ == '__main__':
    app.run(debug=True)

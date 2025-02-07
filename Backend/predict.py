# predict_trades.py

import torch
import pandas as pd
import numpy as np
from stock_trading_env import StockTradingEnv
from ppo import PPOAgent
from preprocessdate import preprocessDate
from datetime import timedelta
from flask import *
import json
from flask import Flask, send_file, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

class TradingPredictor:
    def __init__(self, model_path, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the agent
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=1e-4,
            batch_size=256,
            n_epochs=10
        )
        
        # Load the trained model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.agent.actor_critic.eval()
        
        # Load normalization parameters if they exist
        if 'running_mean' in checkpoint and 'running_std' in checkpoint:
            self.agent.running_mean = checkpoint['running_mean']
            self.agent.running_std = checkpoint['running_std']
    
    def predict_action(self, state):
        """
        Predict the next action given the current state
        Returns: action, action_probability, predicted_value
        """
        # Normalize state
        state = self.agent.normalize_observation(state)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.agent.actor_critic(state)
            
            # Get the action with highest probability
            action = torch.argmax(action_probs).item()
            probability = action_probs[0][action].item()
            
        return action, probability, value.item()

def calculate_technical_indicators(df):
    """Calculate all required technical indicators"""
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Calculate Returns
    df['Returns'] = df['Close'].pct_change()
    
    # Calculate MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate SMAs
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate EMA
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Calculate Volatility (20-day standard deviation)
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Calculate ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Calculate Volume Ratio (comparing to 20-day average)
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    # Calculate MFI (Money Flow Index)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    money_ratio = positive_flow / negative_flow
    df['MFI'] = 100 - (100 / (1 + money_ratio))
    
    return df

def load_and_preprocess_data(csv_path):
    """Load and preprocess the latest market data"""
    # Read CSV
    df = pd.read_csv(csv_path)
    
    try:
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        # Convert numeric columns, handling commas
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Percent Change', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by date and reset index
        df = df.sort_values('Date', ascending=True).reset_index(drop=True)
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Forward fill any NaN values
        df = df.fillna(method='ffill')
        
        # Remove any remaining NaN rows (important for initial periods where indicators cannot be calculated)
        df = df.dropna()
        
        # Reset index again after removing NaN rows
        df = df.reset_index(drop=True)
        
        # Print verification of required columns
        required_columns = ['Close', 'Returns', 'MACD', 'Signal', 'RSI', 'SMA_20', 
                          'SMA_50', 'EMA_20', 'Volatility', 'ATR', 'Volume_Ratio', 'MFI']
        print("\nVerifying required columns:")
        for col in required_columns:
            print(f"{col}: {col in df.columns}")
        
        # Create environment to preprocess data
        env = StockTradingEnv(df)
        return env, df
        
    except Exception as e:
        print(f"Error during data preprocessing: {str(e)}")
        print("DataFrame columns:", df.columns.tolist())
        raise

def interpret_action(action, probability, value):
    """Convert numerical action to human-readable recommendation"""
    actions = {
        0: "HOLD",
        1: "BUY",
        2: "SELL"
    }
    
    return {
        "recommendation": actions[action],
        "confidence": f"{probability:.2%}",
        "predicted_value": f"{value:.2f}"
    }
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

def main(MODEL_PATH, DATA_PATH):
    STATE_DIM = 15  
    ACTION_DIM = 3  

    try:
        print("Loading model...")
        predictor = TradingPredictor(MODEL_PATH, STATE_DIM, ACTION_DIM)
        
        print("Loading market data...")
        df = pd.read_csv(DATA_PATH)
        df = preprocessDate(df)

        print("Initializing environment...")
        env = StockTradingEnv(
            df=df,
            initial_balance=10000,
            max_shares=10,
            transaction_fee_percent=0.001
        )

        print("Resetting environment...")
        initial_observation = env.reset()

        print("\nPredicting next action...")
        action, prob, value = predictor.predict_action(initial_observation)
        result = interpret_action(action, prob, value)

        print("\nTrading Recommendation:")
        print("-" * 50)
        last_date = pd.to_datetime(df['Date'].iloc[-1], dayfirst=True) 
        next_date = last_date + timedelta(days=1)

        print(f"Date: {df['Date'].iloc[-1]}") 
        print(f"Actual Price: {df['Close'].iloc[-1]}")
        print(f"Action: {result['recommendation']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Predicted Value: {result['predicted_value']}")
        print("-" * 50)

        df = df.iloc[-20:]

        # Convert 'Date' column to datetime format for proper date manipulation
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        # Simulate the next day's stock data based on recommendation
        last_row = df.iloc[-1]
        last_date = last_row['Date']
        last_close = last_row['Close']

        # Example recommendation (this should come from your logic)
        recommendation = 'BUY'  # Replace with actual logic

        # Logic for the next day's stock price
        if recommendation == 'BUY':
            next_close = last_close + 5
        elif recommendation == 'SELL':
            next_close = last_close - 5
        else:  # 'HOLD'
            next_close = last_close

        # Create the next day's date (add one day)
        next_date = last_date + timedelta(days=1)

        # Create a new row for the next day with proper date format
        new_row = pd.DataFrame({'Date': [next_date], 'Close': [next_close]})

        # Concatenate the new row to the dataframe
        df = pd.concat([df, new_row], ignore_index=True)

        # Ensure the 'Date' column is of datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Plot the updated stock price graph
        plt.figure(figsize=(10, 6))

        # Plot the close prices (including the last predicted value)
        plt.plot(df['Date'], df['Close'], label='Close Price', marker='o', linestyle='-', color='blue')

        # Highlight the last value with different color and add label for prediction
        if recommendation == 'BUY':
            plt.plot(df['Date'].iloc[-1], df['Close'].iloc[-1], marker='^', linestyle='-', color='green')
            prediction_text = "Prediction: BUY"
        elif recommendation == 'SELL':
            plt.plot(df['Date'].iloc[-1], df['Close'].iloc[-1], marker='v', linestyle='-', color='red')
            prediction_text = "Prediction: SELL"
        else:  # 'HOLD'
            plt.plot(df['Date'].iloc[-1], df['Close'].iloc[-1], marker='>', linestyle='-', color='grey')
            prediction_text = "Prediction: SELL"

        # Title and labels
        plt.text(df['Date'].iloc[-1], df['Close'].iloc[-1] + 2, prediction_text, color='black', ha='center')
        plt.title('LLBS Stock Price with Buy/Sell Actions')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()

        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=45)
        plt.gcf().autofmt_xdate()

        # Save and display the plot
        plt.savefig("stock_price_plot.png", bbox_inches='tight')
        plt.show()
        # Print the most recent market data
        # print("\nRecent Market Data:")
        # print(df.tail()[['Date', 'Open', 'High', 'Low', 'Close', 'Percent Change']].to_string())

        # Render environment state (if applicable)
        env.render()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

@app.route('/predictNUBL', methods=['GET'])
def predictN():
    try:
        # Hardcoded paths for NUBl prediction
        model_path = 'Backend/NublModel.pth'
        data_path = "Backend/NUBL.csv"
        
        # Load predictor
        predictor = TradingPredictor(model_path, state_dim=15, action_dim=3)
        
        # Load and preprocess data
        df = pd.read_csv(data_path)
        df = preprocessDate(df)
        
        # Create environment
        env = StockTradingEnv(df, initial_balance=10000, max_shares=10, transaction_fee_percent=0.001)
        
        # Reset the environment
        initial_observation = env.reset()
        
        # Get prediction
        action, prob, value = predictor.predict_action(initial_observation)
        
        # Interpret results
        result = interpret_action(action, prob, value)

        # Create the plot
        df = df.iloc[-20:]

        # Convert 'Date' column to datetime format for proper date manipulation
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        # Simulate the next day's stock data based on recommendation
        last_row = df.iloc[-1]
        last_date = last_row['Date']
        last_close = last_row['Close']

        # Example recommendation (this should come from your logic)
        recommendation = 'BUY'  # Replace with actual logic

        # Logic for the next day's stock price
        if recommendation == 'BUY':
            next_close = last_close + 5
        elif recommendation == 'SELL':
            next_close = last_close - 5
        else:  # 'HOLD'
            next_close = last_close

        # Create the next day's date (add one day)
        next_date = last_date + timedelta(days=1)

        # Create a new row for the next day with proper date format
        new_row = pd.DataFrame({'Date': [next_date], 'Close': [next_close]})

        # Concatenate the new row to the dataframe
        df = pd.concat([df, new_row], ignore_index=True)

        # Ensure the 'Date' column is of datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Plot the updated stock price graph
        plt.figure(figsize=(10, 6))

        # Plot the close prices (including the last predicted value)
        plt.plot(df['Date'], df['Close'], label='Close Price', marker='o', linestyle='-', color='blue')

        # Highlight the last value with different color and add label for prediction
        if recommendation == 'BUY':
            plt.plot(df['Date'].iloc[-1], df['Close'].iloc[-1], marker='^', linestyle='-', color='green')
            prediction_text = "Prediction: BUY"
        elif recommendation == 'SELL':
            plt.plot(df['Date'].iloc[-1], df['Close'].iloc[-1], marker='v', linestyle='-', color='red')
            prediction_text = "Prediction: SELL"
        else:  # 'HOLD'
            plt.plot(df['Date'].iloc[-1], df['Close'].iloc[-1], marker='>', linestyle='-', color='grey')
            prediction_text = "Prediction: SELL"

        # Title and labels
        plt.text(df['Date'].iloc[-1], df['Close'].iloc[-1] + 2, prediction_text, color='black', ha='center')
        plt.title('NUBL Stock Price with Buy/Sell Actions')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()

        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=45)
        plt.gcf().autofmt_xdate()

        # Save the plot to a BytesIO object
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        # Convert the image to base64
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return the plot as an image
        # return send_file(img, mimetype='image/png')

        return jsonify({
            "date": df['Date'].iloc[-1],
            "actual_price": df['Close'].iloc[-1],
            "action": result["recommendation"],
            "confidence": result["confidence"],
            "predicted_value": result["predicted_value"],
            "img":img_base64,
            "mimetype":'image/png'
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predictLLBS', methods=['GET'])
def predictL():
    try:
        # Hardcoded paths for LLBS prediction
        model_path = 'Backend/LLBSmodel.pth'
        data_path = 'Backend/LLBS.csv'
        
        # Load predictor
        predictor = TradingPredictor(model_path, state_dim=15, action_dim=3)
        
        # Load and preprocess data
        df = pd.read_csv(data_path)
        df = preprocessDate(df)
        
        # Create environment
        env = StockTradingEnv(df, initial_balance=10000, max_shares=10, transaction_fee_percent=0.001)
        
        # Reset the environment
        initial_observation = env.reset()
        
        # Get prediction
        action, prob, value = predictor.predict_action(initial_observation)
        
        # Interpret results
        result = interpret_action(action, prob, value)

        
        # Create the plot
        df = df.iloc[-20:]

        # Convert 'Date' column to datetime format for proper date manipulation
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        # Simulate the next day's stock data based on recommendation
        last_row = df.iloc[-1]
        last_date = last_row['Date']
        last_close = last_row['Close']

        # Example recommendation (this should come from your logic)
        recommendation = 'BUY'  # Replace with actual logic

        # Logic for the next day's stock price
        if recommendation == 'BUY':
            next_close = last_close + 5
        elif recommendation == 'SELL':
            next_close = last_close - 5
        else:  # 'HOLD'
            next_close = last_close

        # Create the next day's date (add one day)
        next_date = last_date + timedelta(days=1)

        # Create a new row for the next day with proper date format
        new_row = pd.DataFrame({'Date': [next_date], 'Close': [next_close]})

        # Concatenate the new row to the dataframe
        df = pd.concat([df, new_row], ignore_index=True)

        # Ensure the 'Date' column is of datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Plot the updated stock price graph
        plt.figure(figsize=(10, 6))

        # Plot the close prices (including the last predicted value)
        plt.plot(df['Date'], df['Close'], label='Close Price', marker='o', linestyle='-', color='blue')

        # Highlight the last value with different color and add label for prediction
        if recommendation == 'BUY':
            plt.plot(df['Date'].iloc[-1], df['Close'].iloc[-1], marker='^', linestyle='-', color='green')
            prediction_text = "Prediction: BUY"
        elif recommendation == 'SELL':
            plt.plot(df['Date'].iloc[-1], df['Close'].iloc[-1], marker='v', linestyle='-', color='red')
            prediction_text = "Prediction: SELL"
        else:  # 'HOLD'
            plt.plot(df['Date'].iloc[-1], df['Close'].iloc[-1], marker='>', linestyle='-', color='grey')
            prediction_text = "Prediction: SELL"

        # Title and labels
        plt.text(df['Date'].iloc[-1], df['Close'].iloc[-1] + 2, prediction_text, color='black', ha='center')
        plt.title('LLBS Stock Price with Buy/Sell Actions')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()

        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=45)
        plt.gcf().autofmt_xdate()

        # Save the plot to a BytesIO object
        img = BytesIO()
        plt.savefig(img, format='png')

        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return the plot as an image
        # return send_file(img, mimetype='image/png')

        return jsonify({
            "date": df['Date'].iloc[-1],
            "actual_price": df['Close'].iloc[-1],
            "action": result["recommendation"],
            "confidence": result["confidence"],
            "predicted_value": result["predicted_value"],
            "img":img_base64,
            "mimetype":'image/png'
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=1234, threaded=True)
    # main('Backend/LLBSmodel.pth','Backend/LLBS.csv')
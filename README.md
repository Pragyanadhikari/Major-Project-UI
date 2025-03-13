# Stock Prediction Mobile App (Flutter & AI Integration)

This repository contains a **Flutter** mobile app designed for **stock prediction** using **AI models**. The app integrates machine learning models for stock price prediction, helping users make informed investment decisions. It utilizes data-driven predictions powered by **Temporal Fusion Transformer (TFT)** and other AI models to predict stock price movements and display results in an intuitive mobile interface.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [UI Design](#ui-design)
- [AI Model Integration](#ai-model-integration)
- [Usage](#usage)
- [Requirements](#requirements)
- [Run](#run)
- [License](#license)

## Overview

The **Stock Prediction Mobile App** is a **Flutter** app that allows users to view predicted stock prices and track their investments. The app connects to AI models that forecast stock price movements using historical data and real-time predictions. The predictions help users make data-driven decisions for their stock portfolio.

### Key Features:
- **Stock Prediction**: AI-driven predictions for stock prices (e.g., predicted next day's Close).
- **Portfolio Management**: Users can add stocks to their portfolio and track performance.
- **User Authentication**: Firebase-based user authentication system for personalized experience.
- **Visualizations**: Plots and graphs for predicted vs. actual stock prices.

## Features

- **Stock Selection**: Users can select a stock from a predefined list or search for a stock symbol.
- **Stock Prediction**: Displays predicted stock prices for the next day based on historical data.
- **Portfolio Tracking**: Users can add and manage stocks in their portfolio.
- **Graphical Visualization**: Predictive charts comparing actual vs. predicted values.
- **Tax Calculation**: Users can input their stock prices and calculate the profit or loss, including broker commission.

## UI Design

The **UI of the Stock Prediction App** is designed to be user-friendly, modern, and responsive. Key UI components include:

### Prediction Page
- **Select Company Name**: Select company name to display prediction.

### Portfolio Page
- **Portfolio Management**: Users can add stocks to their portfolio.

### Profil/Loss Calsulator Page
- **Prediction Input**: Inputs for base price, selling price, and number of stocks.
- **Calculation Results**: Displays profit/loss, broker commission, and tax details.
- **Scrollable Results**: Results are shown in a left-right format (e.g., text on the left, values on the right).


## AI Model Integration

This mobile app integrates AI models to predict stock prices. The **Temporal Fusion Transformer (TFT)** is used to predict the next day's stock price using historical stock data and **Proximal Policy Optimization (PPO)** for trading action to take.

### Workflow
1. **Data Collection**: Stock data is collected from nepsealpha.
2. **Data Preprocessing**: Data is cleaned and prepared for the model.
3. **Model Training**: The TFT model is trained using historical data to forecast future stock prices and PPO model is trained for trading action.
4. **Prediction**: The model generates predictions, which are displayed in the app.
5. **Continuous Improvement**: Predictions are updated regularly based on new data and retrained models.

### Predictive Features:
- **Open, High, Low, Close, Volume**: Used as features to train the AI model.
- **Historical Data**: Past stock data used to forecast the next day's stock price.
- **Model Evaluation**: Regularly evaluates the accuracy of predictions using metrics like Mean Squared Error (MSE).

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-prediction-mobile-app.git
   cd stock-prediction-mobile-app

## Requirements
flutter pub get

## Run
flutter run

## License 
This project is licensed under the MIT License - see the LICENSE file for details.
![License](https://img.shields.io/badge/license-MIT-blue.svg)

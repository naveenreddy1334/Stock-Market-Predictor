# Stock Price and Signal Predictor

This Python project provides a comprehensive framework for analyzing stock data, predicting price movements, and generating actionable trading signals. Leveraging machine learning and technical analysis, the `StockAnalyzer` class combines historical stock data with advanced algorithms to assist traders and investors in making informed decisions.

---

## Features

1. **Data Fetching**: Retrieve historical stock data from Yahoo Finance using the `yfinance` library.
2. **Technical Indicators**: Calculate various indicators like RSI, MACD, Bollinger Bands, SMA, and more.
3. **Machine Learning Models**:
   - **Classification Model**: Predict the direction of price movement (up or down).
   - **Regression Model**: Predict the next day's closing price.
4. **Volatility Analysis**: Calculate annualized volatility.
5. **Support & Resistance Levels**: Identify critical price levels.
6. **Trading Signals**: Provide buy, sell, or hold recommendations based on predictions.
7. **Model Persistence**: Save and load trained models for reuse.


---
## Setting Up a Virtual Environment

Using a virtual environment is a good practice to isolate project dependencies and avoid conflicts with system-wide packages.

### Steps to Create and Activate a Virtual Environment

1. **Create the Virtual Environment**:
   Run the following command in the root directory of your project:
   ```bash
   python -m venv venv
2. **Activate the Virtual Environment:**
   ```bash
   # On Windows:
      venv\Scripts\activate
   
   # On Mac/Linux:
      source venv/bin/activate
   

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sudhanvasp/Stock-Market-Prediction.git
   cd Stock-Market-Prediction
   
2. **Install the requirements.txt**:
   ```bash 
   pip install -r requirements.txt

2. **Run Ui.py to run the Gradio interface**:
   ```bash 
   python run ui.py
---

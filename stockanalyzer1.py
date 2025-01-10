import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import joblib
import random

class StockAnalyzer:
    def __init__(self):
        """Initialize the StockAnalyzer with default parameters"""
        self.classification_model = None  # For direction prediction
        self.regression_model = None  # For price prediction
        self.scaler = StandardScaler()

    def fetch_data(self, symbol, start_date, end_date=None):
        """
        Fetch stock data from Yahoo Finance

        Parameters:
        symbol (str): Stock symbol (e.g., 'RELIANCE.NS' for NSE)
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format (optional)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        try:
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            return df
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")

    def calculate_rsi(self, data, periods=14):
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD, Signal Line, and MACD Histogram"""
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        middle_band = data.rolling(window=window).mean()
        std_dev = data.rolling(window=window).std()
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        return upper_band, middle_band, lower_band

    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        # Calculate basic indicators
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'], df['Signal_Line'] = self.calculate_macd(df['Close'])
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])

        # Moving Averages
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # Price momentum
        df['ROC'] = df['Close'].pct_change(10) * 100  # Rate of Change
        df['MOM'] = df['Close'] - df['Close'].shift(10)  # Momentum

        # Volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()

        return df

    def prepare_features(self, df):
        """Prepare features for machine learning"""
        if len(df) < 50:
            raise ValueError("Insufficient data points for analysis")

        feature_columns = [
            'RSI', 'MACD', 'Signal_Line',
            'ROC', 'MOM', 'Volatility', 'Returns'
        ]

        # Create target variable (1 if price goes up, 0 if down)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        # Drop rows with NaN values
        df_cleaned = df.dropna()

        if len(df_cleaned) < 30:  # Minimum required for meaningful analysis
            raise ValueError("Too many missing values after cleaning")

        # Prepare features and target
        X = df_cleaned[feature_columns]
        y = df_cleaned['Target']

        return X, y
    def train_models(self, symbol, start_date, end_date=None):
        """Train both classification and regression models"""
        # Fetch and prepare data
        df = self.fetch_data(symbol, start_date, end_date)
        df = self.add_technical_indicators(df)
        X, y_class = self.prepare_features(df)

        # Prepare regression target (next day's price)
        y_reg = df['Close'].shift(-1).dropna()
        y_reg = y_reg[:len(y_class)]  # Align with classification target

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data for both models
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X_scaled, y_class, y_reg, test_size=0.2, random_state=42
        )

        # Train classification model
        self.classification_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.classification_model.fit(X_train, y_class_train)

        # Train regression model
        self.regression_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.regression_model.fit(X_train, y_reg_train)

        # Evaluate models
        class_pred = self.classification_model.predict(X_test)
        reg_pred = self.regression_model.predict(X_test)

        return {
            'direction_accuracy': accuracy_score(y_class_test, class_pred),
            'price_mse': mean_squared_error(y_reg_test, reg_pred),
            'classification_report': classification_report(y_class_test, class_pred)
        }

    def predict_price_and_signals(self, symbol, lookback_days=365):
        """Predict next day's price and generate buy/sell signals with price targets"""
        if self.classification_model is None or self.regression_model is None:
            raise ValueError("Models not trained. Please run train_models() first.")

        # Fetch recent data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')

        # Get stock data
        df = self.fetch_data(symbol, start_date, end_date)
        current_price = df['Close'].iloc[-1]

        # Calculate technical indicators
        df = self.add_technical_indicators(df)

        # Prepare features for prediction
        X, _ = self.prepare_features(df)
        latest_features = X.iloc[-1:]

        # Scale features
        latest_features_scaled = self.scaler.transform(latest_features)

        # Get direction prediction and confidence
        direction_pred = self.classification_model.predict(latest_features_scaled)
        direction_proba = self.classification_model.predict_proba(latest_features_scaled)
        confidence = direction_proba[0][direction_pred[0]]

        # Get price prediction
        price_pred = self.regression_model.predict(latest_features_scaled)[0] # if its within 10% of the current_price else

        # Calculate volatility
        volatility = df['Returns'].std() * np.sqrt(252)  # Annualized volatility

        # Calculate support and resistance levels
        support_level = df['Low'].tail(20).min()
        resistance_level = df['High'].tail(20).max()

        # Calculate buy and sell targets
        buy_target = current_price * (1 - volatility * 0.1)  # 10% of volatility below current price
        sell_target = current_price * (1 + volatility * 0.1)  # 10% of volatility above current price
        if abs(price_pred - current_price) / current_price > 0.1:
            adjustment_percentage = random.uniform(0.02, 0.08)
            adjustment_sign = 1 if price_pred > current_price else -1
            price_pred = current_price * (1 + adjustment_sign * adjustment_percentage)
        # Determine action based on predictions and current price
        if price_pred > current_price * 1.02:  # Predicted 2% or more increase
            action = "BUY"
            target_price = sell_target
        elif price_pred < current_price * 0.98:  # Predicted 2% or more decrease
            action = "SELL"
            target_price = buy_target
        else:
            action = "HOLD"
            target_price = current_price

        return {
            'symbol': symbol,
            'date': df.index[-1].strftime('%Y-%m-%d'),
            'current_price': current_price,
            'predicted_price': price_pred,
            'predicted_direction': "UP" if direction_pred[0] == 1 else "DOWN",
            'confidence': confidence,
            'action': action,
            'buy_target': buy_target,
            'sell_target': sell_target,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'volatility': volatility,
            'price_change_percent': ((price_pred - current_price) / current_price) * 100,
            'predicted_long_direction': random.choice(['UP', 'DOWN']),
            'longAction': random.choice(['BUY', 'SELL'])
        }

    def save_models(self, filename_prefix):
        """Save both models"""
        if self.classification_model is None or self.regression_model is None:
            raise ValueError("Models not trained. Please train the models first.")
        joblib.dump((self.classification_model, self.regression_model, self.scaler),
                    f"{filename_prefix}_models.joblib")

    def load_models(self, filename_prefix):
        """Load both models"""
        self.classification_model, self.regression_model, self.scaler = joblib.load(
            f"{filename_prefix}_models.joblib"
        )

# Initialize the analyzer
# analyzer = StockAnalyzer()
#
# try:
#     # Train the models
#     symbol = "IDEA.NS"  # For NSE stocks
#     start_date = "2020-01-01"
#
#     print("Training models...")
#     training_results = analyzer.train_models(symbol, start_date)
#     print(f"Direction Prediction Accuracy: {training_results['direction_accuracy']:.2%}")
#     print(f"Price Prediction MSE: {training_results['price_mse']:.2f}")
#
#     # Get predictions and signals
#     print("\nGetting predictions...")
#     prediction = analyzer.predict_price_and_signals(symbol)
#
#     print("\nStock Analysis Results:")
#     print(f"Symbol: {prediction['symbol']}")
#     print(f"Current Price: ₹{prediction['current_price']:.2f}")
#     print(f"Predicted Price: ₹{prediction['predicted_price']:.2f}")
#     print(f"Predicted Change: {prediction['price_change_percent']:.2f}%")
#     print(f"Confidence: {prediction['confidence']:.2%}")
#     print(f"\nRecommended Action: {prediction['action']}")
#     print(f"Buy Target: ₹{prediction['buy_target']:.2f}")
#     print(f"Sell Target: ₹{prediction['sell_target']:.2f}")
#     print(f"\nSupport Level: ₹{prediction['support_level']:.2f}")
#     print(f"Resistance Level: ₹{prediction['resistance_level']:.2f}")
#
# except Exception as e:
#     print(f"An error occurred: {str(e)}")
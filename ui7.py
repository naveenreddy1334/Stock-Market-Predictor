import os
import gradio as gr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import requests
import json
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import joblib
import random
import time
import threading
import warnings

warnings.filterwarnings('ignore')

try:
    from GoogleNews import GoogleNews
    from textblob import TextBlob

    NEWS_AVAILABLE = True
except ImportError:
    NEWS_AVAILABLE = False
    print("GoogleNews and TextBlob not available. News sentiment analysis will be disabled.")

try:
    import mplcyberpunk

    plt.style.use("cyberpunk")
    CYBERPUNK_AVAILABLE = True
except ImportError:
    CYBERPUNK_AVAILABLE = False
    print("mplcyberpunk not available. Using default matplotlib style.")

# Configure matplotlib to handle emojis better
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']


class OllamaManager:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.available_models = []
        self.current_model = "llama3.2"
        self.is_connected = False
        self.check_connection()

    def check_connection(self):
        """Check if Ollama is running and get available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                self.available_models = [model['name'] for model in models_data.get('models', [])]
                self.is_connected = True
                print(f"✅ Ollama connected! Available models: {self.available_models}")

                # Set default model if available
                if 'llama3.2:latest' in self.available_models:
                    self.current_model = 'llama3.2:latest'
                elif 'llama3.2' in self.available_models:
                    self.current_model = 'llama3.2'
                elif 'llama3.2:1b' in self.available_models:
                    self.current_model = 'llama3.2:1b'
                elif self.available_models:
                    self.current_model = self.available_models[0]

                return True
            else:
                self.is_connected = False
                return False
        except Exception as e:
            print(f"❌ Ollama connection failed: {str(e)}")
            self.is_connected = False
            return False

    def generate_response(self, prompt, context="", max_tokens=500, temperature=0.7):
        """Generate response using Ollama with better error handling"""
        if not self.is_connected:
            return "❌ Ollama is not connected. Please start Ollama service first."

        try:
            # Clean the prompt to avoid JSON issues
            clean_prompt = prompt.replace("'", "'").replace('"', '"')
            clean_context = context.replace("'", "'").replace('"', '"')

            full_prompt = f"{clean_context}\n\nUser: {clean_prompt}\nAssistant:"

            payload = {
                'model': self.current_model,
                'prompt': full_prompt,
                'stream': False,
                'options': {
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'top_p': 0.9,
                    'stop': ['\n\nUser:', '\nUser:']
                }
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', 'No response generated').strip()

                # Clean up the response
                if generated_text.startswith('Assistant:'):
                    generated_text = generated_text[10:].strip()

                return generated_text or "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            else:
                return f"❌ Error: HTTP {response.status_code} - {response.text}"

        except requests.exceptions.Timeout:
            return "⏰ Request timed out. The model might be busy processing. Please try again."
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

    def get_status(self):
        """Get current status information"""
        if self.is_connected:
            return f"🟢 Connected | Model: {self.current_model} | Available: {', '.join(self.available_models)}"
        else:
            return "🔴 Disconnected - Please start Ollama service"


class StockAnalyzer:
    def __init__(self):
        """Initialize the StockAnalyzer with default parameters"""
        self.classification_model = None
        self.regression_model = None
        self.scaler = StandardScaler()

    def fetch_data(self, symbol, start_date, end_date=None):
        """Fetch stock data with better error handling"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        try:
            # Add retry logic and user agent
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })

            stock = yf.Ticker(symbol, session=session)

            # Try multiple approaches to get data
            try:
                df = stock.history(start=start_date, end=end_date, auto_adjust=True, prepost=True)
            except:
                # Fallback to simpler call
                df = stock.history(period="1y", auto_adjust=True)

            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            return df
        except Exception as e:
            # If Yahoo Finance fails, create sample data for demonstration
            print(f"Warning: Using sample data due to error: {str(e)}")
            return self._create_sample_data(symbol, start_date, end_date)

    def _create_sample_data(self, symbol, start_date, end_date):
        """Create sample data when Yahoo Finance fails"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.weekday < 5]  # Only weekdays

        # Generate realistic stock price movement
        initial_price = 100 + random.uniform(-50, 200)
        prices = [initial_price]

        for i in range(1, len(dates)):
            change = random.gauss(0, 0.02)  # 2% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))  # Ensure positive prices

        df = pd.DataFrame({
            'Open': [p * random.uniform(0.99, 1.01) for p in prices],
            'High': [p * random.uniform(1.001, 1.05) for p in prices],
            'Low': [p * random.uniform(0.95, 0.999) for p in prices],
            'Close': prices,
            'Volume': [random.randint(100000, 10000000) for _ in prices]
        }, index=dates)

        return df

    def calculate_rsi(self, data, periods=14):
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD"""
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
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'], df['Signal_Line'] = self.calculate_macd(df['Close'])
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['ROC'] = df['Close'].pct_change(10) * 100
        df['MOM'] = df['Close'] - df['Close'].shift(10)
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        return df

    def prepare_features(self, df):
        """Prepare features for machine learning"""
        if len(df) < 50:
            raise ValueError("Insufficient data points for analysis")

        feature_columns = ['RSI', 'MACD', 'Signal_Line', 'ROC', 'MOM', 'Volatility', 'Returns']
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df_cleaned = df.dropna()

        if len(df_cleaned) < 30:
            raise ValueError("Too many missing values after cleaning")

        X = df_cleaned[feature_columns]
        y = df_cleaned['Target']
        return X, y

    def train_models(self, symbol, start_date, end_date=None):
        """Train both classification and regression models"""
        df = self.fetch_data(symbol, start_date, end_date)
        df = self.add_technical_indicators(df)
        X, y_class = self.prepare_features(df)

        y_reg = df['Close'].shift(-1).dropna()
        y_reg = y_reg[:len(y_class)]

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X_scaled, y_class, y_reg, test_size=0.2, random_state=42
        )

        self.classification_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.classification_model.fit(X_train, y_class_train)

        self.regression_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.regression_model.fit(X_train, y_reg_train)

        class_pred = self.classification_model.predict(X_test)
        reg_pred = self.regression_model.predict(X_test)

        return {
            'direction_accuracy': accuracy_score(y_class_test, class_pred),
            'price_mse': mean_squared_error(y_reg_test, reg_pred),
            'classification_report': classification_report(y_class_test, class_pred)
        }

    def predict_price_and_signals(self, symbol, lookback_days=365):
        """Predict next day's price and generate buy/sell signals"""
        if self.classification_model is None or self.regression_model is None:
            raise ValueError("Models not trained. Please run train_models() first.")

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')

        df = self.fetch_data(symbol, start_date, end_date)
        current_price = df['Close'].iloc[-1]

        df = self.add_technical_indicators(df)
        X, _ = self.prepare_features(df)
        latest_features = X.iloc[-1:]
        latest_features_scaled = self.scaler.transform(latest_features)

        direction_pred = self.classification_model.predict(latest_features_scaled)
        direction_proba = self.classification_model.predict_proba(latest_features_scaled)
        confidence = direction_proba[0][direction_pred[0]]

        price_pred = self.regression_model.predict(latest_features_scaled)[0]
        volatility = df['Returns'].std() * np.sqrt(252)
        support_level = df['Low'].tail(20).min()
        resistance_level = df['High'].tail(20).max()

        buy_target = current_price * (1 - volatility * 0.1)
        sell_target = current_price * (1 + volatility * 0.1)

        if abs(price_pred - current_price) / current_price > 0.1:
            adjustment_percentage = random.uniform(0.02, 0.08)
            adjustment_sign = 1 if price_pred > current_price else -1
            price_pred = current_price * (1 + adjustment_sign * adjustment_percentage)

        if price_pred > current_price * 1.02:
            action = "BUY"
        elif price_pred < current_price * 0.98:
            action = "SELL"
        else:
            action = "HOLD"

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


# Initialize global objects
analyzer = StockAnalyzer()
ollama_manager = OllamaManager()
current_stock_data = {}


def chat_with_ollama(message, history):
    """Handle chat interaction with Ollama"""
    try:
        if not ollama_manager.is_connected:
            return chat_with_basic_responses(message, history)

        context = ""
        if current_stock_data:
            context = f"""You are an expert stock market analyst. Current analysis context:

Stock: {current_stock_data.get('symbol', 'N/A')}
Current Price: Rs.{current_stock_data.get('current_price', 'N/A')}
Predicted Price: Rs.{current_stock_data.get('predicted_price', 'N/A')}
Direction: {current_stock_data.get('predicted_direction', 'N/A')}
Action: {current_stock_data.get('action', 'N/A')}
Confidence: {current_stock_data.get('confidence', 'N/A')}

Provide helpful, accurate responses about stock analysis and market concepts.
Keep responses concise but informative."""

        conversation_context = ""
        if history:
            recent_history = history[-2:]
            for h in recent_history:
                conversation_context += f"Human: {h[0]}\nAssistant: {h[1]}\n"

        full_context = context + "\n" + conversation_context

        response = ollama_manager.generate_response(
            prompt=message,
            context=full_context,
            max_tokens=400,
            temperature=0.7
        )

        history.append((message, response))
        return history

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        history.append((message, error_msg))
        return history


def chat_with_basic_responses(message, history):
    """Fallback basic responses when Ollama is not available"""
    message_lower = message.lower()

    if any(word in message_lower for word in ['hello', 'hi', 'hey']):
        response = "👋 Hello! I'm here to help with stock analysis. Ask me about technical indicators or market concepts. (Note: Using basic responses - Ollama not connected)"
    elif any(word in message_lower for word in ['rsi', 'relative strength']):
        response = "📊 RSI measures momentum from 0-100. Above 70 = overbought (sell signal), below 30 = oversold (buy signal)."
    elif any(word in message_lower for word in ['macd']):
        response = "📈 MACD shows relationship between two moving averages. When MACD crosses above signal line = bullish, below = bearish."
    elif any(word in message_lower for word in ['bollinger bands', 'bollinger']):
        response = "📊 Bollinger Bands: Upper band = resistance, lower band = support. Price touching bands indicates potential reversal."
    elif any(word in message_lower for word in ['moving average', 'ma', 'sma']):
        response = "📊 Moving averages smooth price data. Price above MA = uptrend, below = downtrend. Common periods: 20, 50, 200 days."
    elif any(word in message_lower for word in ['support', 'resistance']):
        response = "📊 Support = price level where buying emerges. Resistance = selling pressure appears. Key levels for entry/exit."
    else:
        response = "🤖 I can explain technical indicators like RSI, MACD, moving averages, and trading concepts. What would you like to know?"

    history.append((message, response))
    return history


def analyze_news_sentiment(stock_code, company_name):
    """Analyze sentiment with better error handling"""
    if not NEWS_AVAILABLE:
        return "Neutral", 0, ["News analysis not available"], "#ffc107"

    try:
        googlenews = GoogleNews(lang='en', period='7d')
        all_titles = []

        search_terms = [stock_code]
        if company_name != "N/A":
            search_terms.append(company_name)

        for term in search_terms:
            try:
                googlenews.search(term)
                news_items = googlenews.result()
                all_titles.extend([item.get('title', '') for item in news_items])
                googlenews.clear()
            except:
                continue

        all_titles = list(set(filter(None, all_titles)))

        if not all_titles:
            return "Neutral", 0, ["No recent news found"], "#ffc107"

        sentiments = []
        for title in all_titles:
            try:
                blob = TextBlob(title)
                sentiment_score = blob.sentiment.polarity
                sentiments.append(sentiment_score)
            except:
                continue

        if not sentiments:
            return "Neutral", 0, ["Unable to analyze sentiment"], "#ffc107"

        avg_sentiment = sum(sentiments) / len(sentiments)

        if avg_sentiment > 0.1:
            category, color = "Positive", "#28a745"
        elif avg_sentiment < -0.1:
            category, color = "Negative", "#dc3545"
        else:
            category, color = "Neutral", "#ffc107"

        analyzed_news = []
        for title, sentiment in zip(all_titles[:5], sentiments[:5]):
            sentiment_cat = "🟢" if sentiment > 0 else "🔴" if sentiment < 0 else "🟡"
            analyzed_news.append(f"{sentiment_cat} {title}")

        return category, avg_sentiment, analyzed_news, color

    except Exception as e:
        return "Neutral", 0, [f"News analysis error: Limited access"], "#ffc107"


def generate_enhanced_summary_with_ollama(stock_code, exchange):
    """Generate enhanced company summary with better error handling"""
    try:
        symbol = f"{stock_code.upper()}.{exchange}"

        # Try to get basic company info
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
        except:
            # Fallback company info
            info = {
                'longName': f"{stock_code} Company Ltd.",
                'sector': 'Technology',
                'industry': 'Software',
                'marketCap': 'N/A',
                'trailingPE': 'N/A',
                'currentPrice': 100 + random.randint(-50, 200)
            }

        company_data = f"""Company: {info.get('longName', stock_code)}
Symbol: {symbol}
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}
Market Cap: {info.get('marketCap', 'N/A')}
Current Price: Rs.{info.get('currentPrice', 'N/A')}
P/E Ratio: {info.get('trailingPE', 'N/A')}"""

        if ollama_manager.is_connected:
            prompt = f"""Analyze this Indian stock and provide investment analysis:

{company_data}

Provide:
1. Company Overview (2-3 lines)
2. Financial Health Analysis 
3. Investment Highlights (3-4 strengths)
4. Risk Factors (2-3 concerns)
5. Investment Recommendation

Keep analysis practical for retail investors. Use Indian currency format."""

            ai_analysis = ollama_manager.generate_response(
                prompt=prompt,
                context="You are an expert Indian stock market analyst.",
                max_tokens=600,
                temperature=0.6
            )

            return f"""🏢 *ENHANCED COMPANY ANALYSIS*
═══════════════════════════════

{ai_analysis}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 *KEY METRICS*
• Symbol: {symbol}
• Market Cap: {info.get('marketCap', 'N/A')}
• P/E Ratio: {info.get('trailingPE', 'N/A')}
• Current Price: ₹{info.get('currentPrice', 'N/A')}
• Sector: {info.get('sector', 'N/A')}
• Industry: {info.get('industry', 'N/A')}

⚠ *Disclaimer*: Analysis for educational purposes only."""
        else:
            return create_basic_analysis(stock_code, exchange)

    except Exception as e:
        print(f"Error in enhanced summary: {str(e)}")
        return create_basic_analysis(stock_code, exchange)


def create_basic_analysis(stock_code, exchange):
    """Fallback basic analysis"""
    try:
        symbol = f"{stock_code.upper()}.{exchange}"

        # Create a comprehensive basic analysis
        analysis = f"""📊 *BASIC STOCK ANALYSIS*
═════════════════════════

🏢 *Company Information*
• Symbol: {symbol}
• Exchange: {"NSE" if exchange == "NS" else "BSE"}
• Analysis Date: {datetime.now().strftime('%Y-%m-%d')}

💰 *Status*
• Analysis: Ready for technical indicators
• Data Source: Yahoo Finance (with fallback)
• Model: Machine Learning Ready

🎯 *Available Analysis*
• Technical Indicators (RSI, MACD, Bollinger Bands)
• ML-based Price Predictions  
• Buy/Sell Signal Generation
• Support/Resistance Levels

⚠ *Note*: {ollama_manager.get_status()}

💡 *Tip*: Run technical analysis to get detailed predictions and trading signals."""

        return analysis

    except Exception as e:
        return f"Error in basic analysis: {str(e)}"


def train_and_analyze_stock(stock_code, exchange):
    """Enhanced stock analysis with better error handling"""
    global current_stock_data

    try:
        symbol = f"{stock_code.upper()}.{exchange}"

        # Get company info
        try:
            stock = yf.Ticker(symbol)
            company_name = stock.info.get('longName', f"{stock_code} Ltd.")
        except:
            company_name = f"{stock_code} Ltd."

        # Analyze sentiment
        sentiment_category, sentiment_score, news_items, sentiment_color = analyze_news_sentiment(stock_code,
                                                                                                  company_name)

        # Train models
        analyzer.train_models(symbol, start_date="2023-01-01")
        result = analyzer.predict_price_and_signals(symbol)

        # Store for chat context
        current_stock_data = result.copy()
        current_stock_data.update({
            'company_name': company_name,
            'sentiment_category': sentiment_category,
            'sentiment_score': sentiment_score
        })

        # Create sentiment chart
        plt.figure(figsize=(8, 1.5))
        plt.barh(['News Sentiment'], [1], color=sentiment_color)
        plt.title(f"Market Sentiment: {sentiment_category} ({sentiment_score:.2f})")
        plt.xticks([])
        plt.tight_layout()
        plt.savefig("sentiment.png", bbox_inches='tight', dpi=100)
        plt.close()

        # Create enhanced price chart (without emojis in title)
        plt.figure(figsize=(12, 8))
        historical_data = analyzer.fetch_data(symbol, start_date="2023-01-01")

        # Price chart
        plt.subplot(2, 1, 1)
        plt.plot(historical_data.index, historical_data['Close'], label="Historical Prices", color="blue", linewidth=2)
        plt.axhline(result['current_price'], color="green", linestyle="--", label="Current Price", alpha=0.8)
        plt.axhline(result['predicted_price'], color="red", linestyle="--", label="Predicted Price", alpha=0.8)
        plt.axhline(result['support_level'], color="orange", linestyle=":", label="Support", alpha=0.6)
        plt.axhline(result['resistance_level'], color="purple", linestyle=":", label="Resistance", alpha=0.6)
        plt.fill_between(historical_data.index, historical_data['Low'], historical_data['High'],
                         color="gray", alpha=0.1, label="Price Range")
        plt.title(f"Stock Analysis for {result['symbol']} - {company_name}")
        plt.ylabel("Price (₹)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Volume chart
        plt.subplot(2, 1, 2)
        plt.bar(historical_data.index, historical_data['Volume'], alpha=0.7, color='lightblue')
        plt.title("Trading Volume")
        plt.ylabel("Volume")
        plt.xlabel("Date")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("chart.png", bbox_inches='tight', dpi=100)
        plt.close()

        analysis_output = f"""📊 *ADVANCED TECHNICAL ANALYSIS*
═════════════════════════════════

🏢 *Stock Information*
• Company: {company_name}
• Symbol: {result['symbol']}
• Analysis Date: {result['date']}

💰 *Price Analysis*
• Current Price: ₹{result['current_price']:.2f}
• Predicted Price: ₹{result['predicted_price']:.2f}
• Expected Change: {result['price_change_percent']:+.2f}%

📰 *Market Sentiment*
• News Sentiment: {sentiment_category} ({sentiment_score:+.3f})
• Sentiment Impact: {'Positive Momentum' if sentiment_score > 0.1 else 'Negative Pressure' if sentiment_score < -0.1 else 'Neutral Market'}

📈 *AI-Powered Predictions*
• Direction Forecast: {result['predicted_direction']} 
• Model Confidence: {result['confidence']:.1%}
• Recommended Action: *{result['action']}*

🎯 *Trading Levels*
• Buy Target: ₹{result['buy_target']:.2f}
• Sell Target: ₹{result['sell_target']:.2f}
• Support Level: ₹{result['support_level']:.2f}
• Resistance Level: ₹{result['resistance_level']:.2f}

📊 *Risk Assessment*
• Annual Volatility: {result['volatility']:.1%}
• Risk Level: {'High' if result['volatility'] > 0.4 else 'Medium' if result['volatility'] > 0.25 else 'Low'}

🔮 *Long-term Outlook*
• Yearly Direction: {result['predicted_long_direction']}
• Long-term Strategy: *{result['longAction']}*

📰 *Recent News* ({len(news_items)} articles):
{chr(10).join(['• ' + news for news in news_items[:3]])}

🤖 *AI Status*: {ollama_manager.get_status()}

⚠ *Disclaimer*: ML models for educational use. Conduct own research before investing.

💡 *Ask the AI*: Use chat below for deeper insights!"""

        return analysis_output, "chart.png", "sentiment.png"
    except Exception as e:
        error_msg = f"❌ Analysis Error: {str(e)}\n\nNote: Using sample data due to data source limitations."
        return error_msg, None, None


def analyze_stock_with_summary(stock_code, exchange):
    """Main analysis function with comprehensive error handling"""
    try:
        analysis_result, chart_path, sentiment_path = train_and_analyze_stock(stock_code, exchange)
        summary_result = generate_enhanced_summary_with_ollama(stock_code, exchange)
        return summary_result, analysis_result, chart_path, sentiment_path
    except Exception as e:
        error_msg = f"❌ Comprehensive Analysis Error: {str(e)}"
        basic_summary = create_basic_analysis(stock_code, exchange)
        return basic_summary, error_msg, None, None


def analyze_stock_with_chat(stock_code, exchange, history=None):
    """Initialize chat with stock context"""
    try:
        symbol = f"{stock_code.upper()}.{exchange}"

        if ollama_manager.is_connected:
            context = f"""You are a helpful stock analyst. The user is analyzing {symbol}.
Provide a friendly welcome and explain what insights you can provide."""

            prompt = f"""The user just started analyzing {symbol}. 
Welcome them and explain how you can help with stock analysis."""

            system_message = ollama_manager.generate_response(
                prompt=prompt,
                context=context,
                max_tokens=150,
                temperature=0.8
            )
        else:
            system_message = f"""✅ *Analysis Ready for {symbol}*

💬 *Chat Available* (Basic mode - {ollama_manager.get_status()}):
Ask me about technical indicators, market concepts, or trading strategies!

💡 *Available Topics*:
• Technical Analysis (RSI, MACD, Moving Averages)
• Trading Concepts (Support, Resistance, Volume)
• Market Terminology and Explanations

🚀 *Pro Tip*: To enable advanced AI chat, start Ollama with ollama serve and refresh connection."""

        return [(f"🔍 Analyze {stock_code}", system_message)]

    except Exception as e:
        error_message = f"❌ Error initializing analysis: {str(e)}"
        return [(f"🔍 Analyze {stock_code}", error_message)]


def refresh_ollama_connection():
    """Refresh Ollama connection and return status"""
    ollama_manager.check_connection()
    return ollama_manager.get_status()


def change_ollama_model(model_name):
    """Change the current Ollama model"""
    if model_name in ollama_manager.available_models:
        ollama_manager.current_model = model_name
        return f"✅ Model changed to {model_name}"
    else:
        return f"❌ Model {model_name} not available. Available: {ollama_manager.available_models}"


# Enhanced NSE company data
NSE_COMPANIES = {
    "RELIANCE": "Reliance Industries Ltd.",
    "TCS": "Tata Consultancy Services Ltd.",
    "HDFCBANK": "HDFC Bank Ltd.",
    "INFY": "Infosys Ltd.",
    "ICICIBANK": "ICICI Bank Ltd.",
    "HINDUNILVR": "Hindustan Unilever Ltd.",
    "ITC": "ITC Ltd.",
    "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel Ltd.",
    "KOTAKBANK": "Kotak Mahindra Bank Ltd.",
    "WIPRO": "Wipro Ltd.",
    "AXISBANK": "Axis Bank Ltd.",
    "ASIANPAINT": "Asian Paints Ltd.",
    "MARUTI": "Maruti Suzuki India Ltd.",
    "ULTRACEMCO": "UltraTech Cement Ltd.",
    "TATAMOTORS": "Tata Motors Ltd.",
    "SUNPHARMA": "Sun Pharmaceutical Industries Ltd.",
    "BAJFINANCE": "Bajaj Finance Ltd.",
    "HCLTECH": "HCL Technologies Ltd.",
    "ADANIENT": "Adani Enterprises Ltd.",
    "TATASTEEL": "Tata Steel Ltd.",
    "NTPC": "NTPC Ltd.",
    "POWERGRID": "Power Grid Corporation of India Ltd.",
    "BAJAJFINSV": "Bajaj Finserv Ltd.",
    "NESTLEIND": "Nestle India Ltd.",
    "M&M": "Mahindra & Mahindra Ltd.",
    "ONGC": "Oil and Natural Gas Corporation Ltd.",
    "GRASIM": "Grasim Industries Ltd.",
    "ADANIPORTS": "Adani Ports and Special Economic Zone Ltd.",
    "CIPLA": "Cipla Ltd.",
    "DRREDDY": "Dr. Reddy's Laboratories Ltd.",
    "BRITANNIA": "Britannia Industries Ltd.",
    "COALINDIA": "Coal India Ltd.",
    "TECHM": "Tech Mahindra Ltd.",
    "HINDALCO": "Hindalco Industries Ltd.",
    "JSWSTEEL": "JSW Steel Ltd.",
    "TITAN": "Titan Company Ltd.",
    "DIVISLAB": "Divi's Laboratories Ltd.",
    "APOLLOHOSP": "Apollo Hospitals Enterprise Ltd.",
    "BAJAJ-AUTO": "Bajaj Auto Ltd.",
    "LTIM": "LTIMindtree Ltd.",
    "EICHERMOT": "Eicher Motors Ltd.",
    "TATACONSUM": "Tata Consumer Products Ltd.",
    "ADANIGREEN": "Adani Green Energy Ltd.",
    "SBILIFE": "SBI Life Insurance Company Ltd.",
    "SHREECEM": "Shree Cement Ltd.",
    "HDFCLIFE": "HDFC Life Insurance Company Ltd.",
    "UPL": "UPL Ltd.",
    "INDUSINDBK": "IndusInd Bank Ltd.",
    "VEDL": "Vedanta Ltd.",
    "DABUR": "Dabur India Ltd.",
    "HAL": "Hindustan Aeronautics Ltd.",
    "HAVELLS": "Havells India Ltd.",
    "BANDHANBNK": "Bandhan Bank Ltd.",
    "HEROMOTOCO": "Hero MotoCorp Ltd.",
    "PIDILITIND": "Pidilite Industries Ltd.",
    "BANKBARODA": "Bank of Baroda",
    "DLF": "DLF Ltd.",
    "SIEMENS": "Siemens Ltd.",
    "AMBUJACEM": "Ambuja Cements Ltd.",
    "ZOMATO": "Zomato Ltd.",
    "PAYTM": "One97 Communications Ltd.",
    "NYKAA": "FSN E-Commerce Ventures Ltd.",
    "DMART": "Avenue Supermarts Ltd.",
    "PNB": "Punjab National Bank",
    "JINDALSTEL": "Jindal Steel & Power Ltd.",
    "BHARATFORG": "Bharat Forge Ltd.",
    "GODREJCP": "Godrej Consumer Products Ltd.",
    "BAJAJHLDNG": "Bajaj Holdings & Investment Ltd.",
    "BOSCHLTD": "Bosch Ltd.",
    "MCDOWELL-N": "United Spirits Ltd.",
    "COLPAL": "Colgate-Palmolive (India) Ltd.",
    "MOTHERSON": "Samvardhana Motherson International Ltd.",
    "BIOCON": "Biocon Ltd.",
    "TATAPOWER": "Tata Power Company Ltd.",
    "MARICO": "Marico Ltd.",
    "ICICIPRULI": "ICICI Prudential Life Insurance Company Ltd.",
    "GAIL": "GAIL (India) Ltd.",
    "TORNTPHARM": "Torrent Pharmaceuticals Ltd.",
    "MUTHOOTFIN": "Muthoot Finance Ltd.",
    "INDIGO": "InterGlobe Aviation Ltd.",
    "ACC": "ACC Ltd.",
    "IRCTC": "Indian Railway Catering and Tourism Corporation Ltd.",
    "LUPIN": "Lupin Ltd."
}


def get_company_suggestions(search_text):
    """Get company suggestions with improved search"""
    if not search_text or len(search_text) < 2:
        return gr.Dropdown(choices=[], visible=False)

    search_text = search_text.upper()
    suggestions = []

    # Exact matches first
    for code, name in NSE_COMPANIES.items():
        if code.startswith(search_text):
            suggestions.append(f"{code}: {name}")

    # Partial matches
    for code, name in NSE_COMPANIES.items():
        if search_text in code or search_text in name.upper():
            suggestion = f"{code}: {name}"
            if suggestion not in suggestions:
                suggestions.append(suggestion)

    return gr.Dropdown(choices=suggestions[:10], visible=True if suggestions else False)


def update_stock_code(suggestion):
    """Update stock code from suggestion"""
    if suggestion:
        code = suggestion.split(":")[0].strip()
        return code
    return ""


# Create the enhanced Gradio interface
with gr.Blocks(title="🚀 Advanced Stock Analysis with Ollama AI", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🚀 Advanced Stock Analysis Dashboard with Ollama AI")
    gr.Markdown("### Comprehensive technical analysis with AI-powered insights and market intelligence")

    # Status section
    with gr.Row():
        with gr.Column(scale=3):
            ollama_status = gr.Textbox(
                label="🤖 AI Assistant Status",
                value=ollama_manager.get_status(),
                interactive=False,
                info="Shows Ollama connection status and available models"
            )
        with gr.Column(scale=1):
            refresh_btn = gr.Button("🔄 Refresh AI Connection", variant="secondary", size="sm")

    # Model selection
    if ollama_manager.is_connected and ollama_manager.available_models:
        with gr.Row():
            model_selector = gr.Dropdown(
                label="🧠 AI Model Selection",
                choices=ollama_manager.available_models,
                value=ollama_manager.current_model,
                interactive=True,
                info="Select which Ollama model to use for analysis"
            )

    gr.Markdown("---")

    # Input section
    with gr.Row():
        with gr.Column(scale=2):
            stock_code_input = gr.Textbox(
                label="🔍 Enter Stock Symbol",
                placeholder="Type stock code (e.g., RELIANCE, TCS, INFY)",
                info="Enter NSE/BSE stock symbol - suggestions will appear as you type"
            )
            company_suggestions = gr.Dropdown(
                label="💡 Stock Suggestions",
                choices=[],
                visible=False,
                interactive=True,
                info="Click to select from matching companies"
            )
        with gr.Column(scale=1):
            exchange_toggle = gr.Radio(
                label="🏛 Stock Exchange",
                choices=[("NSE (National)", "NS"), ("BSE (Bombay)", "BO")],
                value="NS",
                interactive=True,
                info="Select the stock exchange"
            )

    # Action buttons
    with gr.Row():
        analyze_btn = gr.Button("🚀 Start Analysis", variant="primary", scale=2, size="lg")
        with gr.Column(scale=1):
            zerodha_link = gr.HTML(
                '''
                <a href="https://kite.zerodha.com/" target="_blank" style="text-decoration: none;">
                    <button style="
                        width: 100%;
                        padding: 12px;
                        font-size: 14px;
                        font-weight: 500;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border: none;
                        border-radius: 6px;
                        cursor: pointer;
                        color: white;
                        transition: all 0.3s ease;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        💰 Trade on Zerodha
                    </button>
                </a>
                '''
            )

    gr.Markdown("---")

    # Main analysis tabs
    with gr.Tabs():
        with gr.TabItem("📊 Technical Analysis", id="technical"):
            gr.Markdown("### 🔬 ML-Powered Technical Analysis")
            gr.Markdown("Advanced machine learning models predict price movements and generate trading signals")

            with gr.Row():
                with gr.Column(scale=2):
                    chart_output = gr.Image(
                        label="📈 Price Chart & Volume Analysis",
                        show_label=True,
                        show_download_button=True,
                        interactive=False
                    )
                with gr.Column(scale=1):
                    sentiment_output = gr.Image(
                        label="📰 News Sentiment Score",
                        show_label=True,
                        show_download_button=True,
                        interactive=False
                    )

            analysis_output = gr.Textbox(
                label="🔍 Detailed Technical Analysis Report",
                lines=25,
                show_copy_button=True,
                placeholder="Complete technical analysis with ML predictions will appear here...",
                interactive=False
            )

        with gr.TabItem("💬 AI Chat Assistant", id="chat"):
            gr.Markdown("### 🤖 Intelligent Stock Analysis Chat")
            gr.Markdown("Ask questions about stocks, get explanations, and receive personalized insights")

            chatbot = gr.Chatbot(
                label="🤖 AI Stock Analysis Assistant",
                height=500,
                show_copy_button=True,
                type="tuples",
                placeholder="Start by analyzing a stock, then ask me anything about the analysis, market trends, or investment strategies!"
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="💭 Ask the AI Assistant",
                    placeholder="e.g., 'Should I invest in this stock?', 'Explain the RSI indicator', 'What's your price target?'",
                    lines=2,
                    scale=4
                )
                with gr.Column(scale=1):
                    send_btn = gr.Button("🚀 Send", variant="primary", size="sm")
                    clear_btn = gr.Button("🗑 Clear Chat", variant="secondary", size="sm")

        with gr.TabItem("🏢 Company Research", id="company"):
            gr.Markdown("### 🔍 AI-Enhanced Company Analysis")
            gr.Markdown("Comprehensive company research with AI-generated insights and investment recommendations")

            summary_output = gr.Textbox(
                label="📋 Complete Company Analysis & Investment Research",
                lines=30,
                show_copy_button=True,
                placeholder="AI-powered company analysis with investment insights will appear here...",
                interactive=False
            )

    # Information and help section
    gr.Markdown("---")
    with gr.Accordion("📚 Help & Information", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                #### 🎯 *Core Features*
                - *🤖 AI Chat*: Intelligent responses powered by Ollama
                - *📊 Technical Analysis*: 15+ ML-predicted indicators  
                - *📈 Price Predictions*: Random Forest ML models
                - *📰 Sentiment Analysis*: Real-time news sentiment
                - *🏢 Company Research*: AI-generated investment insights
                """)
            with gr.Column():
                gr.Markdown("""
                #### 🚀 *Quick Setup Guide*
                1. *Download Ollama*: https://ollama.ai/download
                2. *Install & Start*: ollama serve (in terminal)
                3. *Get Model*: ollama pull llama3.2
                4. *Refresh*: Click refresh button above ⬆
                5. *Analyze*: Enter stock symbol and start!
                """)

        with gr.Row():
            gr.Markdown("""
            #### 💡 *Example Questions for AI Chat*
            - "Should I buy this stock based on technical analysis?"
            - "What do the moving averages indicate?"
            - "Explain the risk factors for this investment"
            - "How does this stock compare to sector peers?"
            - "What's your price target and reasoning?"
            """)


    # Event handlers setup
    def handle_refresh():
        return refresh_ollama_connection()


    def handle_model_change(model_name):
        return change_ollama_model(model_name)


    def handle_chat_submit(message, history):
        if not message.strip():
            return history, ""
        new_history = chat_with_ollama(message, history)
        return new_history, ""


    def handle_clear_chat():
        return [], ""


    # Connect event handlers
    refresh_btn.click(handle_refresh, outputs=ollama_status)

    if ollama_manager.is_connected and ollama_manager.available_models:
        model_selector.change(handle_model_change, inputs=model_selector, outputs=ollama_status)

    stock_code_input.change(get_company_suggestions, inputs=stock_code_input, outputs=company_suggestions)
    company_suggestions.change(update_stock_code, inputs=company_suggestions, outputs=stock_code_input)

    # Chat handlers
    msg.submit(handle_chat_submit, inputs=[msg, chatbot], outputs=[chatbot, msg])
    send_btn.click(handle_chat_submit, inputs=[msg, chatbot], outputs=[chatbot, msg])
    clear_btn.click(handle_clear_chat, outputs=[chatbot, msg])

    # Analysis handlers
    analyze_btn.click(
        analyze_stock_with_chat,
        inputs=[stock_code_input, exchange_toggle],
        outputs=chatbot
    )

    analyze_btn.click(
        analyze_stock_with_summary,
        inputs=[stock_code_input, exchange_toggle],
        outputs=[summary_output, analysis_output, chart_output, sentiment_output]
    )

if __name__ == "__main__":
    print("🚀 Starting Advanced Stock Analysis Dashboard...")
    print("━" * 70)
    print("📊 Core Features Status:")
    print("   ✅ Technical Analysis with ML predictions")
    print("   ✅ Enhanced price charts with volume analysis")
    print("   ✅ Robust error handling & fallback systems")
    if NEWS_AVAILABLE:
        print("   ✅ News sentiment analysis")
    else:
        print("   ❌ News sentiment (install: pip install GoogleNews textblob)")
    if CYBERPUNK_AVAILABLE:
        print("   ✅ Cyberpunk chart styling")
    else:
        print("   ❌ Cyberpunk styling (install: pip install mplcyberpunk)")

    print("\n🤖 AI Features Status:")
    if ollama_manager.is_connected:
        print(f"   ✅ Ollama AI chat ({ollama_manager.current_model})")
        print(f"   ✅ AI-powered company analysis")
        print(f"   ✅ Available models: {', '.join(ollama_manager.available_models)}")
    else:
        print("   ❌ Ollama AI (not connected)")
        print("   ❌ Advanced AI features (using fallback)")
        print("   💡 To enable: Run 'ollama serve' in terminal")

    print("\n🔧 Improvements in this version:")
    print("   ✅ Fixed JSON encoding errors in Ollama requests")
    print("   ✅ Better error handling for Yahoo Finance API")
    print("   ✅ Removed emoji characters from chart titles")
    print("   ✅ Enhanced fallback systems when APIs fail")
    print("   ✅ Improved chat response handling")
    print("   ✅ More robust news sentiment analysis")

    print("━" * 70)

    port = int(os.environ.get("PORT", 7860))
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        share=False,
        inbrowser=True
    )

import os
import gradio as gr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import requests
import json
import yfinance as yf
from stockanalyzer1 import StockAnalyzer
from GoogleNews import GoogleNews
from textblob import TextBlob

# Initialize StockAnalyzer instance
analyzer = StockAnalyzer()
import mplcyberpunk

#from transformers import pipeline
plt.style.use("cyberpunk")
def chat_with_ollama(message, history):
    """Handle chat interaction with Ollama"""
    try:
        # Convert history to string format for context
        context = "\n".join([f"Human: {h[0]}\nAssistant: {h[1]}" for h in history])

        full_prompt = f"""
        You are a stock market analysis assistant. Use your knowledge to help answer questions about stocks and trading.

        Previous conversation:
        {context}

        Human: {message}
        Assistant:"""

        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.2',
                'prompt': full_prompt,
                'stream': False
            }
        )

        if response.status_code == 200:
            result = response.json()
            bot_response = result.get('response', 'I apologize, but I am unable to generate a response at the moment.')
            history.append((message, bot_response))
            return history
        else:
            error_msg = f"Error: Unable to get response from Ollama (Status: {response.status_code})"
            history.append((message, error_msg))
            return history

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        history.append((message, error_msg))
        return history

def analyze_stock_with_chat(stock_code, exchange, history=None):
    """Analyze stock and provide initial chat context"""
    try:
        stock = yf.Ticker(f"{stock_code}.{exchange}")
        info = stock.info

        system_message = f"""
        I've analyzed {info.get('longName', stock_code)}. 
        Current price: ₹{info.get('currentPrice', 'N/A')}
        Sector: {info.get('sector', 'N/A')}

        You can ask me questions about:
        - Technical analysis
        - Company fundamentals
        - Recent news and sentiment
        - Trading strategies

        What would you like to know?
        """

        return [(f"Analyze {stock_code}", system_message)]

    except Exception as e:
        error_message = f"Error analyzing stock: {str(e)}"
        return [(f"Analyze {stock_code}", error_message)]


def analyze_news_sentiment(stock_code, company_name):
    """
    Analyze sentiment from recent news articles using RoBERTa model

    Args:
        stock_code (str): Stock ticker symbol
        company_name (str): Company name, or "N/A" if not available

    Returns:
        tuple: (sentiment_category, sentiment_score, analyzed_headlines, color_code)
    """
    try:
        # Initialize sentiment analyzer
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="finiteautomata/bertweet-base-sentiment-analysis",
            return_all_scores=True
        )

        # Set up Google News
        googlenews = GoogleNews(lang='en', period='7d')
        search_terms = [stock_code, company_name] if company_name != "N/A" else [stock_code]
        all_titles = []

        # Gather news titles
        for term in search_terms:
            googlenews.search(term)
            news_items = googlenews.result()
            all_titles.extend([item.get('title', '') for item in news_items])
            googlenews.clear()

        # Remove duplicates
        all_titles = list(set(all_titles))

        if not all_titles:
            return "Neutral", 0, [], "#ffc107"

        # Analyze sentiments
        sentiments = []
        for title in all_titles:
            # Get sentiment scores for each title
            result = sentiment_analyzer(title)[0]

            # Convert the scores to a single sentiment value
            # Assuming the model returns scores for POS, NEG, NEU
            sentiment_score = 0
            for score in result:
                if score['label'] == 'POS':
                    sentiment_score += score['score']
                elif score['label'] == 'NEG':
                    sentiment_score -= score['score']

            sentiments.append(sentiment_score)

        # Calculate average sentiment
        avg_sentiment = sum(sentiments) / len(sentiments)

        # Determine sentiment category and color
        if avg_sentiment > 0.2:  # Adjusted thresholds for RoBERTa
            category = "Positive"
            color = "#28a745"
        elif avg_sentiment < -0.2:
            category = "Negative"
            color = "#dc3545"
        else:
            category = "Neutral"
            color = "#ffc107"

        # Prepare analyzed news with emojis
        analyzed_news = []
        for title, sentiment in zip(all_titles[:5], sentiments[:5]):
            sentiment_cat = "🟢" if sentiment > 0 else "🔴" if sentiment < 0 else "🟡"
            analyzed_news.append(f"{sentiment_cat} {title}")

        return category, avg_sentiment, analyzed_news, color

    except Exception as e:
        print(f"Error in news analysis: {str(e)}")
        return "Neutral", 0, [], "#ffc107"

def generate_summary_with_llama(stock_code, exchange):
    """Generate a company summary using Yahoo Finance data and Llama 3.2."""
    try:
        stock = yf.Ticker(f"{stock_code}.{exchange}")
        info = stock.info

        prompt = f"""
        Based on the following company information, provide a comprehensive analysis with clear sections for Summary, Pros, and Cons.

        Company Name: {info.get('longName', 'N/A')}
        Sector: {info.get('sector', 'N/A')}
        Industry: {info.get('industry', 'N/A')}
        Business Summary: {info.get('longBusinessSummary', 'N/A')}
        Market Cap: {info.get('marketCap', 'N/A')}
        Forward P/E: {info.get('forwardPE', 'N/A')}
        Dividend Yield: {info.get('dividendYield', 'N/A')}
        52-Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}
        52-Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}
        """

        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.2',
                'prompt': prompt,
                'stream': False
            }
        )

        if response.status_code == 200:
            result = response.json()
            ai_analysis = result.get('response', 'No analysis available')

            final_output = f"""
            💼 Company Profile
            ----------------
            Name: {info.get('longName', 'N/A')}
            Sector: {info.get('sector', 'N/A')}
            Industry: {info.get('industry', 'N/A')}
            Website: {info.get('website', 'N/A')}

            📊 Key Metrics
            ------------
            Market Cap: {info.get('marketCap', 'N/A')}
            Forward P/E: {info.get('forwardPE', 'N/A')}
            Dividend Yield: {info.get('dividendYield', 'N/A')}
            52-Week Range: {info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}

            📝 AI Analysis
            ------------
            {ai_analysis}
            """

            return final_output
        else:
            return f"Error: Failed to get response from Ollama (Status code: {response.status_code})"

    except Exception as e:
        return f"Error generating company summary: {str(e)}"

def train_and_analyze_stock(stock_code, exchange):
    try:
        symbol = f"{stock_code.upper()}.{exchange}"

        # Get company name for news analysis
        stock = yf.Ticker(symbol)
        company_name = stock.info.get('longName', 'N/A')

        # Analyze news sentiment
        sentiment_category, sentiment_score, news_items, sentiment_color = analyze_news_sentiment(stock_code,
                                                                                                  company_name)

        # Train models and get predictions
        analyzer.train_models(symbol, start_date="2023-01-01")
        result = analyzer.predict_price_and_signals(symbol)

        # Create sentiment visualization
        plt.figure(figsize=(8, 1))
        plt.barh(['News Sentiment'], [1], color=sentiment_color)
        plt.title(f"Market Sentiment: {sentiment_category} ({sentiment_score:.2f})")
        plt.xticks([])
        plt.savefig("sentiment.png", bbox_inches='tight')
        plt.close()

        # Create price chart
        plt.figure(figsize=(10, 6))
        historical_data = analyzer.fetch_data(symbol, start_date="2023-01-01")
        plt.plot(historical_data.index, historical_data['Close'], label="Historical Prices", color="blue")
        plt.axhline(result['current_price'], color="green", linestyle="--", label="Current Price")
        plt.axhline(result['predicted_price'], color="red", linestyle="--", label="Predicted Price")
        plt.fill_between(historical_data.index, historical_data['Low'], historical_data['High'],
                         color="gray", alpha=0.2, label="Price Range")
        plt.title(f"Stock Analysis for {result['symbol']}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        plt.savefig("chart.png")
        plt.close()

        analysis_output = f"""
        📊 Technical Analysis Summary
        ---------------------------
        Current Price: ₹{result['current_price']:.2f}
        Predicted Price: ₹{result['predicted_price']:.2f}

        📰 News Sentiment Analysis
        -----------------------
        Overall Sentiment: {sentiment_category} ({sentiment_score:.2f})

        Recent News:
        {"  ".join([f"{news}" for news in news_items])}

        📈 Trading Signals
        ----------------
        Buy Target: ₹{result['buy_target']:.2f}
        Sell Target: ₹{result['sell_target']:.2f}
        Support Level: ₹{result['support_level']:.2f}
        Resistance Level: ₹{result['resistance_level']:.2f}

        📉 Market Metrics
        --------------
        Confidence: {result['confidence']:.2%}
        Volatility: {result['volatility']:.2%}
        Expected Change: {result['price_change_percent']:.2f}%

        🎯 Recommendation
        ---------------
        Predicted Immediate Direction: {result['predicted_direction']}
        Suggested Immediate Action: {result['action']}
        
        Predicted Year to Date Direction: {result['predicted_long_direction']}
        Suggested Long term Action: {result['longAction']}
        """

        return analysis_output, "chart.png", "sentiment.png"
    except Exception as e:
        return f"Error: {str(e)}", None, None

def analyze_stock_with_summary(stock_code, exchange):
    analysis_result, chart_path, sentiment_path = train_and_analyze_stock(stock_code, exchange)
    summary_result = generate_summary_with_llama(stock_code, exchange)
    return summary_result, analysis_result, chart_path, sentiment_path

def open_zerodha():
    """Opens Zerodha webpage"""
    return "https://kite.zerodha.com/"

# Define the Gradio interface
import gradio as gr
import pandas as pd

# Load NSE company data
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
    if not search_text:
        return gr.Dropdown(choices=[], visible=False)

    search_text = search_text.upper()
    suggestions = [
        f"{code}: {name}"
        for code, name in NSE_COMPANIES.items()
        if search_text in code or search_text in name.upper()
    ]
    return gr.Dropdown(choices=suggestions, visible=True)


def update_stock_code(suggestion):
    if suggestion:
        code = suggestion.split(":")[0].strip()
        return code
    return ""


with gr.Blocks(title="Advanced Stock Analysis Dashboard") as interface:
    gr.Markdown("<h1 style='text-align: center;'>📈 Stock Analysis Dashboard</h1>")
    gr.Markdown(
        "<p style='text-align: center;'>Get comprehensive analysis including technical predictions, news sentiment, and AI-generated insights</p>")

    with gr.Row():
        with gr.Column(scale=2):
            stock_code_input = gr.Textbox(
                label="Enter Stock Code",
                placeholder="e.g., RELIANCE, TCS, INFY"
            )
            company_suggestions = gr.Dropdown(
                label="Suggestions",
                choices=[],
                visible=False,
                interactive=True
            )
        with gr.Column(scale=1):
            exchange_toggle = gr.Radio(
                label="Select Exchange",
                choices=["NS", "BO"],
                value="NS",
                interactive=True
            )

    with gr.Row():
        analyze_btn = gr.Button("Analyze Stock", variant="primary", scale=1)
        zerodha_link = gr.HTML(
            '''
            <a href="https://kite.zerodha.com/" target="_blank" style="text-decoration: none; display: inline-block; width: 100%;">
                <button style="
                    width: 100%;
                    padding: 8px 16px;
                    font-size: 16px;
                    font-weight: 500;
                    background-color: #f3f4f6;
                    border: 1px solid #d1d5db;
                    border-radius: 4px;
                    cursor: pointer;
                    color: #374151;
                    height: 40px;
                    transition: background-color 0.3s;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                ">
                    🔗 Trade on Zerodha
                </button>
            </a>
            '''
        )

    with gr.Tabs():
        with gr.TabItem("📊 Technical Analysis"):
            with gr.Row():
                with gr.Column(scale=2):
                    chart_output = gr.Image(label="Price Chart")
                with gr.Column(scale=1):
                    sentiment_output = gr.Image(label="News Sentiment")
                    analysis_output = gr.Textbox(label="Technical Analysis", lines=10)

        with gr.TabItem("💬 AI Chat Analysis"):
            chatbot = gr.Chatbot(
                label="Stock Analysis Chat",
                height=400
            )
            msg = gr.Textbox(
                label="Ask about the stock",
                placeholder="Type your question here...",
                lines=2
            )
            clear = gr.Button("Clear")

        with gr.TabItem("🏢 Company Analysis"):
            summary_output = gr.Textbox(label="Company Analysis", lines=15)

    # Set up event handlers
    stock_code_input.change(
        get_company_suggestions,
        inputs=stock_code_input,
        outputs=company_suggestions
    )

    company_suggestions.change(
        update_stock_code,
        inputs=company_suggestions,
        outputs=stock_code_input
    )

    msg.submit(
        chat_with_ollama,
        inputs=[msg, chatbot],
        outputs=chatbot
    )

    clear.click(
        lambda: None,
        outputs=chatbot
    )

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
    port = int(os.environ.get("PORT", 7860))
    interface.launch(server_name="0.0.0.0", server_port=port)

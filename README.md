# **STOCK-MARKET-PREDICTOR**

This repository contains an Advanced Stock Market Prediction system that provides AI-powered technical analysis and real-time market intelligence using machine learning and Large Language Models to analyze stock data and generate investment recommendations.

**Learn More**
* [Gradio Documentation](https://gradio.app/docs/) - ML web interfaces
* [Ollama Documentation](https://ollama.ai/docs) - Local LLMs
* [yfinance Documentation](https://pypi.org/project/yfinance/) - Yahoo Finance API

## **Table of Contents**
* [Overview](#overview)
* [Features](#features)
* [Project Structure](#project-structure)
* [Getting Started](#getting-started)
* [AI Setup](#ai-setup)
* [Contributing](#contributing)
* [License](#license)

## **Overview**
AI-powered stock market prediction dashboard combining traditional technical analysis with modern AI capabilities. Features ML price predictions, sentiment analysis, and intelligent chat-based stock insights for Indian stock markets (NSE/BSE).

## **Features**
* **🤖 AI Chat Assistant**: Intelligent responses using Ollama LLMs
* **📊 Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages
* **📈 Price Predictions**: Random Forest ML models for forecasting
* **📰 News Sentiment**: Real-time sentiment scoring
* **🎯 Trading Signals**: Buy/sell/hold recommendations
* **📊 Interactive Charts**: Price trends and volume analysis
* **💰 Risk Assessment**: Volatility and support/resistance levels

## **Project Structure**

```
└── Stock-Market-Predictor/
    ├── README.md
    ├── ui7.py                     # Main application file
    └── requirements.txt
       
```

### **Key Components**
* **StockAnalyzer Class** - Core ML engine for technical analysis
* **OllamaManager Class** - AI chat functionality
* **Technical Indicators** - RSI, MACD, Bollinger Bands, Moving Averages
* **Gradio Interface** - User-friendly web interface

## **Getting Started**

### **Prerequisites**
* Python 3.8+
* pip package manager
* Internet connection for real-time data
* Ollama (optional, for AI features)

### **Installation**
1. Clone the repository:
```bash
git clone https://github.com/naveenreddy1334/Stock-Market-Predictor
cd Stock-Market-Predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### **Usage**
```bash
python main.py
```

Open http://localhost:7860 in your browser.

### **Basic Workflow**
1. Enter Indian stock symbol (e.g., RELIANCE, TCS, INFY)
2. Select exchange (NSE/BSE)
3. Click "🚀 Start Analysis"
4. View ML predictions and chat with AI

## **AI Setup**

### **Ollama Integration**
For enhanced AI features:

1. **Download Ollama**: Visit [ollama.ai](https://ollama.ai/download)
2. **Start Service**: `ollama serve`
3. **Install Model**: `ollama pull llama3.2`
4. **Refresh**: Click refresh button in the web interface

### **Supported Models**
* llama3.2:latest (Recommended)
* llama3.2:1b (Faster)
* Any Ollama-compatible model

## **Contributing**
* **💬 Join Discussions**: Share insights and feedback
* **🐛 Report Issues**: Submit bugs or feature requests
* **💡 Submit PRs**: Review and contribute code improvements

### **Contributing Guidelines**
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## **Disclaimer**
⚠️ **Educational purposes only**. Not financial advice. Always conduct your own research before investing.



# Reddit Sentiment Analysis & Stock Market Correlation

## 📊 Project Overview
This project analyzes the relationship between Reddit sentiment and stock market movements by scraping posts from r/wallstreetbets, performing sentiment analysis, and correlating the results with actual stock price data. The analysis includes sentiment tracking, mention counting, topic modeling, and various visualization techniques to understand the relationship between social media discourse and stock market behavior.

## 🚀 Features
- Reddit data scraping from r/wallstreetbets
- Sentiment analysis of Reddit posts
- Stock price data collection using yfinance
- Topic modeling using Latent Dirichlet Allocation (LDA)
- Multiple visualization techniques:
  - Sentiment over time
  - Stock price vs. sentiment correlation
  - Mention frequency analysis
  - Correlation heatmaps

## 🛠️ Prerequisites
```
- Python 3.x
- praw
- pandas
- numpy
- matplotlib
- seaborn
- textblob
- yfinance
- scikit-learn
- nltk
```

## 📦 Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up Reddit API credentials:
   - Create a Reddit account
   - Go to https://www.reddit.com/prefs/apps
   - Create a new application
   - Replace the credentials in the code with your own:
     ```python
     reddit = praw.Reddit(
         client_id='your_client_id',
         client_secret='your_client_secret',
         user_agent='your_user_agent'
     )
     ```

## 🏗️ Project Structure
```
project/
├── Project2.py                  # Main script
├── Visualizations/              # Folder containing visualized data
│   ├── Raw_reddit_data/         # Folder for raw Reddit data CSV files
│   │   ├── raw_reddit_data_*.csv  # Raw Reddit data for each stock
│   ├── Processed_reddit_data/   # Folder for processed Reddit data CSV files
│   │   ├── processed_data_*.csv   # Processed Reddit data for each stock
│   ├── Stock_data/              # Folder for processed stock data CSV files
│   │   ├── stock_data_*.csv       # Processed stock data for each stock
│
├── Data/                        # Folder containing data visualizations
│   ├── Sentiment_over_time/     # Folder for sentiment over time visualizations
│   │   ├── sentiment_over_time_*.png   # Sentiment over time for each stock
│   ├── Sentiment_vs_price/      # Folder for sentiment vs price visualizations
│   │   ├── sentiment_vs_price_*.png    # Sentiment vs price for each stock
│   ├── Mentions_over_time/      # Folder for mentions over time visualizations
│   │   ├── mentions_over_time_*.png    # Mentions over time for each stock
│   ├── Correlations_heatmap/    # Folder for correlation heatmaps
│   │   ├── correlation_heatmap_*.png   # Correlation heatmap for each stock
│
└── README.md                    # Documentation for the project

```

## 🔍 Core Components

### 1. Data Collection
- **Reddit Scraping**: Uses PRAW (Python Reddit API Wrapper) to collect posts from r/wallstreetbets
- **Stock Data**: Fetches historical stock data using yfinance
- Function: `scrape_reddit_data()`, `get_stock_data()`

### 2. Data Analysis & Feature Extraction
- Sentiment analysis using TextBlob
- Stock mention counting
- Date-based aggregation
- Functions: `analyze_sentiment()`, `process_data()`

### 3. Topic Modeling
- Text preprocessing with NLTK
- Latent Dirichlet Allocation (LDA) implementation
- Functions: `preprocess_text()`, `perform_topic_modeling()`

### 4. Visualization & Reporting
Multiple visualization functions:
- `plot_sentiment_over_time()`: Tracks sentiment changes across time
- `plot_sentiment_vs_price()`: Compares sentiment with stock prices
- `plot_mentions_over_time()`: Shows frequency of stock mentions
- `plot_correlation_heatmap()`: Displays correlations between different metrics

## 📈 Output
The script generates several types of outputs in the `output` directory:

### CSV Files:
- Raw Reddit data
- Processed Reddit data with sentiment scores
- Historical stock price data

### Visualizations:
- Sentiment trends over time
- Stock price vs. sentiment comparisons
- Mention frequency analysis
- Correlation heatmaps

## 🎯 Usage
Run the main script:
```bash
python Project2.py
```

The script will:
1. Create an 'output' directory
2. Process each stock in the predefined list
3. Generate all visualizations and data files
4. Save results in the output directory

## 📊 Analyzed Stocks
The project currently analyzes the following stocks:
- INTC (Intel)
- TSLA (Tesla)
- WBA (Walgreens)
- BA (Boeing)
- SMCI (Super Micro Computer)
- QCOM (Qualcomm)
- FDX (FedEx)
- LUNR (Intuitive Machines)
- LW (Lamb Weston)
- VZ (Verizon)
- MKC (McCormick)
- COIN (Coinbase)



import praw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Download required NLTK resources for text processing
# punkt: For sentence tokenization
# stopwords: For removing common words that don't add meaning
nltk.download('punkt')
nltk.download('stopwords')

# 1. Data Collection Functions
def scrape_reddit_data(subreddit_name, query, limit=1000):
    """
    Scrapes data from Reddit using PRAW (Python Reddit API Wrapper)
    Parameters:
        subreddit_name: Target subreddit to scrape
        query: Search term (usually stock ticker)
        limit: Maximum number of posts to retrieve
    Returns:
        DataFrame containing post data
    """
    # Initialize Reddit API connection
    reddit = praw.Reddit(client_id='a2Pa80XclX4btPF6MIsHfQ',
                         client_secret='qQv5Hwuy0Z1_Z3sLNG2zY-DwtLPLfg',
                         user_agent='my_reddit_scraper/0.4')
    
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    
    # Collect post data including title, score, comments, etc.
    for post in subreddit.search(query, limit=limit):
        posts.append({
            'title': post.title,
            'score': post.score,  # Post karma/upvotes
            'id': post.id,        # Unique post identifier
            'url': post.url,      # Post URL
            'comms_num': post.num_comments,  # Number of comments
            'created': datetime.fromtimestamp(post.created),  # Post creation time
            'body': post.selftext  # Post content
        })
    
    return pd.DataFrame(posts)

# 2. Data Analysis & Feature Extraction Functions
def analyze_sentiment(text):
    """
    Performs sentiment analysis on text using TextBlob
    Returns polarity score between -1 (negative) and 1 (positive)
    """
    return TextBlob(text).sentiment.polarity

def process_data(df, stock_ticker):
    """
    Processes raw Reddit data by:
    - Adding sentiment scores
    - Converting timestamps to dates
    - Counting stock mentions
    """
    # Calculate sentiment scores for post titles
    df['sentiment'] = df['title'].apply(analyze_sentiment)
    # Convert timestamp to date for aggregation
    df['date'] = df['created'].dt.date
    # Count mentions of the stock ticker in title and body
    df['mention_count'] = df['title'].str.count(stock_ticker) + df['body'].str.count(stock_ticker)
    return df

def get_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data using yfinance
    Returns DataFrame with OHLCV data
    """
    stock = yf.Ticker(ticker)
    return stock.history(start=start_date, end=end_date)

# 3. Topic Modeling Functions
def preprocess_text(text):
    """
    Preprocesses text for topic modeling:
    - Converts to lowercase
    - Tokenizes text
    - Removes stopwords and non-alphanumeric tokens
    """
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in tokens if word.isalnum() and word not in stop_words])

def perform_topic_modeling(df, n_topics=5, n_top_words=10):
    """
    Performs Latent Dirichlet Allocation (LDA) topic modeling
    Parameters:
        df: DataFrame containing text data
        n_topics: Number of topics to identify
        n_top_words: Number of words to include per topic
    Returns:
        List of topics with their top words
    """
    # Preprocess text data
    preprocessed_docs = df['body'].fillna('').apply(preprocess_text)
    
    # Create document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(preprocessed_docs)
    
    # Perform LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)
    
    # Extract feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Generate topic summaries
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    
    return topics

# 4. Visualization & Reporting Functions
def plot_sentiment_over_time(df, stock_ticker):
    """
    Creates a line plot showing average daily sentiment over time
    Handles errors and saves plot to file
    """
    try:
        print(f"Plotting sentiment over time for {stock_ticker}...")
        # Calculate daily average sentiment
        daily_sentiment = df.groupby('date')['sentiment'].mean().reset_index()
        
        # Create and save plot
        plt.figure(figsize=(12, 6))
        plt.plot(daily_sentiment['date'], daily_sentiment['sentiment'])
        plt.title(f'Average Daily Sentiment for {stock_ticker}')
        plt.xlabel('Date')
        plt.ylabel('Sentiment')
        plt.savefig(f'sentiment_over_time_{stock_ticker}.png')
        plt.close()
        print(f"Sentiment over time plot saved for {stock_ticker}")
    except Exception as e:
        print(f"Error plotting sentiment over time for {stock_ticker}: {str(e)}")

def plot_sentiment_vs_price(sentiment_df, price_df, stock_ticker):
    """
    Creates a dual-axis plot comparing sentiment and stock price
    Handles date alignment and timezone issues
    """
    try:
        print(f"Plotting sentiment vs price for {stock_ticker}...")
        # Align dates and timezones
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
        
        # Merge sentiment and price data
        merged_df = pd.merge(sentiment_df, price_df, left_on='date', right_index=True, how='inner')
        
        if merged_df.empty:
            print(f"No overlapping data found for {stock_ticker}")
            return
        
        # Create dual-axis plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot sentiment on primary y-axis
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sentiment', color='tab:blue')
        ax1.plot(merged_df['date'], merged_df['sentiment'], color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Plot price on secondary y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Stock Price', color='tab:orange')
        ax2.plot(merged_df['date'], merged_df['Close'], color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        
        plt.title(f'Sentiment vs Stock Price for {stock_ticker}')
        fig.tight_layout()
        plt.savefig(f'sentiment_vs_price_{stock_ticker}.png')
        plt.close()
        print(f"Sentiment vs price plot saved for {stock_ticker}")
    except Exception as e:
        print(f"Error plotting sentiment vs price for {stock_ticker}: {str(e)}")

def plot_mentions_over_time(df, stock_ticker):
    """
    Creates a line plot showing the number of daily mentions of the stock
    """
    try:
        print(f"Plotting mentions over time for {stock_ticker}...")
        # Calculate daily mention counts
        daily_mentions = df.groupby('date')['mention_count'].sum().reset_index()
        
        # Create and save plot
        plt.figure(figsize=(12, 6))
        plt.plot(daily_mentions['date'], daily_mentions['mention_count'])
        plt.title(f'Daily Mentions of {stock_ticker}')
        plt.xlabel('Date')
        plt.ylabel('Number of Mentions')
        plt.savefig(f'mentions_over_time_{stock_ticker}.png')
        plt.close()
        print(f"Mentions over time plot saved for {stock_ticker}")
    except Exception as e:
        print(f"Error plotting mentions over time for {stock_ticker}: {str(e)}")

def plot_correlation_heatmap(df, stock_data, stock_ticker):
    """
    Creates a heatmap showing correlations between:
    - Sentiment
    - Mention count
    - Post score
    - Stock price
    """
    try:
        print(f"Plotting correlation heatmap for {stock_ticker}...")
        # Aggregate daily metrics
        daily_data = df.groupby('date').agg({
            'sentiment': 'mean',     # Average daily sentiment
            'mention_count': 'sum',  # Total daily mentions
            'score': 'mean'         # Average daily post score
        }).reset_index()
        
        # Merge with stock price data
        merged_data = pd.merge(daily_data, stock_data['Close'], left_on='date', right_index=True, how='inner')
        
        if merged_data.empty:
            print(f"No overlapping data found for {stock_ticker}")
            return
        
        # Calculate and plot correlation matrix
        corr_matrix = merged_data[['sentiment', 'mention_count', 'score', 'Close']].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title(f'Correlation Heatmap for {stock_ticker}')
        plt.savefig(f'correlation_heatmap_{stock_ticker}.png')
        plt.close()
        print(f"Correlation heatmap saved for {stock_ticker}")
    except Exception as e:
        print(f"Error plotting correlation heatmap for {stock_ticker}: {str(e)}")

# Main execution
if __name__ == "__main__":
    # Configuration
    subreddit = 'wallstreetbets'  # Target subreddit
    # List of stock tickers to analyze
    stock_list = ['INTC','TSLA','WBA','BA','SMCI','QCOM','FDX','LUNR','LW','VZ','MKC','COIN']
    
    # Create output directory for results
    os.makedirs('output', exist_ok=True)
    os.chdir('output')
    
    # Initialize containers for aggregated data
    all_sentiment_data = {}
    all_mention_data = {}
    
    # Process each stock ticker
    for stock_ticker in stock_list:
        print(f"Analyzing {stock_ticker}...")
        
        # 1. Data Collection
        df = scrape_reddit_data(subreddit, stock_ticker)
        
        if df.empty:
            print(f"No data found for {stock_ticker}. Skipping to next stock.")
            continue
        
        # Save raw scraped data
        df.to_csv(f'raw_reddit_data_{stock_ticker}.csv', index=False)
        
        # 2. Process and analyze data
        df = process_data(df, stock_ticker)
        df.to_csv(f'processed_reddit_data_{stock_ticker}.csv', index=False)
        
        # 3. Fetch corresponding stock data
        start_date = df['date'].min()
        end_date = df['date'].max()
        stock_data = get_stock_data(stock_ticker, start_date, end_date)
        stock_data.to_csv(f'stock_data_{stock_ticker}.csv')
        
        # 4. Generate visualizations
        plot_sentiment_over_time(df, stock_ticker)
        plot_sentiment_vs_price(df, stock_data, stock_ticker)
        plot_mentions_over_time(df, stock_ticker)
        plot_correlation_heatmap(df, stock_data, stock_ticker)

print("Analysis complete. Results are saved in the 'output' directory.")

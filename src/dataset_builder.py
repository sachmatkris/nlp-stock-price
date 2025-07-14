import pandas as pd
from src.utils.feature_engineering import add_date_indicators, market_features_df, peers_features_df, stock_technicals_df
from src.utils.market_news_sentiment import add_sentiment_score, add_market_news

# CHOOSE STOCK:
STOCK_NAME = "DISNEY"
STOCK_TICKER = "DIS"
NO_OF_PAGES_IN_BUSINESS_INSIDER = 113
PEER_TICKERS = ["NFLX", "PARA", "WBD", "CMCSA", "SONY"]
MARKET_TICKERS = {
        "vix": "^VIX",
        "sp500": "^GSPC",
        "nasdaq": "^NDX",
    }
OUTPUT_PATH = "df_disney_final.csv"

if __name__ == '__main__':
    
    df_stock = stock_technicals_df(STOCK_TICKER)
    print('Stock prices downloaded and technical indicators added.')
    
    df_stock = add_market_news(df_stock, STOCK_TICKER, NO_OF_PAGES_IN_BUSINESS_INSIDER)
    print('Headlines processed.')

    df_peers_features = peers_features_df(PEER_TICKERS)
    print('Peers data downloaded and added.')
    
    df_market_features = market_features_df(MARKET_TICKERS)
    print('Market indicators data downloaded and added.')

    df_market = pd.merge(df_market_features, df_peers_features, left_index=True, right_index=True, how='left')
    df_stock = pd.merge(df_stock, df_market, left_index=True, right_index=True, how='left')
    
    df_stock = add_date_indicators(df_stock)
    df_stock = add_sentiment_score(df_stock)
    print('Date indicators and sentiment scores added.')
        
    df_stock.to_csv(OUTPUT_PATH)
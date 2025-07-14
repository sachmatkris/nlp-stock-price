import numpy as np
import pandas as pd
import yfinance as yf


def add_technical_indicators(df_orig: pd.DataFrame) -> pd.DataFrame:
    df = df_orig.copy()
    # === Log Returns ===
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    # === Percentage Returns === 
    df['pct_return'] = df['Close'].pct_change()
    # === Moving Averages and Standard Deviations ===
    df['sma_5'] = df['Close'].rolling(window=5).mean()
    df['sma_21'] = df['Close'].rolling(window=21).mean()
    df['sma_63'] = df['Close'].rolling(window=63).mean()
    df['std_5'] = df['Close'].rolling(window=5).std()
    df['std_21'] = df['Close'].rolling(window=21).std()
    df['std_63'] = df['Close'].rolling(window=63).std()
    # === Price Movement ===
    df['price_range'] = df['High'] - df['Low']
    df['price_change'] = df['Close'] - df['Open']
    # === Relative Strength Index (RSI) ===
    delta = df['Close'].diff()
    for window in [5, 21, 63]:
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
    # === MACD ===
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # === Momentum ===
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_21'] = df['Close'] - df['Close'].shift(21)
    df['Momentum_63'] = df['Close'] - df['Close'].shift(63)
    # === On-Balance Volume (OBV) ===
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    # === Volume Z-Score ===
    df['Volume_Z'] = (df['Volume'] - df['Volume'].rolling(21).mean()) / df['Volume'].rolling(21).std()
    # === True Range (TR) ===
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    df['True_Range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # === Average True Range (ATR) ===
    df['ATR_10'] = df['True_Range'].rolling(window=10).mean()
    # === Bollinger Bands ===
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_upper'] = sma20 + 2 * std20
    df['BB_lower'] = sma20 - 2 * std20
    df['BB_percent'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    # === Stochastic Oscillator (%K and %D) ===
    low10 = df['Low'].rolling(10).min()
    high10 = df['High'].rolling(10).max()
    df['stoch_K'] = 100 * (df['Close'] - low10) / (high10 - low10)
    df['stoch_D'] = df['stoch_K'].rolling(3).mean()
    # === Lagged Returns ===
    for lag in [1, 5, 10]:
        df[f'return_lag_{lag}'] = df['pct_return'].shift(lag)  
    # === Realized Volatility ===
    for window in [5, 21, 63]:
        df[f'realized_vol_{window}'] = df['log_return'].rolling(window).std() * np.sqrt(252)
    return df

def stock_technicals_df(ticker):
    df_yf = yf.download(ticker, interval="1d", start="2015-01-01", end="2024-12-31")
    df_yf.columns = df_yf.columns.droplevel(1)
    df_technical = add_technical_indicators(df_yf)
    return df_technical


def add_market_technical_indicators(df_orig: pd.DataFrame) -> pd.DataFrame:
    df = df_orig.copy()
    # === Percentage Returns === 
    df['pct_return'] = df['Close'].pct_change()
    # === Moving Averages and Standard Deviations ===
    for window in [5, 21, 63]:
        df[f'sma_{window}'] = df['Close'].rolling(window).mean()
        df[f'std_{window}'] = df['Close'].rolling(window).std()
    # === RSI ===
    delta = df['Close'].diff()
    for window in [5, 21, 63]:
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-6)  # prevent div by 0
        df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
    # === MACD ===
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # === Momentum ===
    for lag in [5, 21, 63]:
        df[f'Momentum_{lag}'] = df['Close'] - df['Close'].shift(lag)
    # === Realized Volatility ===
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    for window in [5, 21, 63]:
        df[f'realized_vol_{window}'] = log_returns.rolling(window).std() * np.sqrt(252)
    return df

def market_features_df(tickers):
    df_market_features = pd.DataFrame()
    for name, symbol in tickers.items():
        df = yf.download(symbol, start="2015-01-01", end="2024-12-31", interval="1d")
        if not df.empty:
            df_feat = add_market_technical_indicators(df)
            df_feat.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_feat.columns]
            df_feat = df_feat.add_prefix(f"{name}_")
            df_feat.index.name = "Date"
            df_feat = df_feat.filter(regex=f"^{name}_(pct_return|sma|std|RSI|MACD|Momentum|realized_vol)")
            if df_market_features.empty:
                df_market_features = df_feat
            else:
                df_market_features = df_market_features.join(df_feat, how='outer')
    df_market_features = df_market_features.sort_index().ffill()
    return(df_market_features)



def add_peer_indicators(df_orig: pd.DataFrame, window_short=5, window_long=21):
    df = df_orig.copy()
    returns = df['Close'].pct_change()
    result = pd.DataFrame(index=df.index)
    result[f'mean_return_{window_short}'] = returns.rolling(window_short).mean()
    result[f'mean_return_{window_long}'] = returns.rolling(window_long).mean()
    result[f'vol_{window_short}'] = returns.rolling(window_short).std()
    result[f'vol_{window_long}'] = returns.rolling(window_long).std()
    return result

def peers_features_df(tickers):
    df_peers_features = pd.DataFrame()
    for ticker in tickers:
        df_peer = yf.download(ticker, start="2015-01-01", end="2024-12-31", interval="1d")
        df_features = add_peer_indicators(df_peer)
        df_features = df_features.add_prefix(f'{ticker}_')
        if df_peers_features.empty:
            df_peers_features = df_features
        else:
            df_peers_features = df_peers_features.join(df_features, how='outer')
    df_peers_features = df_peers_features.ffill()
    return df_peers_features



def add_date_indicators(df_orig: pd.DataFrame):
    # === Date indicators ===
    df_orig['month_sin'] = np.sin(2 * np.pi * df_orig.index.month / 12)
    df_orig['month_cos'] = np.cos(2 * np.pi * df_orig.index.month / 12)

    df_orig['day_sin'] = np.sin(2 * np.pi * df_orig.index.day / 31)
    df_orig['day_cos'] = np.cos(2 * np.pi * df_orig.index.day / 31)

    df_orig['weekday_sin'] = np.sin(2 * np.pi * df_orig.index.weekday / 7)
    df_orig['weekday_cos'] = np.cos(2 * np.pi * df_orig.index.weekday / 7)
    return df_orig
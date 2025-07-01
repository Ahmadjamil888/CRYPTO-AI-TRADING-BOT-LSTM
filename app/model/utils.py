# app/model/utils.py

import pandas as pd
import numpy as np

def compute_rsi(series, period=14):
    """
    Compute the Relative Strength Index (RSI) for a given series.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def load_crypto_data(filepath='data/crypto-markets.csv', symbol='BTC'):
    """
    Load, filter, and preprocess crypto market data with RSI and EMA indicators.

    Args:
        filepath (str): Path to the CSV file.
        symbol (str): Cryptocurrency symbol to filter (e.g., 'BTC').

    Returns:
        pd.DataFrame: Processed DataFrame with features for model input.
    """
    # Load CSV
    df = pd.read_csv(filepath)

    # Filter by symbol
    df = df[df['symbol'] == symbol.upper()]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Select core columns
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()

    # Compute technical indicators
    df['ema'] = df['close'].ewm(span=14).mean()
    df['rsi'] = compute_rsi(df['close'])

    # Drop rows with missing values (from indicators)
    df.dropna(inplace=True)

    return df

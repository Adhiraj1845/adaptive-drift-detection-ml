# src/utils.py
import os
import yfinance as yf
import pandas as pd

def download_financial_data(ticker, start_date, end_date):
    """
    Download and clean historical financial data using yfinance.
    Output schema is standardized to: Open, High, Low, Close, Volume
    """
    df = yf.download(ticker, start=start_date, end=end_date)

    # Flatten column index if multi-level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Keep only the columns we will use (no Adj Close)
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_cols = [c for c in required_cols if c in df.columns]
    df = df[available_cols]

    df.dropna(inplace=True)
    return df

def save_data(df, path):
    """
    Save DataFrame to CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)

def load_data(path):
    """
    Load DataFrame from CSV.
    """
    return pd.read_csv(path, index_col=0)

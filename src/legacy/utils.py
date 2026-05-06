import os
import yfinance as yf
import pandas as pd

def download_financial_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_cols = [c for c in required_cols if c in df.columns]
    df = df[available_cols]

    df.dropna(inplace=True)
    return df

def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)

def load_data(path):
    return pd.read_csv(path, index_col=0)

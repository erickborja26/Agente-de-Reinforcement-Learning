import pandas as pd
import yfinance as yf

def load_yahoo_close(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = df.rename(columns=str.lower)
    df = df[["close"]].copy()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

import pandas as pd
import yfinance as yf
from config import Config

def init_mt5():
    """
    Mock initialization for MT5 since we switched to yfinance
    """
    print("Using yfinance instead of MT5 for testing.")
    return True

def fetch_data():
    """
    Fetches the 5m candles for EURUSD using yfinance.
    Returns a Pandas DataFrame.
    """
    print(f"Fetching data for EURUSD via yfinance...")
    
    # EURUSD ticker in yfinance is EURUSD=X
    # 5m interval is supported up to 60 days
    ticker = yf.Ticker("EURUSD=X")
    df = ticker.history(period="1mo", interval="5m")
    
    if df is None or df.empty:
        print("Failed to fetch rates from yfinance.")
        return None
        
    # yfinance returns column names capitalized (Open, High, Low, Close, Volume)
    # We rename them to lowercase to match our existing code
    df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'tick_volume'
    }, inplace=True)
    
    print(f"Successfully fetched {len(df)} rows from yfinance.")
    return df

if __name__ == "__main__":
    if init_mt5():
        df = fetch_data()
        if df is not None:
            print(df.head())


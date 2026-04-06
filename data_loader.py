"""
Data Loader — Expert Fixed Time v2.3
======================================
Uses yfinance for instant historical data (1700+ candles in <1 second).
DXY is permanently frozen at 100.0 — no threading, no crashes.
MT5 is fully bypassed.
"""

import pandas as pd
import yfinance as yf
from config import Config


def init_mt5():
    """Legacy stub — MT5 is no longer used. Kept for import compatibility."""
    print("\033[93m[Data Loader] MT5 bypassed. Expert Fixed Time v2.3 active.\033[0m")
    return True


def fetch_data(**kwargs):
    """
    Fetches live candle data from yfinance (reliable, instant, 24/7).
    
    Returns:
        pd.DataFrame with columns: open, high, low, close, real_volume, DXY_Close
        or None on failure.
    """
    # Map broker symbols to yfinance tickers
    symbol_map = {
        "BTCUSD": "BTC-USD",
        "ETHUSD": "ETH-USD",
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "JPY=X",
        "XAUUSD": "GC=F",
    }
    
    ticker_name = symbol_map.get(Config.SYMBOL, "BTC-USD")
    print(f"Fetching Live Data for {ticker_name}... (Expert Fixed Time v2.3)")
    
    try:
        ticker = yf.Ticker(ticker_name)
        df = ticker.history(period="7d", interval="5m")
        
        if df is None or df.empty:
            print("\033[91m[Data] yfinance returned empty data.\033[0m")
            return None
        
        # Normalize column names to match pipeline expectations
        df.index = df.index.tz_localize(None)
        df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'real_volume'
        }, inplace=True)
        
        # Freeze DXY at 100.0 — no thread, no crash
        df['DXY_Close'] = 100.0
        print(f"\033[94m[DXY Freeze] Static DXY = 100.0 (no external dependency)\033[0m")
        
        print(f"\033[92m[Data] Loaded {len(df)} candles. Latest: {df['close'].iloc[-1]:.2f}\033[0m")
        return df
        
    except Exception as e:
        print(f"\033[91m[Data] yfinance fetch failed: {e}\033[0m")
        return None

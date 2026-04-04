import pandas as pd
import yfinance as yf
import MetaTrader5 as mt5
from config import Config

def init_mt5():
    """
    Initializes MT5 Connection
    """
    print("Initializing MT5 Connection...")
    if not mt5.initialize(login=Config.LOGIN, server=Config.SERVER, password=Config.PASSWORD):
        print("MT5 initialize() failed, error code =", mt5.last_error())
        return False
    return True

def fetch_data():
    """
    Fetches the live candles from MT5 (EURUSD) and merges DXY from yfinance.
    Returns a Pandas DataFrame.
    """
    print(f"Fetching Live Data from MT5 for {Config.SYMBOL}...")
    
    # 1. Fetch EURUSD from MT5
    rates = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME, 0, Config.DATA_POINTS)
    if rates is None or len(rates) == 0:
        print(f"Failed to fetch rates from MT5 for {Config.SYMBOL}, error code = {mt5.last_error()}")
        print("Falling back to yfinance for BTC-USD...")
        btc_ticker = yf.Ticker("BTC-USD")
        # 1m max allowed is 7 days => ~10,080 rows
        df_mt5 = btc_ticker.history(period="7d", interval="1m")
        if df_mt5.empty:
            print("Fallback failed. No data.")
            return None
        df_mt5.index = df_mt5.index.tz_localize(None)
        # Rename Open, High, Low, Close, Volume to lowercase to match MT5 format
        df_mt5.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'real_volume'}, inplace=True)
    else:
        df_mt5 = pd.DataFrame(rates)
        df_mt5['time'] = pd.to_datetime(df_mt5['time'], unit='s')
        df_mt5.set_index('time', inplace=True)
    
    print(f"Successfully fetched {len(df_mt5)} rows for primary asset.")
    
    # 2. Fetch DXY from yfinance
    print(f"Fetching DXY Live Data via yfinance ({Config.DXY_TICKER})...")
    dxy_ticker = yf.Ticker(Config.DXY_TICKER)
    df_dxy = dxy_ticker.history(period="1mo", interval="5m")
    
    if df_dxy is not None and not df_dxy.empty:
        # Remove timezone for smooth merging with MT5 naive datetime
        df_dxy.index = df_dxy.index.tz_localize(None) 
        df_dxy = df_dxy[['Close']].rename(columns={'Close': 'DXY_Close'})
        
        # 3. Merge MT5 and DXY
        print("Merging MT5 EURUSD and yfinance DXY...")
        df_combined = df_mt5.join(df_dxy, how='left')
        df_combined['DXY_Close'] = df_combined['DXY_Close'].ffill()
        df_combined.dropna(subset=['DXY_Close'], inplace=True)
        return df_combined
    else:
        print("Failed to fetch DXY. Continuing with EURUSD only.")
        return df_mt5

if __name__ == "__main__":
    if init_mt5():
        df = fetch_data()
        if df is not None:
            print(df.tail())
        mt5.shutdown()

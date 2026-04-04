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
    DXY is fetched in a parallel thread to avoid blocking the primary data fetch.
    Returns a Pandas DataFrame.
    """
    from threading import Thread
    
    print(f"Fetching Live Data from MT5 for {Config.SYMBOL}...")
    
    # ===== DXY THREAD (runs in background while we fetch BTC/MT5) =====
    dxy_result = {"df": None}
    
    def _fetch_dxy():
        try:
            print(f"[Thread] Fetching DXY via yfinance ({Config.DXY_TICKER})...")
            dxy_ticker = yf.Ticker(Config.DXY_TICKER)
            df_dxy = dxy_ticker.history(period="1mo", interval="5m")
            if df_dxy is not None and not df_dxy.empty:
                df_dxy.index = df_dxy.index.tz_localize(None)
                dxy_result["df"] = df_dxy[['Close']].rename(columns={'Close': 'DXY_Close'})
                print(f"[Thread] DXY fetched: {len(dxy_result['df'])} rows.")
            else:
                print("[Thread] DXY fetch returned empty.")
        except Exception as e:
            print(f"[Thread] DXY fetch failed: {e}")
    
    # Start DXY fetch in background immediately
    dxy_thread = Thread(target=_fetch_dxy, daemon=True)
    dxy_thread.start()
    
    # ===== PRIMARY ASSET (runs on main thread simultaneously) =====
    rates = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME, 0, Config.DATA_POINTS)
    if rates is None or len(rates) == 0:
        print(f"Failed to fetch rates from MT5 for {Config.SYMBOL}, error code = {mt5.last_error()}")
        print("Falling back to yfinance for BTC-USD...")
        btc_ticker = yf.Ticker("BTC-USD")
        df_mt5 = btc_ticker.history(period="7d", interval="1m")
        if df_mt5.empty:
            print("Fallback failed. No data.")
            return None
        df_mt5.index = df_mt5.index.tz_localize(None)
        df_mt5.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'real_volume'}, inplace=True)
    else:
        df_mt5 = pd.DataFrame(rates)
        df_mt5['time'] = pd.to_datetime(df_mt5['time'], unit='s')
        df_mt5.set_index('time', inplace=True)
    
    print(f"Successfully fetched {len(df_mt5)} rows for primary asset.")
    
    # ===== WAIT FOR DXY THREAD TO FINISH (max 10s) =====
    dxy_thread.join(timeout=10)
    
    if dxy_result["df"] is not None:
        print("Merging primary asset and DXY...")
        df_combined = df_mt5.join(dxy_result["df"], how='left')
        df_combined['DXY_Close'] = df_combined['DXY_Close'].ffill()
        df_combined.dropna(subset=['DXY_Close'], inplace=True)
        print(f"Merged data: {len(df_combined)} rows (dropped {len(df_mt5) - len(df_combined)} rows due to DXY gaps).")
        return df_combined
    else:
        print("DXY unavailable. Continuing with primary asset only.")
        return df_mt5

if __name__ == "__main__":
    if init_mt5():
        df = fetch_data()
        if df is not None:
            print(df.tail())
        mt5.shutdown()

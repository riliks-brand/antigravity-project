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

def fetch_data(weekend_mode=False):
    """
    Fetches the live candles from MT5 (EURUSD) or yfinance (BTC-USD in weekend mode) 
    and merges DXY from yfinance. DXY is frozen if in weekend mode to prevent model NaNs.
    """
    from threading import Thread
    
    ticker = "BTC-USD" if weekend_mode else Config.SYMBOL
    print(f"Fetching Live Data for {ticker}... (Weekend Mode: {weekend_mode})")
    
    # ===== DXY THREAD =====
    dxy_result = {"df": None}
    
    def _fetch_dxy():
        try:
            dxy_ticker = yf.Ticker(Config.DXY_TICKER)
            df_dxy = dxy_ticker.history(period="1mo", interval="5m")
            if df_dxy is not None and not df_dxy.empty:
                df_dxy.index = df_dxy.index.tz_localize(None)
                dxy_result["df"] = df_dxy[['Close']].rename(columns={'Close': 'DXY_Close'})
            else:
                print("[Thread] DXY fetch returned empty.")
        except Exception as e:
            print(f"[Thread] DXY fetch failed: {e}")
    
    dxy_thread = Thread(target=_fetch_dxy, daemon=True)
    dxy_thread.start()
    
    # ===== PRIMARY ASSET =====
    if weekend_mode:
        print("[Weekend Mode] Bypassing MT5. Using yfinance 24/7 data for BTC-USD...")
        btc_ticker = yf.Ticker("BTC-USD")
        # Fetch 5m interval to match the model training requirements (which uses 5m candles normally)
        # Note: If Config.TIMEFRAME says M1 but the user predicts 5-minute lookaheads, we ensure 
        # yfinance supplies matching granularities. Since the original used 5m for DXY and MT5 was whatever.
        df_primary = btc_ticker.history(period="7d", interval="5m")
        if df_primary.empty:
            print("Failed to fetch Yfinance data for BTC-USD.")
            return None
        df_primary.index = df_primary.index.tz_localize(None)
        df_primary.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'real_volume'}, inplace=True)
    else:
        rates = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME, 0, Config.DATA_POINTS)
        if rates is None or len(rates) == 0:
            print(f"Failed to fetch rates from MT5 for {Config.SYMBOL}, error code = {mt5.last_error()}")
            return None
        df_primary = pd.DataFrame(rates)
        df_primary['time'] = pd.to_datetime(df_primary['time'], unit='s')
        df_primary.set_index('time', inplace=True)
    
    # ===== MERGE DXY =====
    dxy_thread.join(timeout=10)
    
    if dxy_result["df"] is not None:
        if weekend_mode:
            # DXY is frozen on weekends. Use the last available value as static input to avoid NaN dropping overlapping BTC data
            last_dxy_val = dxy_result["df"]['DXY_Close'].dropna().iloc[-1]
            df_primary['DXY_Close'] = last_dxy_val
            print(f"\033[94m[DXY Freeze] Applied static DXY value: {last_dxy_val} to prevent pipeline corruption.\033[0m")
            df_combined = df_primary
        else:
            print("Merging primary asset and DXY...")
            df_combined = df_primary.join(dxy_result["df"], how='left')
            df_combined['DXY_Close'] = df_combined['DXY_Close'].ffill()
            df_combined.dropna(subset=['DXY_Close'], inplace=True)
            print(f"Merged data: {len(df_combined)} rows.")
        return df_combined
    else:
        print("DXY unavailable. Continuing with primary asset only.")
        return df_primary

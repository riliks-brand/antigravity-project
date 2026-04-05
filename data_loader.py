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
    Fetches the live candles from:
    - MT5 (EURUSD/BTCUSD) in normal weekday mode
    - yfinance (BTC-USD) in weekend mode
    - OTC Scraper (DOM/WebSocket) if Config.SYMBOL contains "OTC"
    
    Merges DXY from yfinance when available.
    DXY is frozen if in weekend/OTC mode to prevent model NaNs.
    """
    from threading import Thread
    
    # ===== OTC MODE DETECTION =====
    otc_mode = "OTC" in Config.SYMBOL.upper()
    
    if otc_mode:
        return _fetch_otc_data()
    
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


def _fetch_otc_data():
    """
    Fetches candle data from the OTC Scraper (DOM + WebSocket).
    Bypasses MT5 and yfinance entirely.
    Adds a frozen DXY value to maintain feature compatibility.
    """
    from otc_scraper import get_otc_scraper, OTCScraper
    
    print(f"\n\033[95m{'='*55}\033[0m")
    print(f"\033[95m       🎯 OTC MODE: Fetching from Browser DOM\033[0m")
    print(f"\033[95m{'='*55}\033[0m")
    
    scraper = get_otc_scraper(candle_interval=60)
    
    # Check readiness
    candle_count = OTCScraper.get_candle_count()
    min_required = Config.SEQUENCE_LENGTH + 50  # Need extra for indicator warmup
    
    if candle_count < min_required:
        print(f"\033[93m[OTC Data] Gathering candles... ({candle_count}/{min_required})\033[0m")
        print(f"\033[93m[OTC Data] WebSocket may provide instant history. DOM polling in progress.\033[0m")
        
        # Wait up to 30 seconds for WebSocket history
        if not OTCScraper._ws_history_loaded.wait(timeout=30):
            # Check again after waiting
            candle_count = OTCScraper.get_candle_count()
            if candle_count < min_required:
                print(f"\033[91m[OTC Data] Only {candle_count} candles available. Need {min_required}. Still gathering...\033[0m")
                return None
    
    # Fetch candles
    df = scraper.get_candles(count=Config.DATA_POINTS)
    
    if df is None or df.empty:
        print("\033[91m[OTC Data] Scraper returned empty data.\033[0m")
        return None
    
    # Add frozen DXY column for feature compatibility
    # OTC assets don't correlate with DXY but the model pipeline requires it
    df['DXY_Close'] = 100.0  # Neutral static value
    print(f"\033[94m[DXY Freeze] Applied static DXY value: 100.0 for OTC mode.\033[0m")
    
    # Ensure 'Volatility' column exists (some features reference it)
    if 'Volatility' not in df.columns:
        df['Volatility'] = 0
    
    print(f"\033[92m[OTC Data] Loaded {len(df)} OTC candles from browser. Latest: {df['close'].iloc[-1]:.5f}\033[0m")
    print(f"\033[95m{'='*55}\033[0m\n")
    
    return df

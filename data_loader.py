"""
Data Loader — Direct Platform Intelligence v2.2
=================================================
Smart data routing:
- Forex/Crypto (BTC, EUR/USD, etc): Uses yfinance for instant historical data
- OTC assets: Uses the OTC Scraper (DOM/WebSocket) when yfinance has no data

DXY is permanently frozen at 100.0 — no threading, no crashes.
MT5 is fully bypassed.
"""

import pandas as pd
import yfinance as yf
from config import Config


def init_mt5():
    """Legacy stub — MT5 is no longer used. Kept for import compatibility."""
    print("\033[93m[Data Loader] MT5 bypassed. Platform Intelligence v2.2 active.\033[0m")
    return True


def fetch_data(**kwargs):
    """
    Fetches live candle data.
    
    For Forex/Crypto assets (like BTC): Uses yfinance (reliable, instant, 24/7).
    For OTC assets: Falls back to the OTC Scraper (DOM/WebSocket).
    
    DXY is always frozen at 100.0 to prevent pipeline crashes.
    
    Returns:
        pd.DataFrame with columns: open, high, low, close, real_volume, DXY_Close
        or None on failure.
    """
    # Route to OTC scraper only for true OTC assets
    if "OTC" in Config.SYMBOL.upper():
        return _fetch_otc_data()
    
    # ===== FOREX/CRYPTO: Use yfinance (instant, reliable) =====
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
    print(f"Fetching Live Data for {ticker_name}... (Platform Intelligence v2.2)")
    
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


def _fetch_otc_data():
    """
    Fetches candle data from the OTC Scraper (DOM + WebSocket).
    Only used for TRUE OTC assets that have no external data source.
    """
    from otc_scraper import get_otc_scraper, OTCScraper
    
    print(f"\n\033[95m{'='*55}\033[0m")
    print(f"\033[95m       🎯 OTC MODE: Fetching from Browser DOM\033[0m")
    print(f"\033[95m{'='*55}\033[0m")
    
    scraper = get_otc_scraper(candle_interval=60)
    
    candle_count = OTCScraper.get_candle_count()
    min_required = Config.SEQUENCE_LENGTH + 50
    
    if candle_count < min_required:
        print(f"\033[93m[OTC Data] Gathering candles... ({candle_count}/{min_required})\033[0m")
        
        if not OTCScraper._ws_history_loaded.wait(timeout=30):
            candle_count = OTCScraper.get_candle_count()
            if candle_count < min_required:
                print(f"\033[91m[OTC Data] Only {candle_count} candles. Need {min_required}.\033[0m")
                return None
    
    df = scraper.get_candles(count=Config.DATA_POINTS)
    
    if df is None or df.empty:
        print("\033[91m[OTC Data] Scraper returned empty data.\033[0m")
        return None
    
    df['DXY_Close'] = 100.0
    print(f"\033[94m[DXY Freeze] Static DXY = 100.0 for OTC mode.\033[0m")
    
    if 'Volatility' not in df.columns:
        df['Volatility'] = 0
    
    print(f"\033[92m[OTC Data] Loaded {len(df)} candles. Latest: {df['close'].iloc[-1]:.5f}\033[0m")
    print(f"\033[95m{'='*55}\033[0m\n")
    
    return df

"""
Data Loader — Expert Fixed Time v2.3
======================================
Uses yfinance for instant historical data (1700+ candles in <1 second).
DXY is permanently frozen at 100.0 — no threading, no crashes.
MT5 is fully bypassed.
"""

import pandas as pd
import yfinance as yf
import re
from config import Config
from otc_scraper import get_otc_scraper


def init_mt5():
    """Legacy stub"""
    print("\033[93m[Data Loader] YFinance Dynamic Mode Active.\033[0m")
    return True


def fetch_data(**kwargs):
    """
    Reads the active market from the browser tab title, then fetches instant 
    historical data via yfinance for THAT specific market.
    """
    scraper = get_otc_scraper(candle_interval=60)
    session = scraper.get_playwright_session()
    
    active_asset = "Unknown"
    if session and session[3]:
        page = session[3]
        try:
            title = page.title()
            # Title looks like: '▼ 781.4734 Halal Market Axis | Trading platform for web traders – Olymptrade'
            if '|' in title:
                left_part = title.split('|')[0].strip()
                # Strip leading arrows, spaces, and numbers/decimals
                active_asset = re.sub(r'^[▲▼▶]?\s*[\d\.\,]+\s*', '', left_part).strip()
            elif '—' in title:
                active_asset = title.split('—')[0].strip()
            elif '-' in title:
                active_asset = title.split('-')[0].strip()
            else:
                active_asset = title.strip()
                
            # Failsafe if regex failed
            if not active_asset:
                active_asset = "Unknown"
                
        except Exception as e:
            print(f"[Title Parse Error] {e}")

    # Comprehensive Mapping of Olymp Trade names to yfinance tickers
    symbol_map = {
        # Crypto
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Litecoin": "LTC-USD",
        "Dash": "DASH-USD",
        "Ripple": "XRP-USD",
        "Basic Altcoin Index": "ETH-USD", # Proxy for altcoins
        "Bitcoin Cash": "BCH-USD",
        
        # Forex
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "USD/CHF": "CHF=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CAD": "CAD=X",
        "NZD/USD": "NZDUSD=X",
        "EUR/JPY": "EURJPY=X",
        "GBP/JPY": "GBPJPY=X",
        "EUR/GBP": "EURGBP=X",
        
        # Metals / Commodities
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Brent": "BZ=F",
        "Crude Oil": "CL=F",
        
        # Olymp Trade Exclusive Synthetic / Composite Indices
        "Asia Composite Index": "^N225", # Proxy Nikkei 225
        "Europe Composite Index": "VGK", # Europe ETF
        "Commodity Composite Index": "DBC", # Commodity ETF
        "Maha Jantar Index": "^NSEI", # Nifty 50 (India)
        "Arabian General Index": "^TASI", # Tadawul (KSA)
        "Halal Axis": "SPUS", # S&P 500 Shariah ETF proxy
        "Halal Market Axis": "SPUS", # S&P 500 Shariah ETF proxy
        "Latam Index": "ILF"  # Latin America 40 ETF
    }
    
    # Check if the active asset contains OTC
    clean_asset = active_asset.replace(" OTC", "").strip()
    
    ticker_name = symbol_map.get(clean_asset)
    
    # If not exactly found, try a fuzzy match
    if not ticker_name:
        if "Halal" in clean_asset:
            ticker_name = "SPUS"
        elif "Asia" in clean_asset:
            ticker_name = "^N225"
        elif "Europe" in clean_asset:
            ticker_name = "VGK"
        elif "Arabian" in clean_asset:
            ticker_name = "^TASI"

    if not ticker_name:
        print(f"\033[91m[Scout] FATAL: I don't know the real-world ticker for '{clean_asset}'.\033[0m")
        print(f"\033[93m[Scout] I cannot magically analyze '{clean_asset}' because it is missing from my symbol_map.\033[0m")
        print(f"\033[93m[Scout] Please select a known asset (like Bitcoin, EUR/USD, Asia Composite Index, Halal Axis).\033[0m")
        return None
    
    print(f"\033[96m[Scout] You are viewing: '{active_asset}' in the browser.\033[0m")
    print(f"\033[92m[Data] Fetching instant history for {ticker_name}...\033[0m")
    
    try:
        ticker = yf.Ticker(ticker_name)
        df = ticker.history(period="7d", interval="5m")
        
        if df is None or df.empty:
            print("\033[91m[Data] yfinance returned empty data.\033[0m")
            return None
        
        df.index = df.index.tz_localize(None)
        df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'real_volume'
        }, inplace=True)
        
        df = df[['open', 'high', 'low', 'close', 'real_volume']].copy()
        df['DXY_Close'] = 100.0
        
        return df
        
    except Exception as e:
        print(f"\033[91m[Data] yfinance fetch failed: {e}\033[0m")
        return None


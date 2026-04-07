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


def fetch_data(trading_mode='FIXED_TIME', manual_symbol=None):
    """
    If FIXED_TIME: Reads the active market from the browser tab title.
    If FOREX: Directly uses the symbol provided in config or manual_symbol.
    """
    active_asset = "Unknown"
    
    if trading_mode == 'FIXED_TIME':
        scraper = get_otc_scraper(candle_interval=60)
        session = scraper.get_playwright_session()
        
        if session and session[3]:
            page = session[3]
            try:
                title = page.title()
                # Title looks like: '▼ 781.4734 Halal Market Axis | Trading platform for web traders – Olymptrade'
                if '|' in title:
                    left_part = title.split('|')[0].strip()
                    active_asset = re.sub(r'^[▲▼▶]?\s*[\d\.\,]+\s*', '', left_part).strip()
                elif '—' in title:
                    active_asset = title.split('—')[0].strip()
                elif '-' in title:
                    active_asset = title.split('-')[0].strip()
                else:
                    active_asset = title.strip()
            except Exception as e:
                print(f"[Title Parse Error] {e}")
    else:
        # FOREX MODE: Use the hardcoded symbol from Config
        active_asset = manual_symbol if manual_symbol else getattr(Config, 'FOREX_SYMBOL', 'BTCUSD')

    # Comprehensive Mapping
    symbol_map = {
        # Crypto
        "Bitcoin": "BTC-USD",
        "BTCUSD": "BTC-USD",
        "BTCUSDm": "BTC-USD",
        "Ethereum": "ETH-USD",
        "ETHUSD": "ETH-USD",
        "ETHUSDm": "ETH-USD",
        "Litecoin": "LTC-USD",
        "Dash": "DASH-USD",
        "Ripple": "XRP-USD",
        "Basic Altcoin Index": "ETH-USD",
        "Bitcoin Cash": "BCH-USD",
        
        # Forex
        "EUR/USD": "EURUSD=X",
        "EURUSD": "EURUSD=X",
        "EURUSDm": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "GBPUSD": "GBPUSD=X",
        "GBPUSDm": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "USDJPY": "JPY=X",
        "USD/CHF": "CHF=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CAD": "CAD=X",
        "NZD/USD": "NZDUSD=X",
        "EUR/JPY": "EURJPY=X",
        "GBP/JPY": "GBPJPY=X",
        "EUR/GBP": "EURGBP=X",
        
        # Metals / Commodities
        "Gold": "GC=F",
        "XAUUSD": "GC=F",
        "XAUUSDm": "GC=F",
        "Silver": "SI=F",
        "Brent": "BZ=F",
        "Crude Oil": "CL=F",
        
        # Olymp Trade Exclusive Synthetic
        "Asia Composite Index": "^N225",
        "Europe Composite Index": "VGK",
        "Commodity Composite Index": "DBC",
        "Maha Jantar Index": "^NSEI",
        "Arabian General Index": "^TASI",
        "Halal Axis": "SPUS",
        "Halal Market Axis": "SPUS",
        "Latam Index": "ILF"
    }
    
    clean_asset = active_asset.replace(" OTC", "").strip()
    ticker_name = symbol_map.get(clean_asset, None)
    
    if not ticker_name:
        # Secondary fuzzy match
        if "BTC" in clean_asset: ticker_name = "BTC-USD"
        elif "ETH" in clean_asset: ticker_name = "ETH-USD"
        elif "EURUSD" in clean_asset: ticker_name = "EURUSD=X"
        elif "GBPUSD" in clean_asset: ticker_name = "GBPUSD=X"
        elif "Halal" in clean_asset: ticker_name = "SPUS"
        elif "Asia" in clean_asset: ticker_name = "^N225"

    if not ticker_name:
        print(f"\033[91m[Scout] FATAL: Tracker cannot resolve ticker for '{clean_asset}'.\033[0m")
        return None
    
    if trading_mode == 'FIXED_TIME':
        print(f"\033[96m[Scout] Browser Detection: '{active_asset}' matched to {ticker_name}.\033[0m")
    else:
        print(f"\033[92m[Data] Forex Mode: Fetching {ticker_name} automatically.\033[0m")
    
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
        
    except Exception as e:
        print(f"\033[91m[Data] yfinance fetch failed: {e}\033[0m")
        return None


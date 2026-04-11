"""
Data Loader — Elite v3.0
==========================
All data sourced DIRECTLY from MetaTrader 5 (NOT yfinance).
Ensures execution price == analysis price.

Features:
- Multi-Timeframe fetch (M5, M15, H1)
- Server time sync
- Edge case handling (market closed, symbol unavailable, zero volume)
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import datetime
import logging
from config import Config

logger = logging.getLogger("DataLoader")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(Config.LOG_FILE, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("\033[92m%(asctime)s\033[0m [%(levelname)s] %(message)s"))
    logger.addHandler(_ch)


def fetch_mt5_ohlc(symbol=None, timeframe=None, count=None):
    """
    Fetches OHLC data directly from MT5.

    Args:
        symbol: Trading symbol (default: Config.FOREX_SYMBOL)
        timeframe: MT5 timeframe constant (default: Config.TIMEFRAME)
        count: Number of candles to fetch (default: Config.DATA_POINTS)

    Returns:
        pd.DataFrame with columns: open, high, low, close, real_volume, spread
        Index is datetime (timezone-naive UTC).
        Returns None on failure.
    """
    symbol = symbol or Config.FOREX_SYMBOL
    timeframe = timeframe or Config.TIMEFRAME
    count = count or Config.DATA_POINTS

    # Ensure symbol is visible in Market Watch
    if not mt5.symbol_select(symbol, True):
        logger.error("[Data] Symbol '%s' not available in MT5.", symbol)
        return None

    # Check if market is open
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.error("[Data] Cannot get symbol info for '%s'.", symbol)
        return None

    # Fetch candles
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    if rates is None or len(rates) == 0:
        logger.error("[Data] MT5 returned empty data for %s (TF: %s). Market may be closed.",
                     symbol, _tf_name(timeframe))
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)

    # Convert time column from Unix timestamp to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Rename to standard format
    df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tick_volume': 'real_volume',
        'spread': 'spread',
        'real_volume': 'mt5_real_volume',
    }, inplace=True)

    # Keep only needed columns
    cols_to_keep = ['open', 'high', 'low', 'close', 'real_volume', 'spread']
    available_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[available_cols].copy()

    # Edge case: zero volume filter (optional — log but don't remove)
    zero_vol_count = (df['real_volume'] == 0).sum() if 'real_volume' in df.columns else 0
    if zero_vol_count > 0:
        logger.warning("[Data] %d candles with zero volume detected in %s.", zero_vol_count, symbol)

    logger.info("[Data] ✅ Fetched %d candles for %s (%s)",
                len(df), symbol, _tf_name(timeframe))
    return df


def fetch_mtf_data(symbol=None):
    """
    Fetches Multi-Timeframe data:
    - M5  (primary — for entry signals)
    - M15 (confirmation)
    - H1  (trend direction)

    Returns:
        dict with keys 'M5', 'M15', 'H1', each containing a DataFrame.
        Returns None if M5 (primary) fetch fails.
    """
    symbol = symbol or Config.FOREX_SYMBOL

    logger.info("[MTF] Fetching multi-timeframe data for %s...", symbol)

    # Primary timeframe — MUST succeed
    df_m5 = fetch_mt5_ohlc(symbol, Config.TIMEFRAME, Config.DATA_POINTS)
    if df_m5 is None or df_m5.empty:
        logger.error("[MTF] Primary timeframe (M5) fetch failed. Aborting.")
        return None

    # Confirmation timeframe
    df_m15 = fetch_mt5_ohlc(symbol, Config.TIMEFRAME_CONFIRM, Config.DATA_POINTS)
    if df_m15 is None or df_m15.empty:
        logger.warning("[MTF] Confirmation timeframe (M15) fetch failed. Proceeding without.")
        df_m15 = pd.DataFrame()

    # Trend timeframe
    df_h1 = fetch_mt5_ohlc(symbol, Config.TIMEFRAME_TREND, Config.DATA_POINTS)
    if df_h1 is None or df_h1.empty:
        logger.warning("[MTF] Trend timeframe (H1) fetch failed. Proceeding without.")
        df_h1 = pd.DataFrame()

    logger.info("[MTF] ✅ M5: %d | M15: %d | H1: %d candles",
                len(df_m5), len(df_m15), len(df_h1))

    return {
        "M5": df_m5,
        "M15": df_m15,
        "H1": df_h1,
    }


def fetch_tick_data(symbol=None):
    """
    Fetches the latest tick (bid/ask) for a symbol.

    Returns:
        dict with keys: bid, ask, spread, time, volume
        or None on failure.
    """
    symbol = symbol or Config.FOREX_SYMBOL
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.warning("[Tick] No tick data for %s.", symbol)
        return None

    info = mt5.symbol_info(symbol)
    point = info.point if info else 0.00001
    spread_points = (tick.ask - tick.bid) / point if point > 0 else 0

    return {
        "bid": tick.bid,
        "ask": tick.ask,
        "spread_points": spread_points,
        "time": datetime.datetime.utcfromtimestamp(tick.time),
        "volume": tick.volume,
        "last": tick.last,
    }


def get_server_time_from_tick(symbol=None):
    """
    Get server time from the latest tick.
    This is the CORRECT time source for session filtering.
    """
    symbol = symbol or Config.FOREX_SYMBOL
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        return datetime.datetime.utcfromtimestamp(tick.time)
    return None


def is_market_open(symbol=None):
    """
    Check if the market is currently open for the symbol.
    Uses MT5's trade mode flag.
    """
    symbol = symbol or Config.FOREX_SYMBOL
    info = mt5.symbol_info(symbol)
    if not info:
        return False

    # trade_mode: 0=disabled, 2=full
    if info.trade_mode == 0:
        logger.warning("[Market] %s trading is disabled.", symbol)
        return False

    # Also check session status via recent tick
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.warning("[Market] No tick for %s. Market may be closed.", symbol)
        return False

    # Check staleness — if last tick is older than 5 minutes, market is likely closed
    tick_time = datetime.datetime.utcfromtimestamp(tick.time)
    now = datetime.datetime.utcnow()
    age_seconds = (now - tick_time).total_seconds()

    if age_seconds > 300:
        logger.warning("[Market] Last tick for %s is %.0fs old. Market likely closed.", symbol, age_seconds)
        return False

    return True


def _tf_name(timeframe):
    """Human-readable name for MT5 timeframe constants."""
    names = {
        mt5.TIMEFRAME_M1: "M1",
        mt5.TIMEFRAME_M5: "M5",
        mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_M30: "M30",
        mt5.TIMEFRAME_H1: "H1",
        mt5.TIMEFRAME_H4: "H4",
        mt5.TIMEFRAME_D1: "D1",
        mt5.TIMEFRAME_W1: "W1",
        mt5.TIMEFRAME_MN1: "MN1",
    }
    return names.get(timeframe, str(timeframe))


# =========================================
# LEGACY COMPATIBILITY
# =========================================

def fetch_data(trading_mode='FOREX', manual_symbol=None):
    """
    Legacy-compatible wrapper.
    Now routes everything through MT5.
    """
    symbol = manual_symbol or Config.FOREX_SYMBOL

    if trading_mode == 'FOREX':
        return fetch_mt5_ohlc(symbol)
    else:
        # Fixed Time mode still supported but also uses MT5 data
        logger.info("[Legacy] Fixed Time mode — using MT5 data for %s", symbol)
        return fetch_mt5_ohlc(symbol)


def init_mt5():
    """Legacy stub — connection is now handled by mt5_engine.connect_to_exness()."""
    logger.info("[Data Loader] MT5 Native Mode Active.")
    return True

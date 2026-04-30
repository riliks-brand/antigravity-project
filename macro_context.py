"""
Macro Context Analyzer — Phase 3
==================================
Provides macro-level context (DXY Dollar Strength) to the ensemble engine.
Uses strength-based logic (momentum) rather than hard directional blockers.
If native DXY is unavailable from the broker, it computes a Synthetic DXY
basket from EURUSD, USDJPY, and GBPUSD.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
from config import Config

logger = logging.getLogger("MacroContext")

def get_mt5_data(symbol, timeframe, count=50):
    """Fetch MT5 data cleanly."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def calculate_trend_strength(df):
    """
    Calculates a continuous strength score [-1.0 to 1.0] based on EMA slopes and RSI.
    """
    if df is None or len(df) < 30:
        return 0.0

    # Calculate EMAs
    close = df['close']
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    # EMA Slope over last 5 candles
    slope20 = (ema20.iloc[-1] - ema20.iloc[-5]) / ema20.iloc[-5] * 10000 # scaling factor
    slope50 = (ema50.iloc[-1] - ema50.iloc[-5]) / ema50.iloc[-5] * 10000

    # RSI 14
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    # Normalize RSI to [-1.0, 1.0] where 50 is 0
    rsi_norm = (current_rsi - 50) / 50.0

    # Combine slope and RSI for momentum score
    # Tanh used to bound the score between -1 and 1 smoothly
    momentum = np.tanh(slope20 + slope50 * 0.5 + rsi_norm * 1.5)
    
    return float(momentum)

def compute_synthetic_dxy(timeframe=mt5.TIMEFRAME_H1, count=50):
    """
    Computes a synthetic DXY series based on the major USD pairs.
    DXY Formula approx = 50.14348112 × EURUSD^-0.576 × USDJPY^0.136 × GBPUSD^-0.119
    """
    eurusd = get_mt5_data("EURUSD", timeframe, count)
    usdjpy = get_mt5_data("USDJPY", timeframe, count)
    gbpusd = get_mt5_data("GBPUSD", timeframe, count)

    if eurusd is None or usdjpy is None or gbpusd is None:
        logger.warning("[Macro] Failed to fetch data for synthetic DXY.")
        return None

    # Align indexes just in case
    df = pd.concat([eurusd['close'].rename('EURUSD'), 
                    usdjpy['close'].rename('USDJPY'), 
                    gbpusd['close'].rename('GBPUSD')], axis=1).dropna()

    if len(df) < 30:
        return None

    # Compute synthetic index
    # We apply the power weights exactly as the ICE DXY formula
    df['synthetic_dxy'] = 50.14348112 * (df['EURUSD'] ** -0.576) * (df['USDJPY'] ** 0.136) * (df['GBPUSD'] ** -0.119)
    
    # Format to look like standard OHLC df for strength calc
    synth_df = pd.DataFrame({'close': df['synthetic_dxy']})
    return synth_df

def get_dxy_strength():
    """
    Master function to get the current Dollar Strength (-1.0 to 1.0).
    -1.0 = Extreme Bearish USD
    +1.0 = Extreme Bullish USD
    0.0  = Neutral / Range
    """
    dxy_ticker = getattr(Config, 'DXY_TICKER', "DX-Y.NYB")
    timeframe = mt5.TIMEFRAME_H1  # Evaluate macro trend on H1
    
    # 1. Try fetching native DXY
    df = get_mt5_data(dxy_ticker, timeframe, count=60)
    
    # 2. Fallback to Synthetic DXY
    if df is None:
        logger.debug(f"[Macro] Native {dxy_ticker} not found or no data. Using Synthetic DXY.")
        df = compute_synthetic_dxy(timeframe, count=60)

    if df is None:
        logger.error("[Macro] Could not compute DXY strength (no data).")
        return 0.0

    # 3. Calculate Strength Score
    strength = calculate_trend_strength(df)
    
    if abs(strength) > 0.6:
        logger.info(f"[Macro] DXY Strength: {strength:+.2f} (STRONG {'BULL' if strength > 0 else 'BEAR'})")
    
    return strength

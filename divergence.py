"""
Divergence Detector — Phase 1 (PDF Vision Layer)
===================================================
Detects RSI-based divergences (bullish and bearish):
  - Regular Bullish Divergence: Price makes Lower Low, RSI makes Higher Low
  - Regular Bearish Divergence: Price makes Higher High, RSI makes Lower High
  - Hidden Bullish Divergence: Price makes Higher Low, RSI makes Lower Low (trend continuation)
  - Hidden Bearish Divergence: Price makes Lower High, RSI makes Higher High (trend continuation)

Divergences are one of the most reliable leading reversal signals
described in the PDF and referenced in the Executive Summary.
"""

import numpy as np
import pandas as pd
import logging
from scipy.signal import argrelextrema

logger = logging.getLogger("Divergence")


def _find_peaks(series: np.ndarray, order: int = 5) -> np.ndarray:
    """Find indices of local maxima."""
    return argrelextrema(series, np.greater_equal, order=order)[0]


def _find_valleys(series: np.ndarray, order: int = 5) -> np.ndarray:
    """Find indices of local minima."""
    return argrelextrema(series, np.less_equal, order=order)[0]


# =========================================
# MAIN API: add_divergence_features()
# =========================================

def add_divergence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects RSI divergences and adds them as features.
    Requires: high, low, close, RSI columns.

    New columns:
        RSI_BullDiv       - Regular bullish divergence (binary)
        RSI_BearDiv       - Regular bearish divergence (binary)
        RSI_HiddenBullDiv - Hidden bullish divergence (binary)
        RSI_HiddenBearDiv - Hidden bearish divergence (binary)
        divergence_score  - Composite: +1 for bull div, -1 for bear div (continuous)

    Returns:
        DataFrame with divergence columns added.
    """
    n = len(df)
    highs = df['high'].values
    lows = df['low'].values
    rsi = df['RSI'].values

    # Output arrays
    bull_div = np.zeros(n, dtype=int)
    bear_div = np.zeros(n, dtype=int)
    hidden_bull = np.zeros(n, dtype=int)
    hidden_bear = np.zeros(n, dtype=int)

    # Parameters
    order = 5           # Swing detection sensitivity
    lookback = 30       # Window to search for divergence pairs
    min_gap = 5         # Minimum bars between swing points

    # Find swing points on price
    price_peaks = _find_peaks(highs, order=order)
    price_valleys = _find_valleys(lows, order=order)

    # Find swing points on RSI
    rsi_clean = np.nan_to_num(rsi, nan=50.0)
    rsi_peaks = _find_peaks(rsi_clean, order=order)
    rsi_valleys = _find_valleys(rsi_clean, order=order)

    # =========================================
    # REGULAR BEARISH DIVERGENCE
    # Price: Higher High, RSI: Lower High
    # =========================================
    for i in range(1, len(price_peaks)):
        idx_curr = price_peaks[i]
        idx_prev = price_peaks[i - 1]

        if idx_curr - idx_prev < min_gap or idx_curr - idx_prev > lookback:
            continue

        # Price makes Higher High
        if highs[idx_curr] <= highs[idx_prev]:
            continue

        # Find RSI peaks near these price peaks
        rsi_near_curr = rsi_peaks[(rsi_peaks >= idx_curr - 2) & (rsi_peaks <= idx_curr + 2)]
        rsi_near_prev = rsi_peaks[(rsi_peaks >= idx_prev - 2) & (rsi_peaks <= idx_prev + 2)]

        if len(rsi_near_curr) == 0 or len(rsi_near_prev) == 0:
            continue

        rsi_val_curr = rsi_clean[rsi_near_curr[0]]
        rsi_val_prev = rsi_clean[rsi_near_prev[0]]

        # RSI makes Lower High
        if rsi_val_curr < rsi_val_prev:
            bear_div[idx_curr] = 1

    # =========================================
    # REGULAR BULLISH DIVERGENCE
    # Price: Lower Low, RSI: Higher Low
    # =========================================
    for i in range(1, len(price_valleys)):
        idx_curr = price_valleys[i]
        idx_prev = price_valleys[i - 1]

        if idx_curr - idx_prev < min_gap or idx_curr - idx_prev > lookback:
            continue

        # Price makes Lower Low
        if lows[idx_curr] >= lows[idx_prev]:
            continue

        # Find RSI valleys near these price valleys
        rsi_near_curr = rsi_valleys[(rsi_valleys >= idx_curr - 2) & (rsi_valleys <= idx_curr + 2)]
        rsi_near_prev = rsi_valleys[(rsi_valleys >= idx_prev - 2) & (rsi_valleys <= idx_prev + 2)]

        if len(rsi_near_curr) == 0 or len(rsi_near_prev) == 0:
            continue

        rsi_val_curr = rsi_clean[rsi_near_curr[0]]
        rsi_val_prev = rsi_clean[rsi_near_prev[0]]

        # RSI makes Higher Low
        if rsi_val_curr > rsi_val_prev:
            bull_div[idx_curr] = 1

    # =========================================
    # HIDDEN BULLISH DIVERGENCE (Trend continuation)
    # Price: Higher Low, RSI: Lower Low
    # =========================================
    for i in range(1, len(price_valleys)):
        idx_curr = price_valleys[i]
        idx_prev = price_valleys[i - 1]

        if idx_curr - idx_prev < min_gap or idx_curr - idx_prev > lookback:
            continue

        # Price makes Higher Low
        if lows[idx_curr] <= lows[idx_prev]:
            continue

        rsi_near_curr = rsi_valleys[(rsi_valleys >= idx_curr - 2) & (rsi_valleys <= idx_curr + 2)]
        rsi_near_prev = rsi_valleys[(rsi_valleys >= idx_prev - 2) & (rsi_valleys <= idx_prev + 2)]

        if len(rsi_near_curr) == 0 or len(rsi_near_prev) == 0:
            continue

        rsi_val_curr = rsi_clean[rsi_near_curr[0]]
        rsi_val_prev = rsi_clean[rsi_near_prev[0]]

        # RSI makes Lower Low
        if rsi_val_curr < rsi_val_prev:
            hidden_bull[idx_curr] = 1

    # =========================================
    # HIDDEN BEARISH DIVERGENCE (Trend continuation)
    # Price: Lower High, RSI: Higher High
    # =========================================
    for i in range(1, len(price_peaks)):
        idx_curr = price_peaks[i]
        idx_prev = price_peaks[i - 1]

        if idx_curr - idx_prev < min_gap or idx_curr - idx_prev > lookback:
            continue

        # Price makes Lower High
        if highs[idx_curr] >= highs[idx_prev]:
            continue

        rsi_near_curr = rsi_peaks[(rsi_peaks >= idx_curr - 2) & (rsi_peaks <= idx_curr + 2)]
        rsi_near_prev = rsi_peaks[(rsi_peaks >= idx_prev - 2) & (rsi_peaks <= idx_prev + 2)]

        if len(rsi_near_curr) == 0 or len(rsi_near_prev) == 0:
            continue

        rsi_val_curr = rsi_clean[rsi_near_curr[0]]
        rsi_val_prev = rsi_clean[rsi_near_prev[0]]

        # RSI makes Higher High
        if rsi_val_curr > rsi_val_prev:
            hidden_bear[idx_curr] = 1

    # =========================================
    # Assign to DataFrame
    # =========================================
    df['RSI_BullDiv'] = bull_div
    df['RSI_BearDiv'] = bear_div
    df['RSI_HiddenBullDiv'] = hidden_bull
    df['RSI_HiddenBearDiv'] = hidden_bear

    # Composite divergence score
    df['divergence_score'] = (
        (df['RSI_BullDiv'] + df['RSI_HiddenBullDiv']) -
        (df['RSI_BearDiv'] + df['RSI_HiddenBearDiv'])
    ).clip(-2, 2)

    total = bull_div.sum() + bear_div.sum() + hidden_bull.sum() + hidden_bear.sum()
    logger.info(
        "[Divergence] Added 4 divergence flags + composite score. "
        "Total divergences detected: %d (Bull=%d, Bear=%d, HBull=%d, HBear=%d)",
        total, bull_div.sum(), bear_div.sum(), hidden_bull.sum(), hidden_bear.sum()
    )
    return df

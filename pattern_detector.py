"""
Chart Pattern Detector — Phase 1 (PDF Vision Layer)
=====================================================
Detects multi-bar structural chart patterns from the PDF:
  - Double Top / Double Bottom
  - Triple Top / Triple Bottom
  - Head & Shoulders / Inverse H&S
  - Ascending / Descending / Symmetrical Triangle
  - Rising / Falling Wedge
  - Bull / Bear Flag
  - Volatility Compression (Bollinger Squeeze)

All patterns use swing-point detection and linear regression
on rolling windows, producing binary flags and continuous measures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from scipy.signal import argrelextrema

logger = logging.getLogger("PatternDetector")

# =========================================
# MODULE-LEVEL CACHE AND CONSTANTS (Fix 1)
# =========================================

# Per-symbol cache dict — persists across calls within the same process.
# Structure: { symbol: { "last_row_count": int, "last_index": Any, "columns": dict[str, np.ndarray] } }
_pattern_cache: dict[str, dict] = {}

# All 15 output columns produced by add_chart_patterns()
OUTPUT_COLUMNS = [
    "DoubleTop_Flag", "DoubleBottom_Flag", "TripleTop_Flag", "TripleBottom_Flag",
    "HS_Flag", "InvHS_Flag", "AscTriangle_Flag", "DescTriangle_Flag",
    "SymTriangle_Flag", "RisingWedge_Flag", "FallingWedge_Flag",
    "BullFlag_Flag", "BearFlag_Flag", "Volatility_Compress", "pattern_bias_score"
]

# Maximum number of rows to reprocess on an incremental cache update
CACHE_LOOKBACK = 1000


# =========================================
# SWING POINT DETECTION (Core utility)
# =========================================

def _find_swing_highs(highs: np.ndarray, order: int = 5) -> np.ndarray:
    """Find indices of local maxima (swing highs) using argrelextrema."""
    indices = argrelextrema(highs, np.greater_equal, order=order)[0]
    return indices


def _find_swing_lows(lows: np.ndarray, order: int = 5) -> np.ndarray:
    """Find indices of local minima (swing lows) using argrelextrema."""
    indices = argrelextrema(lows, np.less_equal, order=order)[0]
    return indices


def _linear_slope(values: np.ndarray) -> float:
    """Returns the slope of a simple linear regression through `values`."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    A = np.vstack([x, np.ones(len(x))]).T
    try:
        m, _ = np.linalg.lstsq(A, values, rcond=None)[0]
        return float(m)
    except Exception:
        return 0.0


# =========================================
# MAIN API: add_chart_patterns()
# =========================================

def _run_rolling_window_logic(df: pd.DataFrame, start_offset: int = 0) -> dict:
    """
    Core rolling-window pattern detection logic.

    Runs the pattern detection loop over df, starting from index `start_offset`
    (relative to df). Returns a dict mapping each output column name to its
    computed numpy array (length == len(df)).

    When called for an incremental update, `df` is the slice `full_df[start_idx:]`
    and `start_offset` is 0 (we always process the full slice).
    """
    epsilon = 1e-8
    n = len(df)
    atr = df['ATR'].values + epsilon
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    # Pre-allocate output arrays
    double_top = np.zeros(n, dtype=int)
    double_bottom = np.zeros(n, dtype=int)
    triple_top = np.zeros(n, dtype=int)
    triple_bottom = np.zeros(n, dtype=int)
    hs_flag = np.zeros(n, dtype=int)
    inv_hs_flag = np.zeros(n, dtype=int)
    asc_triangle = np.zeros(n, dtype=int)
    desc_triangle = np.zeros(n, dtype=int)
    sym_triangle = np.zeros(n, dtype=int)
    rising_wedge = np.zeros(n, dtype=int)
    falling_wedge = np.zeros(n, dtype=int)
    bull_flag = np.zeros(n, dtype=int)
    bear_flag = np.zeros(n, dtype=int)

    # =========================================
    # Rolling window analysis
    # We look back over windows of different sizes
    # =========================================
    lookback = 30     # Main pattern window
    order = 3         # Swing point detection order

    for i in range(lookback, n):
        window_highs = highs[i - lookback:i]
        window_lows = lows[i - lookback:i]
        window_closes = closes[i - lookback:i]
        current_atr = atr[i]

        # Find swing points within the window
        swing_high_idx = _find_swing_highs(window_highs, order=order)
        swing_low_idx = _find_swing_lows(window_lows, order=order)

        swing_high_vals = window_highs[swing_high_idx] if len(swing_high_idx) > 0 else np.array([])
        swing_low_vals = window_lows[swing_low_idx] if len(swing_low_idx) > 0 else np.array([])

        # --- DOUBLE TOP: 2 highs within tolerance ---
        if len(swing_high_vals) >= 2:
            top_tolerance = current_atr * 0.5
            last_two = swing_high_vals[-2:]
            if abs(last_two[0] - last_two[1]) < top_tolerance:
                double_top[i] = 1

        # --- DOUBLE BOTTOM: 2 lows within tolerance ---
        if len(swing_low_vals) >= 2:
            bot_tolerance = current_atr * 0.5
            last_two = swing_low_vals[-2:]
            if abs(last_two[0] - last_two[1]) < bot_tolerance:
                double_bottom[i] = 1

        # --- TRIPLE TOP: 3 highs within tolerance ---
        if len(swing_high_vals) >= 3:
            top_tolerance = current_atr * 0.6
            last_three = swing_high_vals[-3:]
            spread = max(last_three) - min(last_three)
            if spread < top_tolerance:
                triple_top[i] = 1

        # --- TRIPLE BOTTOM: 3 lows within tolerance ---
        if len(swing_low_vals) >= 3:
            bot_tolerance = current_atr * 0.6
            last_three = swing_low_vals[-3:]
            spread = max(last_three) - min(last_three)
            if spread < bot_tolerance:
                triple_bottom[i] = 1

        # --- HEAD & SHOULDERS: middle peak higher than neighbors ---
        if len(swing_high_vals) >= 3 and len(swing_high_idx) >= 3:
            h1, head, h2 = swing_high_vals[-3], swing_high_vals[-2], swing_high_vals[-1]
            shoulder_tol = current_atr * 0.5
            if (head > h1 and head > h2 and
                    abs(h1 - h2) < shoulder_tol and
                    (head - max(h1, h2)) > current_atr * 0.3):
                hs_flag[i] = 1

        # --- INVERSE HEAD & SHOULDERS: middle valley lower than neighbors ---
        if len(swing_low_vals) >= 3 and len(swing_low_idx) >= 3:
            l1, head_low, l2 = swing_low_vals[-3], swing_low_vals[-2], swing_low_vals[-1]
            shoulder_tol = current_atr * 0.5
            if (head_low < l1 and head_low < l2 and
                    abs(l1 - l2) < shoulder_tol and
                    (min(l1, l2) - head_low) > current_atr * 0.3):
                inv_hs_flag[i] = 1

        # --- TRIANGLES (using slopes of swing highs and lows) ---
        if len(swing_high_vals) >= 3 and len(swing_low_vals) >= 3:
            high_slope = _linear_slope(swing_high_vals[-3:])
            low_slope = _linear_slope(swing_low_vals[-3:])

            # Normalize slopes by ATR for cross-instrument consistency
            high_slope_norm = high_slope / current_atr
            low_slope_norm = low_slope / current_atr

            # Ascending: flat highs + rising lows
            if abs(high_slope_norm) < 0.3 and low_slope_norm > 0.2:
                asc_triangle[i] = 1

            # Descending: falling highs + flat lows
            if high_slope_norm < -0.2 and abs(low_slope_norm) < 0.3:
                desc_triangle[i] = 1

            # Symmetrical: falling highs + rising lows (converging)
            if high_slope_norm < -0.15 and low_slope_norm > 0.15:
                sym_triangle[i] = 1

            # Rising Wedge: both rising, but highs slope < lows slope (converging up)
            if (high_slope_norm > 0.1 and low_slope_norm > 0.1 and
                    high_slope_norm < low_slope_norm):
                rising_wedge[i] = 1

            # Falling Wedge: both falling, but lows slope > highs slope (converging down)
            if (high_slope_norm < -0.1 and low_slope_norm < -0.1 and
                    low_slope_norm > high_slope_norm):
                falling_wedge[i] = 1

        # --- FLAG PATTERNS (sharp move + consolidation) ---
        if i >= 15:
            # Bull Flag: strong rally in [i-15:i-8], then tight range in [i-8:i]
            rally_move = closes[i - 8] - closes[i - 15]
            consol_range = max(highs[i - 8:i]) - min(lows[i - 8:i])
            if rally_move > 2 * current_atr and consol_range < 1.0 * current_atr:
                bull_flag[i] = 1

            # Bear Flag: strong drop in [i-15:i-8], then tight range
            drop_move = closes[i - 15] - closes[i - 8]
            if drop_move > 2 * current_atr and consol_range < 1.0 * current_atr:
                bear_flag[i] = 1

    # =========================================
    # VOLATILITY COMPRESSION (Bollinger Squeeze — vectorized)
    # ATR at 20th percentile of rolling window → squeeze imminent
    # =========================================
    atr_series = df['ATR']
    atr_rolling_pctile = atr_series.rolling(50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    vol_compress_arr = np.where(atr_rolling_pctile < 0.2, 1, 0)

    return {
        "DoubleTop_Flag": double_top,
        "DoubleBottom_Flag": double_bottom,
        "TripleTop_Flag": triple_top,
        "TripleBottom_Flag": triple_bottom,
        "HS_Flag": hs_flag,
        "InvHS_Flag": inv_hs_flag,
        "AscTriangle_Flag": asc_triangle,
        "DescTriangle_Flag": desc_triangle,
        "SymTriangle_Flag": sym_triangle,
        "RisingWedge_Flag": rising_wedge,
        "FallingWedge_Flag": falling_wedge,
        "BullFlag_Flag": bull_flag,
        "BearFlag_Flag": bear_flag,
        "Volatility_Compress": vol_compress_arr,
    }


def _assign_arrays_to_df(df: pd.DataFrame, arrays: dict) -> pd.DataFrame:
    """
    Assigns the pattern arrays to df and computes the composite pattern_bias_score.
    `arrays` must contain all 14 flag columns (keys matching OUTPUT_COLUMNS[:-1]).
    """
    for col, arr in arrays.items():
        df[col] = arr

    # =========================================
    # COMPOSITE: Pattern Bias Score (continuous)
    # Positive = bullish patterns active, Negative = bearish
    # =========================================
    bull_patterns = (
        df['DoubleBottom_Flag'] + df['TripleBottom_Flag'] + df['InvHS_Flag'] +
        df['AscTriangle_Flag'] + df['FallingWedge_Flag'] + df['BullFlag_Flag']
    )
    bear_patterns = (
        df['DoubleTop_Flag'] + df['TripleTop_Flag'] + df['HS_Flag'] +
        df['DescTriangle_Flag'] + df['RisingWedge_Flag'] + df['BearFlag_Flag']
    )
    df['pattern_bias_score'] = (bull_patterns - bear_patterns).clip(-3, 3)
    return df


def add_chart_patterns(df: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
    """
    Adds chart pattern flags to the DataFrame using rolling window analysis.
    Requires: open, high, low, close, ATR columns.

    Uses a per-symbol in-memory cache to avoid reprocessing unchanged rows:
      - Cache hit (same row count + same last index): returns cached columns immediately.
      - Incremental update (new rows appended): recomputes only the last CACHE_LOOKBACK rows.
      - Full computation (no cache or row count shrank): runs the full rolling-window logic.

    Args:
        df:     Primary OHLC DataFrame with an ATR column.
        symbol: Cache key — pass the trading symbol name (e.g. "XAUUSD").

    Returns:
        DataFrame with new binary and continuous columns for each pattern.
    """
    global _pattern_cache
    n = len(df)
    cached = _pattern_cache.get(symbol)

    # =========================================
    # Branch 1 — Cache hit: same row count AND same last index
    # =========================================
    if (cached is not None
            and cached["last_row_count"] == n
            and df.index[-1] == cached["last_index"]):
        logger.info(
            "[Patterns][%s] Cache hit — %d rows, 0 new rows processed.", symbol, n
        )
        for col, arr in cached["columns"].items():
            df[col] = arr
        return df

    # =========================================
    # Branch 2 — Incremental update: new rows appended
    # (also handles the case where row count is equal but last_index differs —
    #  that falls through to Branch 3 / full computation below)
    # =========================================
    if cached is not None and n > cached["last_row_count"]:
        new_rows = n - cached["last_row_count"]
        start_idx = max(0, n - CACHE_LOOKBACK)

        # Run the existing rolling-window logic on the slice only
        df_slice = df.iloc[start_idx:].copy()
        slice_arrays = _run_rolling_window_logic(df_slice)

        # Build full-length arrays: use cached values for rows before start_idx,
        # then overwrite with freshly computed values from start_idx onward.
        merged_arrays = {}
        for col in OUTPUT_COLUMNS[:-1]:  # all flag columns except pattern_bias_score
            full_arr = cached["columns"][col].copy() if col in cached["columns"] else np.zeros(n, dtype=int)
            # Resize cached array if it is shorter than start_idx (safety guard)
            if len(full_arr) < start_idx:
                padded = np.zeros(n, dtype=full_arr.dtype)
                padded[:len(full_arr)] = full_arr
                full_arr = padded
            # Combine: keep cached prefix, overwrite suffix with fresh computation
            combined = np.empty(n, dtype=full_arr.dtype)
            combined[:start_idx] = full_arr[:start_idx]
            combined[start_idx:] = slice_arrays[col]
            merged_arrays[col] = combined

        # Assign merged arrays to df and compute composite score
        df = _assign_arrays_to_df(df, merged_arrays)

        logger.info(
            "[Patterns][%s] Incremental — %d new rows processed (lookback=%d).",
            symbol, new_rows, n - start_idx,
        )

    else:
        # =========================================
        # Branch 3 — Full computation:
        #   • First call for this symbol (no cache), OR
        #   • Row count shrank (data reload / reset), OR
        #   • Same row count but last_index differs (data replaced — cache invalidation)
        # =========================================
        logger.info("[Patterns][%s] Full computation — %d rows.", symbol, n)

        full_arrays = _run_rolling_window_logic(df)
        df = _assign_arrays_to_df(df, full_arrays)

    # =========================================
    # Store result in cache (after any computation path)
    # =========================================
    _pattern_cache[symbol] = {
        "last_row_count": n,
        "last_index": df.index[-1],
        "columns": {col: df[col].values.copy() for col in OUTPUT_COLUMNS},
    }

    logger.info(
        "[Patterns] Added 14 chart pattern flags + composite bias score + volatility compression."
    )
    return df

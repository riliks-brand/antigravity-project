"""
Candlestick Pattern Detector — Phase 1 (PDF Vision Layer)
===========================================================
Detects single-candle and multi-candle patterns from the PDF:
  - Engulfing (Bullish/Bearish)
  - Hammer / Inverted Hammer
  - Morning Star / Evening Star
  - Doji (Dragonfly / Gravestone / Neutral)
  - Piercing Line / Dark Cloud Cover
  - Three White Soldiers / Three Black Crows

All patterns output binary flags (0/1) and are ATR-normalized
to work across any instrument and timeframe.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("Candles")


def _body(df):
    """Signed body: positive = bullish, negative = bearish."""
    return df['close'] - df['open']


def _body_abs(df):
    """Absolute body size."""
    return abs(df['close'] - df['open'])


def _upper_wick(df):
    return df['high'] - df[['open', 'close']].max(axis=1)


def _lower_wick(df):
    return df[['open', 'close']].min(axis=1) - df['low']


def _candle_range(df):
    return df['high'] - df['low']


def _is_bullish(df):
    return df['close'] > df['open']


def _is_bearish(df):
    return df['close'] < df['open']


# =========================================
# MAIN API: add_candlestick_patterns()
# =========================================

def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all candlestick pattern flags to the DataFrame.
    Requires: open, high, low, close, ATR columns.

    Returns:
        DataFrame with new binary columns for each pattern.
    """
    epsilon = 1e-8
    atr = df['ATR'] + epsilon
    body = _body(df)
    body_abs = _body_abs(df)
    upper = _upper_wick(df)
    lower = _lower_wick(df)
    cr = _candle_range(df) + epsilon
    bullish = _is_bullish(df)
    bearish = _is_bearish(df)

    # Previous candle data (shifted by 1)
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    prev_body = (prev_close - prev_open)
    prev_body_abs = abs(prev_body)
    prev_bullish = prev_close > prev_open
    prev_bearish = prev_close < prev_open

    # Two candles ago
    prev2_open = df['open'].shift(2)
    prev2_close = df['close'].shift(2)
    prev2_body = prev2_close - prev2_open
    prev2_body_abs = abs(prev2_body)
    prev2_bullish = prev2_close > prev2_open
    prev2_bearish = prev2_close < prev2_open

    # Body size relative to ATR (used for "significant" checks)
    body_atr_ratio = body_abs / atr
    prev_body_atr_ratio = prev_body_abs / atr

    # =========================================
    # 1. BULLISH ENGULFING
    # Current bullish candle fully engulfs previous bearish candle body
    # =========================================
    df['Engulf_Bull'] = np.where(
        bullish & prev_bearish &
        (df['open'] <= prev_close) &
        (df['close'] >= prev_open) &
        (body_atr_ratio > 0.3),  # Must be a meaningful candle
        1, 0
    ).astype(int)

    # =========================================
    # 2. BEARISH ENGULFING
    # Current bearish candle fully engulfs previous bullish candle body
    # =========================================
    df['Engulf_Bear'] = np.where(
        bearish & prev_bullish &
        (df['open'] >= prev_close) &
        (df['close'] <= prev_open) &
        (body_atr_ratio > 0.3),
        1, 0
    ).astype(int)

    # =========================================
    # 3. HAMMER (Bullish reversal at bottom)
    # Small body at top, long lower shadow >= 2x body, small upper shadow
    # =========================================
    df['Hammer_Bull'] = np.where(
        (lower >= 2 * body_abs) &
        (upper < body_abs * 0.5) &
        (body_atr_ratio > 0.05) &
        (body_atr_ratio < 0.5),  # Not too large
        1, 0
    ).astype(int)

    # =========================================
    # 4. INVERTED HAMMER / SHOOTING STAR (Bearish reversal at top)
    # Small body at bottom, long upper shadow >= 2x body, small lower shadow
    # =========================================
    df['Hammer_Bear'] = np.where(
        (upper >= 2 * body_abs) &
        (lower < body_abs * 0.5) &
        (body_atr_ratio > 0.05) &
        (body_atr_ratio < 0.5),
        1, 0
    ).astype(int)

    # =========================================
    # 5. DRAGONFLY DOJI (Bullish: open≈close at top, long lower shadow)
    # =========================================
    df['DragonflyDoji'] = np.where(
        (body_abs / cr < 0.05) &  # Very small body
        (lower >= 0.7 * cr) &     # Lower shadow dominates
        (upper < 0.1 * cr),       # Tiny upper shadow
        1, 0
    ).astype(int)

    # =========================================
    # 6. GRAVESTONE DOJI (Bearish: open≈close at bottom, long upper shadow)
    # =========================================
    df['GravestoneDoji'] = np.where(
        (body_abs / cr < 0.05) &
        (upper >= 0.7 * cr) &
        (lower < 0.1 * cr),
        1, 0
    ).astype(int)

    # =========================================
    # 7. NEUTRAL DOJI (Indecision: open≈close, shadows both sides)
    # =========================================
    df['NeutralDoji'] = np.where(
        (body_abs / cr < 0.05) &
        ~(df['DragonflyDoji'].astype(bool)) &
        ~(df['GravestoneDoji'].astype(bool)),
        1, 0
    ).astype(int)

    # =========================================
    # 8. MORNING STAR (3-candle bullish reversal)
    # Candle[-2]: Big bearish → Candle[-1]: Small body (star) → Candle[0]: Big bullish
    # =========================================
    star_body = prev_body_abs  # The "star" is the middle candle
    star_small = star_body < (0.3 * atr)  # Star has a tiny body

    df['MorningStar'] = np.where(
        prev2_bearish &                           # 1st: big bearish
        (prev2_body_abs > 0.5 * atr) &
        star_small &                               # 2nd: small star
        bullish &                                  # 3rd: big bullish
        (body_abs > 0.5 * atr) &
        (df['close'] > (prev2_open + prev2_close) / 2),  # Closes above midpoint of 1st
        1, 0
    ).astype(int)

    # =========================================
    # 9. EVENING STAR (3-candle bearish reversal)
    # Candle[-2]: Big bullish → Candle[-1]: Small body (star) → Candle[0]: Big bearish
    # =========================================
    df['EveningStar'] = np.where(
        prev2_bullish &
        (prev2_body_abs > 0.5 * atr) &
        star_small &
        bearish &
        (body_abs > 0.5 * atr) &
        (df['close'] < (prev2_open + prev2_close) / 2),
        1, 0
    ).astype(int)

    # =========================================
    # 10. PIERCING LINE (2-candle bullish reversal)
    # Previous bearish, current opens below prev low, closes above prev midpoint
    # =========================================
    prev_mid = (prev_open + prev_close) / 2
    df['PiercingLine'] = np.where(
        prev_bearish &
        bullish &
        (df['open'] < prev_low) &
        (df['close'] > prev_mid) &
        (df['close'] < prev_open),  # Doesn't fully engulf
        1, 0
    ).astype(int)

    # =========================================
    # 11. DARK CLOUD COVER (2-candle bearish reversal)
    # Previous bullish, current opens above prev high, closes below prev midpoint
    # =========================================
    df['DarkCloudCover'] = np.where(
        prev_bullish &
        bearish &
        (df['open'] > prev_high) &
        (df['close'] < prev_mid) &
        (df['close'] > prev_open),
        1, 0
    ).astype(int)

    # =========================================
    # 12. THREE WHITE SOLDIERS (3 consecutive bullish, strong bodies)
    # =========================================
    df['ThreeSoldiers'] = np.where(
        bullish & prev_bullish & prev2_bullish &
        (body_abs > 0.3 * atr) &
        (prev_body_abs > 0.3 * atr) &
        (prev2_body_abs > 0.3 * atr) &
        (df['close'] > prev_close) &
        (prev_close > prev2_close),
        1, 0
    ).astype(int)

    # =========================================
    # 13. THREE BLACK CROWS (3 consecutive bearish, strong bodies)
    # =========================================
    df['ThreeCrows'] = np.where(
        bearish & prev_bearish & prev2_bearish &
        (body_abs > 0.3 * atr) &
        (prev_body_abs > 0.3 * atr) &
        (prev2_body_abs > 0.3 * atr) &
        (df['close'] < prev_close) &
        (prev_close < prev2_close),
        1, 0
    ).astype(int)

    # =========================================
    # 14. SPINNING TOP (Small body, long shadows both sides — indecision)
    # =========================================
    df['SpinningTop'] = np.where(
        (body_abs / cr < 0.3) &
        (upper > body_abs * 0.5) &
        (lower > body_abs * 0.5) &
        (body_atr_ratio > 0.02),  # Not a doji
        1, 0
    ).astype(int)

    # =========================================
    # 15. MARUBOZU BULL (No shadows, strong bullish)
    # =========================================
    df['MarubozuBull'] = np.where(
        bullish &
        (upper < 0.05 * cr) &
        (lower < 0.05 * cr) &
        (body_atr_ratio > 0.5),
        1, 0
    ).astype(int)

    # =========================================
    # 16. MARUBOZU BEAR (No shadows, strong bearish)
    # =========================================
    df['MarubozuBear'] = np.where(
        bearish &
        (upper < 0.05 * cr) &
        (lower < 0.05 * cr) &
        (body_atr_ratio > 0.5),
        1, 0
    ).astype(int)

    # =========================================
    # COMPOSITE: Reversal Signal Strength (continuous)
    # Aggregates all reversal patterns into a single score
    # +1 bullish reversal signals, -1 bearish reversal signals
    # =========================================
    bull_signals = (
        df['Engulf_Bull'] + df['Hammer_Bull'] + df['DragonflyDoji'] +
        df['MorningStar'] + df['PiercingLine'] + df['ThreeSoldiers'] +
        df['MarubozuBull']
    )
    bear_signals = (
        df['Engulf_Bear'] + df['Hammer_Bear'] + df['GravestoneDoji'] +
        df['EveningStar'] + df['DarkCloudCover'] + df['ThreeCrows'] +
        df['MarubozuBear']
    )
    df['candle_reversal_score'] = (bull_signals - bear_signals).clip(-3, 3)

    logger.info("[Candles] Added 16 candlestick pattern flags + composite reversal score.")
    return df

"""
Feature Engineering Pipeline — Elite v3.0
==========================================
Comprehensive feature extraction for LSTM model.

Features:
- Technical Indicators: RSI, MACD, Bollinger Bands, ATR
- Trend Detection: EMA 50/200 crossover, ADX
- Momentum: ROC (Rate of Change)
- Support/Resistance: Pivot Points (Standard)
- Session Awareness: Binary flags for London/NY/Asia
- Candlestick Price Action: Body size, shadow ratios
- Feature Drift Detection: Rolling mean/std monitoring
- Multi-Timeframe feature injection
"""

import pandas as pd
import numpy as np
import logging
from config import Config
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, ADXIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

logger = logging.getLogger("Features")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(Config.LOG_FILE, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("\033[95m%(asctime)s\033[0m [%(levelname)s] %(message)s"))
    logger.addHandler(_ch)


# =========================================
# CORE TECHNICAL INDICATORS
# =========================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds RSI, MACD, Bollinger Bands, and ATR."""
    # RSI
    df['RSI'] = RSIIndicator(close=df["close"], window=14).rsi()

    # MACD
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df['BB_high'] = bb.bollinger_hband()
    df['BB_low'] = bb.bollinger_lband()
    df['BB_mid'] = bb.bollinger_mavg()
    df['BB_width'] = (df['BB_high'] - df['BB_low']) / (df['BB_mid'] + 1e-8)
    df['BB_position'] = (df['close'] - df['BB_low']) / (df['BB_high'] - df['BB_low'] + 1e-8)

    # ATR
    df['ATR'] = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()

    # Volatility (normalized ATR)
    df['Volatility'] = df['ATR'] / (df['close'] + 1e-8)

    return df


# =========================================
# TREND DETECTION (EMA 50/200 + ADX)
# =========================================

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds EMA crossover and ADX for trend detection.
    - EMA 50/200 crossover gives the "Big Picture" direction
    - ADX tells us if the market is trending or ranging
    """
    # EMA 50 & 200
    df['EMA_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
    df['EMA_200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()

    # Trend direction: 1 = bullish (EMA50 > EMA200), -1 = bearish, 0 = flat
    df['trend_direction'] = np.where(
        df['EMA_50'] > df['EMA_200'], 1,
        np.where(df['EMA_50'] < df['EMA_200'], -1, 0)
    )

    # Distance between EMAs (normalized)
    df['ema_spread'] = (df['EMA_50'] - df['EMA_200']) / (df['close'] + 1e-8)

    # ADX — Average Directional Index (trend strength)
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['ADX'] = adx.adx()
    df['DI_plus'] = adx.adx_pos()
    df['DI_minus'] = adx.adx_neg()

    # Is market trending? (ADX > threshold)
    df['is_trending'] = np.where(df['ADX'] >= Config.ADX_RANGING_THRESHOLD, 1, 0)

    return df


# =========================================
# MOMENTUM (ROC — Rate of Change)
# =========================================

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds momentum indicators: ROC at multiple windows."""
    df['ROC_5'] = ROCIndicator(close=df['close'], window=5).roc()
    df['ROC_10'] = ROCIndicator(close=df['close'], window=10).roc()
    df['ROC_20'] = ROCIndicator(close=df['close'], window=20).roc()

    # Momentum direction consistency
    df['momentum_agreement'] = np.where(
        (df['ROC_5'] > 0) & (df['ROC_10'] > 0) & (df['ROC_20'] > 0), 1,
        np.where(
            (df['ROC_5'] < 0) & (df['ROC_10'] < 0) & (df['ROC_20'] < 0), -1, 0
        )
    )

    return df


# =========================================
# SUPPORT / RESISTANCE (Pivot Points)
# =========================================

def add_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Standard Pivot Points.
    Pivot = (High + Low + Close) / 3
    """
    # Use previous candle's data for pivots
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    prev_close = df['close'].shift(1)

    df['Pivot'] = (prev_high + prev_low + prev_close) / 3
    df['R1'] = 2 * df['Pivot'] - prev_low
    df['S1'] = 2 * df['Pivot'] - prev_high
    df['R2'] = df['Pivot'] + (prev_high - prev_low)
    df['S2'] = df['Pivot'] - (prev_high - prev_low)

    # Distance from current price to nearest S/R levels (normalized)
    df['dist_to_R1'] = (df['R1'] - df['close']) / (df['ATR'] + 1e-8)
    df['dist_to_S1'] = (df['close'] - df['S1']) / (df['ATR'] + 1e-8)
    df['dist_to_Pivot'] = (df['close'] - df['Pivot']) / (df['ATR'] + 1e-8)

    return df


# =========================================
# SESSION AWARENESS (Binary Flags)
# =========================================

def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds binary session flags based on candle timestamp.
    NOTE: These are based on the candle's own timestamp (UTC from MT5),
    which is the server time.
    """
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df['is_london'] = 0
        df['is_ny'] = 0
        df['is_asia'] = 0
        df['session_overlap'] = 0
        logger.warning("[Sessions] Index is not datetime. Session features set to 0.")
        return df

    hours = df.index.hour

    df['is_london'] = np.where(
        (hours >= Config.SESSION_LONDON[0]) & (hours < Config.SESSION_LONDON[1]), 1, 0
    )
    df['is_ny'] = np.where(
        (hours >= Config.SESSION_NY[0]) & (hours < Config.SESSION_NY[1]), 1, 0
    )
    df['is_asia'] = np.where(
        (hours >= Config.SESSION_ASIA[0]) & (hours < Config.SESSION_ASIA[1]), 1, 0
    )

    # London-NY overlap (highest liquidity)
    df['session_overlap'] = np.where(
        (df['is_london'] == 1) & (df['is_ny'] == 1), 1, 0
    )

    return df


# =========================================
# CANDLESTICK PRICE ACTION
# =========================================

def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates numerical representations of candlestick patterns."""
    epsilon = 1e-8

    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_shadow_ratio'] = df['upper_shadow'] / (df['body_size'] + epsilon)
    df['lower_shadow_ratio'] = df['lower_shadow'] / (df['body_size'] + epsilon)

    # Body direction: 1 bullish, -1 bearish, 0 doji
    df['body_direction'] = np.where(
        df['close'] > df['open'], 1,
        np.where(df['close'] < df['open'], -1, 0)
    )

    # Candle range relative to ATR
    df['candle_range_atr'] = (df['high'] - df['low']) / (df['ATR'] + epsilon)

    return df


# =========================================
# MULTI-TIMEFRAME FEATURE INJECTION
# =========================================

def inject_mtf_features(df_primary: pd.DataFrame, df_confirm: pd.DataFrame,
                        df_trend: pd.DataFrame) -> pd.DataFrame:
    """
    Injects higher-timeframe context into the primary (M5) DataFrame.
    Uses forward-fill to align different timeframe data.
    """
    df = df_primary.copy()

    # M15 Features (Confirmation)
    if df_confirm is not None and not df_confirm.empty:
        df_confirm = df_confirm.copy()
        df_confirm['M15_RSI'] = RSIIndicator(close=df_confirm['close'], window=14).rsi()
        df_confirm['M15_EMA_50'] = EMAIndicator(close=df_confirm['close'], window=50).ema_indicator()
        df_confirm['M15_trend'] = np.where(
            df_confirm['close'] > df_confirm['M15_EMA_50'], 1, -1
        )

        # Reindex to M5 timeline with forward fill
        mtf_cols = ['M15_RSI', 'M15_EMA_50', 'M15_trend']
        df_confirm_reindexed = df_confirm[mtf_cols].reindex(df.index, method='ffill')
        df = pd.concat([df, df_confirm_reindexed], axis=1)
        logger.info("[MTF] Injected M15 features: %s", mtf_cols)

    # H1 Features (Trend Direction)
    if df_trend is not None and not df_trend.empty:
        df_trend = df_trend.copy()
        df_trend['H1_EMA_50'] = EMAIndicator(close=df_trend['close'], window=50).ema_indicator()
        df_trend['H1_EMA_200'] = EMAIndicator(close=df_trend['close'], window=200).ema_indicator()
        df_trend['H1_trend'] = np.where(
            df_trend['H1_EMA_50'] > df_trend['H1_EMA_200'], 1, -1
        )
        adx_h1 = ADXIndicator(high=df_trend['high'], low=df_trend['low'],
                               close=df_trend['close'], window=14)
        df_trend['H1_ADX'] = adx_h1.adx()

        mtf_cols = ['H1_EMA_50', 'H1_EMA_200', 'H1_trend', 'H1_ADX']
        df_trend_reindexed = df_trend[mtf_cols].reindex(df.index, method='ffill')
        df = pd.concat([df, df_trend_reindexed], axis=1)
        logger.info("[MTF] Injected H1 features: %s", mtf_cols)

    return df


# =========================================
# FEATURE DRIFT DETECTION
# =========================================

def detect_feature_drift(df: pd.DataFrame) -> dict:
    """
    Monitors key features for distribution shifts.
    Returns a dict of features that have drifted beyond the threshold.
    """
    if not Config.DRIFT_DETECTION_ENABLED:
        return {}

    drift_report = {}
    monitor_cols = ['RSI', 'ATR', 'BB_width', 'ADX', 'Volatility']

    for col in monitor_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if len(series) < Config.DRIFT_WINDOW * 2:
            continue

        # Compare recent vs historical
        historical = series.iloc[:-Config.DRIFT_WINDOW]
        recent = series.iloc[-Config.DRIFT_WINDOW:]

        hist_mean = historical.mean()
        hist_std = historical.std()

        if hist_std <= 0:
            continue

        recent_mean = recent.mean()
        z_score = abs(recent_mean - hist_mean) / hist_std

        if z_score >= Config.DRIFT_THRESHOLD_ZSCORE:
            drift_report[col] = {
                "z_score": round(z_score, 2),
                "hist_mean": round(hist_mean, 6),
                "recent_mean": round(recent_mean, 6),
                "hist_std": round(hist_std, 6),
            }
            logger.warning(
                "[DRIFT] ⚠️ Feature '%s' drifted! Z-score: %.2f (threshold: %.1f) | "
                "Historical μ=%.6f → Recent μ=%.6f",
                col, z_score, Config.DRIFT_THRESHOLD_ZSCORE, hist_mean, recent_mean,
            )

    if drift_report:
        logger.warning("[DRIFT] %d features drifted. Consider retraining.", len(drift_report))
    else:
        logger.debug("[DRIFT] All features within normal range.")

    return drift_report


# =========================================
# TARGET GENERATION
# =========================================

def generate_target_column(df: pd.DataFrame, lookahead: int = Config.PREDICT_LOOKAHEAD) -> pd.DataFrame:
    """
    Creates the Target column.
    1 if Close price N periods from now is strictly greater than current Close.
    0 otherwise.
    """
    df['future_close'] = df['close'].shift(-lookahead)
    df['Target'] = np.where(df['future_close'] > df['close'], 1, 0)
    df.drop(['future_close'], axis=1, inplace=True)
    return df


# =========================================
# MASTER PIPELINE
# =========================================

def feature_engineering_pipeline(df: pd.DataFrame, df_confirm=None, df_trend=None) -> pd.DataFrame:
    """
    Runs the complete feature engineering pipeline.

    Args:
        df: Primary timeframe DataFrame (M5)
        df_confirm: Confirmation timeframe DataFrame (M15) — optional
        df_trend: Trend timeframe DataFrame (H1) — optional

    Returns:
        Fully featured DataFrame ready for LSTM.
    """
    logger.info("Starting feature engineering pipeline...")
    df = df.copy()

    # Core indicators
    df = add_technical_indicators(df)
    df = add_trend_features(df)
    df = add_momentum_features(df)
    df = add_pivot_points(df)
    df = add_session_features(df)
    df = add_price_action_features(df)

    # Time-based features
    if pd.api.types.is_datetime64_any_dtype(df.index):
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
    else:
        df['hour'] = 0
        df['day_of_week'] = 0

    # Multi-Timeframe injection
    if df_confirm is not None or df_trend is not None:
        df = inject_mtf_features(df, df_confirm, df_trend)

    # Target
    df = generate_target_column(df)

    # ATR Liquidity Filter
    initial_len = len(df)
    df = df[df['ATR'] >= Config.ATR_THRESHOLD]
    dropped = initial_len - len(df)
    if dropped > 0:
        logger.info("Dropped %d rows due to ATR liquidity filter (< %s).", dropped, Config.ATR_THRESHOLD)

    # Feature Drift Detection (runs on every cycle)
    drift_report = detect_feature_drift(df)

    # Drop NaNs (from indicators and target shift)
    pre_shape = df.shape
    subset_cols = [c for c in df.columns if c != 'Target']
    df.dropna(subset=subset_cols, inplace=True)
    logger.info("Data shape: %s → %s (after NaN cleanup)", pre_shape, df.shape)

    return df

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

# Phase 1: PDF Vision Layer — Advanced Detectors
from candles import add_candlestick_patterns
from pattern_detector import add_chart_patterns
from divergence import add_divergence_features

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

    # Is market trending? (ADX > 25 = TREND (1), else RANGE (0))
    df['is_trending'] = np.where(df['ADX'] > 25, 1, 0)

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
# SMART MONEY CONCEPTS (Phase 2)
# =========================================

def add_market_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Market Structure (BOS, HH/HL, Structure Trend)"""
    window = 10
    
    # Past window max/min to avoid lookahead bias
    df['past_max'] = df['high'].shift(1).rolling(window=window).max()
    df['past_min'] = df['low'].shift(1).rolling(window=window).min()
    
    # HH / LL flags
    df['higher_high'] = np.where(df['high'] > df['past_max'], 1, 0)
    df['lower_low'] = np.where(df['low'] < df['past_min'], 1, 0)
    
    # Structure Trend
    df['structure_event'] = np.where(df['higher_high'] == 1, 1, np.where(df['lower_low'] == 1, -1, 0))
    df['structure_trend'] = df['structure_event'].replace(0, np.nan).ffill().fillna(0)
    
    # BOS Strength
    bos_raw = np.where(
        df['higher_high'] == 1, (df['close'] - df['past_max']) / (df['ATR'] + 1e-8),
        np.where(df['lower_low'] == 1, (df['past_min'] - df['close']) / (df['ATR'] + 1e-8), 0.0)
    )
    df['bos_strength'] = pd.Series(bos_raw).clip(lower=0.0).values
    
    df.drop(['past_max', 'past_min', 'structure_event'], axis=1, inplace=True)
    return df

def add_order_block_features(df: pd.DataFrame) -> pd.DataFrame:
    """Order Blocks (Distance, Zone, Strength)"""
    # Impulsive moves (past 3 candles)
    df['move_3'] = df['close'] - df['close'].shift(3)
    df['impulsive_bullish'] = df['move_3'] > (2 * df['ATR'])
    df['impulsive_bearish'] = df['move_3'] < -(2 * df['ATR'])
    
    df['ob_bullish_price'] = df['low'].shift(3).rolling(3).min()
    df['ob_bearish_price'] = df['high'].shift(3).rolling(3).max()
    
    df['active_bullish_ob'] = np.where(df['impulsive_bullish'], df['ob_bullish_price'], np.nan)
    df['active_bearish_ob'] = np.where(df['impulsive_bearish'], df['ob_bearish_price'], np.nan)
    
    df['active_bullish_ob'] = pd.Series(df['active_bullish_ob']).ffill().values
    df['active_bearish_ob'] = pd.Series(df['active_bearish_ob']).ffill().values
    
    df['dist_to_bullish_ob'] = df['close'] - df['active_bullish_ob']
    df['dist_to_bearish_ob'] = df['active_bearish_ob'] - df['close']
    
    # distance_to_ob
    df['distance_to_ob'] = df[['dist_to_bullish_ob', 'dist_to_bearish_ob']].abs().min(axis=1) / (df['ATR'] + 1e-8)
    
    # inside_ob_zone
    df['inside_ob_zone'] = np.where(df['distance_to_ob'] < 0.5, 1, 0)
    
    # ob_strength
    df['ob_strength_bullish'] = np.where(df['impulsive_bullish'], df['move_3'] / df['ATR'], np.nan)
    df['ob_strength_bearish'] = np.where(df['impulsive_bearish'], abs(df['move_3']) / df['ATR'], np.nan)
    df['ob_strength_bullish'] = pd.Series(df['ob_strength_bullish']).ffill().values
    df['ob_strength_bearish'] = pd.Series(df['ob_strength_bearish']).ffill().values
    
    df['ob_strength'] = np.where(
        abs(df['dist_to_bullish_ob']) < abs(df['dist_to_bearish_ob']), 
        df['ob_strength_bullish'], 
        df['ob_strength_bearish']
    )
    df['ob_strength'] = df['ob_strength'].fillna(0)
    
    drop_cols = ['move_3', 'impulsive_bullish', 'impulsive_bearish', 'ob_bullish_price', 'ob_bearish_price', 
                 'dist_to_bullish_ob', 'dist_to_bearish_ob',
                 'ob_strength_bullish', 'ob_strength_bearish']
    # NOTE: active_bullish_ob and active_bearish_ob are KEPT — main.py reads them for state tracking
    df.drop(drop_cols, axis=1, inplace=True)
    return df

def add_fvg_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fair Value Gaps (Size, Distance, Filled)"""
    df['fvg_bullish_gap'] = df['low'] - df['high'].shift(2)
    df['is_fvg_bullish'] = df['fvg_bullish_gap'] > 0
    
    df['fvg_bearish_gap'] = df['low'].shift(2) - df['high']
    df['is_fvg_bearish'] = df['fvg_bearish_gap'] > 0
    
    df['fvg_size'] = np.where(df['is_fvg_bullish'], df['fvg_bullish_gap'] / (df['ATR'] + 1e-8),
                     np.where(df['is_fvg_bearish'], df['fvg_bearish_gap'] / (df['ATR'] + 1e-8), 0.0))
    
    df['last_fvg_price'] = np.where(df['is_fvg_bullish'], df['high'].shift(2) + (df['fvg_bullish_gap']/2),
                           np.where(df['is_fvg_bearish'], df['low'].shift(2) - (df['fvg_bearish_gap']/2), np.nan))
    df['last_fvg_price'] = pd.Series(df['last_fvg_price']).ffill().values
    
    df['distance_to_fvg'] = abs(df['close'] - df['last_fvg_price']) / (df['ATR'] + 1e-8)
    df['distance_to_fvg'] = df['distance_to_fvg'].fillna(0)
    
    df['fvg_filled'] = np.where((df['distance_to_fvg'] < 0.2) & (df['last_fvg_price'].notna()), 1, 0)
    
    df.drop(['fvg_bullish_gap', 'is_fvg_bullish', 'fvg_bearish_gap', 'is_fvg_bearish'], axis=1, inplace=True)
    return df

def add_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Liquidity Sweeps and Equal Highs"""
    window = 10
    past_highs = df['high'].shift(1).rolling(window).max()
    past_lows = df['low'].shift(1).rolling(window).min()
    
    df['equal_highs_count'] = np.where(abs(df['high'] - past_highs) < (0.1 * df['ATR']), 1, 0)
    df['equal_highs_count'] = pd.Series(df['equal_highs_count']).rolling(window).sum().fillna(0).values
    
    sweep_bullish = (df['high'] > past_highs) & (df['close'] < past_highs) & (df['upper_shadow_ratio'] > 0.5)
    sweep_bearish = (df['low'] < past_lows) & (df['close'] > past_lows) & (df['lower_shadow_ratio'] > 0.5)
    
    df['liquidity_sweep_flag'] = np.where(sweep_bullish, -1, np.where(sweep_bearish, 1, 0))
    return df

# =========================================
# TARGET GENERATION
# =========================================

def generate_target_column(df: pd.DataFrame, lookahead: int = 6, symbol: str = None) -> pd.DataFrame:
    """
    Creates the Target column with strict ATR-based noise filtering.
    BUY (1) if future move > ATR_LOOKAHEAD_MULT * ATR
    SELL (0) if future move < -ATR_LOOKAHEAD_MULT * ATR
    HOLD (NaN) otherwise.

    v5.1: Uses per-symbol ATR_LOOKAHEAD_MULT to match training targets exactly.
    """
    df['future_close'] = df['close'].shift(-lookahead)
    future_move = df['future_close'] - df['close']

    # v5.1: Use per-symbol multiplier if available, else fall back to global default
    per_symbol_mults = getattr(Config, 'ATR_LOOKAHEAD_MULT_PER_SYMBOL', {})
    if symbol and symbol in per_symbol_mults:
        atr_mult = per_symbol_mults[symbol]
    else:
        atr_mult = getattr(Config, 'ATR_LOOKAHEAD_MULT', 1.2)

    threshold = df['ATR'] * atr_mult

    # 1 = BUY, 0 = SELL, np.nan = HOLD (noise)
    df['Target'] = np.where(future_move > threshold, 1,
                            np.where(future_move < -threshold, 0, np.nan))

    df.drop(['future_close'], axis=1, inplace=True)
    return df


# =========================================
# MASTER PIPELINE
# =========================================

def feature_engineering_pipeline(df: pd.DataFrame, df_confirm=None, df_trend=None, symbol: str = None) -> pd.DataFrame:
    """
    Runs the complete feature engineering pipeline.

    Args:
        df: Primary timeframe DataFrame (M5)
        df_confirm: Confirmation timeframe DataFrame (M15) — optional
        df_trend: Trend timeframe DataFrame (H1) — optional
        symbol: Trading symbol (e.g. 'XAUUSD') — used for per-symbol ATR_LOOKAHEAD_MULT

    Returns:
        Fully featured DataFrame ready for XGBoost/RF.
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

    # Smart Money Concepts (Phase 2)
    df = add_market_structure_features(df)
    df = add_order_block_features(df)
    df = add_fvg_features(df)
    df = add_liquidity_features(df)

    # Phase 1: PDF Vision Layer — Advanced Detectors
    logger.info("Applying PDF Vision Layer detectors...")
    df = add_candlestick_patterns(df)     # 16 candle patterns + composite
    df = add_chart_patterns(df)            # 14 chart patterns + composite + vol squeeze
    df = add_divergence_features(df)       # 4 divergence types + composite
    logger.info("PDF Vision Layer complete. +36 new features added.")

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

    # Defragment before target generation (fixes PerformanceWarning)
    df = df.copy()

    # Target — pass symbol for per-symbol ATR_LOOKAHEAD_MULT
    df = generate_target_column(df, symbol=symbol)

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

    # Defragment DataFrame (fixes PerformanceWarning from many column insertions)
    df = df.copy()

    return df

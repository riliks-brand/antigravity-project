"""
Random Forest Model — Elite v3.1
==================================
Complementary model for the Ensemble Voting System.

Key Design: DE-CORRELATED from LSTM
- LSTM sees: raw sequential time-series data
- RF sees: engineered INTERACTION features (cross-products, rolling stats, divergences)

This ensures the two models "think differently" and the ensemble
provides genuine diversification, not just redundancy.

Features:
- Engineered interactive features (RSI*ATR, MACD divergence, etc.)
- Rolling statistical features (std, skew, z-scores)
- Feature Importance exposure for market insight
- Configurable retraining schedule
"""

import numpy as np
import pandas as pd
import os
import time
import datetime
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
from config import Config

from logging_setup import setup_module_logger
logger = setup_module_logger("RF_Model", Config.LOG_FILE, console_color="\033[93m")

# Model persistence
RF_MODEL_PATH = "rf_model.joblib"
RF_SCALER_PATH = "rf_scaler.joblib"


# =========================================
# DE-CORRELATED FEATURE ENGINEERING FOR RF
# =========================================

def engineer_rf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates interaction and statistical features specifically for Random Forest.
    These are DIFFERENT from what LSTM sees (raw sequences).

    Categories:
    1. Cross-product interactions (RSI*ATR, ADX*Volatility)
    2. Rolling statistical moments (std, skew over windows)
    3. Divergence signals (price vs indicator direction)
    4. Ratio features (normalized relationships)
    5. Lagged delta features (rate of change of indicators)
    """
    rf = pd.DataFrame(index=df.index)

    # --- 1. CROSS-PRODUCT INTERACTIONS ---
    # These capture non-linear relationships between indicators
    if 'RSI' in df.columns and 'ATR' in df.columns:
        rf['RSI_x_ATR'] = df['RSI'] * df['ATR']

    if 'ADX' in df.columns and 'Volatility' in df.columns:
        rf['ADX_x_Vol'] = df['ADX'] * df['Volatility']

    if 'MACD' in df.columns and 'RSI' in df.columns:
        rf['MACD_x_RSI'] = df['MACD'] * (df['RSI'] - 50) / 50  # Centered RSI

    if 'BB_position' in df.columns and 'ADX' in df.columns:
        rf['BB_pos_x_ADX'] = df['BB_position'] * df['ADX']

    if 'body_size' in df.columns and 'ATR' in df.columns:
        rf['body_ATR_ratio'] = df['body_size'] / (df['ATR'] + 1e-8)

    # --- 2. ROLLING STATISTICAL MOMENTS ---
    # RF excels at splitting on distribution boundaries
    for window in [10, 20, 50]:
        if 'close' in df.columns:
            rf[f'close_std_{window}'] = df['close'].rolling(window).std()
            rf[f'close_zscore_{window}'] = (
                (df['close'] - df['close'].rolling(window).mean()) /
                (df['close'].rolling(window).std() + 1e-8)
            )

        if 'RSI' in df.columns:
            rf[f'RSI_std_{window}'] = df['RSI'].rolling(window).std()

        if 'ATR' in df.columns:
            rf[f'ATR_mean_{window}'] = df['ATR'].rolling(window).mean()

    # Skewness of returns (asymmetry detector)
    if 'close' in df.columns:
        returns = df['close'].pct_change()
        rf['returns_skew_20'] = returns.rolling(20).skew()
        rf['returns_kurt_20'] = returns.rolling(20).kurt()

    # --- 3. DIVERGENCE SIGNALS ---
    # Price makes new high but RSI doesn't → bearish divergence
    if 'RSI' in df.columns and 'close' in df.columns:
        price_high_10 = df['close'].rolling(10).max()
        rsi_high_10 = df['RSI'].rolling(10).max()
        rf['price_rsi_divergence'] = (
            (df['close'] / (price_high_10 + 1e-8)) -
            (df['RSI'] / (rsi_high_10 + 1e-8))
        )

    # MACD histogram direction vs price direction
    if 'MACD_hist' in df.columns and 'close' in df.columns:
        rf['macd_price_agree'] = np.sign(df['MACD_hist']) * np.sign(df['close'].diff())

    # --- 4. RATIO FEATURES ---
    if 'upper_shadow' in df.columns and 'lower_shadow' in df.columns:
        rf['shadow_ratio'] = (df['upper_shadow'] - df['lower_shadow']) / (
            df['upper_shadow'] + df['lower_shadow'] + 1e-8
        )

    if 'EMA_50' in df.columns and 'EMA_200' in df.columns:
        rf['ema_ratio'] = df['EMA_50'] / (df['EMA_200'] + 1e-8)

    if 'DI_plus' in df.columns and 'DI_minus' in df.columns:
        rf['di_ratio'] = (df['DI_plus'] - df['DI_minus']) / (
            df['DI_plus'] + df['DI_minus'] + 1e-8
        )

    # --- 5. LAGGED DELTA (Rate of change of indicators) ---
    for col in ['RSI', 'ADX', 'ATR', 'MACD']:
        if col in df.columns:
            rf[f'{col}_delta_1'] = df[col].diff(1)
            rf[f'{col}_delta_5'] = df[col].diff(5)
            rf[f'{col}_accel'] = df[col].diff(1).diff(1)  # Acceleration

    # --- 6. VOLATILITY REGIME ---
    if 'ATR' in df.columns:
        atr_ma = df['ATR'].rolling(50).mean()
        rf['vol_regime'] = df['ATR'] / (atr_ma + 1e-8)  # >1 = high vol, <1 = low vol

    # --- 7. SESSION & TIME ENCODING ---
    if 'is_london' in df.columns:
        rf['is_london'] = df['is_london']
    if 'is_ny' in df.columns:
        rf['is_ny'] = df['is_ny']
    if 'session_overlap' in df.columns:
        rf['session_overlap'] = df['session_overlap']

    # --- 8. TREND CONTEXT FROM HIGHER TF ---
    for col in ['H1_trend', 'H1_ADX', 'M15_trend', 'M15_RSI']:
        if col in df.columns:
            rf[col] = df[col]

    # --- INCLUDE TARGET ---
    if 'Target' in df.columns:
        rf['Target'] = df['Target']

    return rf


# =========================================
# TRAINING & PREDICTION
# =========================================

class RFModel:
    """Random Forest model wrapper with retraining schedule."""

    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.feature_names = []
        self.feature_importances = {}
        self.last_train_time = None
        self.train_count = 0
        self.accuracy = 0.0

        # Try loading saved model
        self._load_model()

    def _load_model(self):
        """Load previously trained model if available."""
        if os.path.exists(RF_MODEL_PATH) and os.path.exists(RF_SCALER_PATH):
            try:
                self.model = joblib.load(RF_MODEL_PATH)
                self.scaler = joblib.load(RF_SCALER_PATH)
                if os.path.exists("rf_features.joblib"):
                    self.feature_names = joblib.load("rf_features.joblib")
                logger.info("[RF] Loaded saved model from disk with %d features.", len(self.feature_names))
            except Exception as e:
                logger.warning("[RF] Failed to load saved model: %s", e)
                self.model = None

    def _save_model(self):
        """Persist trained model to disk."""
        try:
            joblib.dump(self.model, RF_MODEL_PATH)
            joblib.dump(self.scaler, RF_SCALER_PATH)
            joblib.dump(self.feature_names, "rf_features.joblib")
            logger.info("[RF] Model saved to disk.")
        except Exception as e:
            logger.warning("[RF] Failed to save model: %s", e)

    def needs_retraining(self, candle_count=0):
        """Check if RF needs retraining based on schedule."""
        if self.model is None:
            return True

        if self.last_train_time is None:
            return True

        # Time-based check
        hours_since = (datetime.datetime.utcnow() - self.last_train_time).total_seconds() / 3600
        if hours_since >= Config.RF_RETRAIN_EVERY_HOURS:
            logger.info("[RF] Retraining triggered: %.1f hours since last train.", hours_since)
            return True

        # Candle-based check
        if candle_count > 0 and (candle_count - self.train_count) >= Config.RF_RETRAIN_EVERY_CANDLES:
            logger.info("[RF] Retraining triggered: %d candles since last train.",
                        candle_count - self.train_count)
            return True

        return False

    def train(self, df_full: pd.DataFrame):
        """
        Train the Random Forest on de-correlated features.

        Args:
            df_full: The fully featured DataFrame (from feature_engineering_pipeline)

        Returns:
            accuracy on test set
        """
        logger.info("[RF] Engineering de-correlated features...")
        rf_df = engineer_rf_features(df_full)

        # Drop NaNs
        rf_df = rf_df.dropna()

        if len(rf_df) < 100:
            logger.warning("[RF] Not enough data (%d rows). Need at least 100.", len(rf_df))
            return 0.0

        # Separate features and target
        feature_cols = [c for c in rf_df.columns if c != 'Target']
        self.feature_names = feature_cols

        X = rf_df[feature_cols].values
        y = rf_df['Target'].values

        # Chronological split (80/20)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Scale
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Build and train RF — v6.0: Tuned for better signal/noise separation
        # المشكلة: 68% من الـ predictions في noise zone [0.45-0.55]
        # الحل: 
        #   1. More trees = smoother probabilities
        #   2. Deeper trees = capture complex patterns
        #   3. Stricter leaf requirements = less noise
        self.model = RandomForestClassifier(
            n_estimators=300,           # v6.0: raised from Config (was 200) — more stable predictions
            max_depth=8,                # v6.0: raised from Config (was 6) — capture deeper patterns
            min_samples_split=20,       # v6.0: raised from 10 — prevent overfitting
            min_samples_leaf=10,        # v6.0: raised from 5 — smoother predictions
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        self.accuracy = accuracy_score(y_test, y_pred)

        # Feature importance
        importances = self.model.feature_importances_
        self.feature_importances = dict(zip(self.feature_names, importances))
        sorted_imp = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)

        # Log results
        logger.info("[RF] ✅ Training complete. Accuracy: %.2f%%", self.accuracy * 100)
        logger.info("[RF] Top 5 Features:")
        for i, (feat, imp) in enumerate(sorted_imp[:5], 1):
            logger.info("[RF]   %d. %s: %.4f", i, feat, imp)

        # Print report
        print(f"\n\033[93m{'='*55}\033[0m")
        print(f"\033[93m        RANDOM FOREST REPORT\033[0m")
        print(f"\033[93m{'='*55}\033[0m")
        print(f"\033[93m  Accuracy     : {self.accuracy * 100:.2f}%\033[0m")
        print(f"\033[93m  Trees        : {Config.RF_N_ESTIMATORS}\033[0m")
        print(f"\033[93m  Max Depth    : {Config.RF_MAX_DEPTH}\033[0m")
        print(f"\033[93m  Train Size   : {len(X_train)}\033[0m")
        print(f"\033[93m  Test Size    : {len(X_test)}\033[0m")
        print(f"\033[93m  RF Features  : {len(feature_cols)} (de-correlated)\033[0m")
        print(f"\033[93m  Top Feature  : {sorted_imp[0][0]} ({sorted_imp[0][1]:.4f})\033[0m")
        print(f"\033[93m{'='*55}\033[0m\n")

        # Update schedule tracking
        self.last_train_time = datetime.datetime.utcnow()
        self._save_model()

        return self.accuracy

    def predict_proba(self, df_full: pd.DataFrame):
        """
        Get RF probability for the latest data point.

        v6.0: Apply probability sharpening to reduce noise zone concentration.
        
        Strategy: Power transformation to push probabilities away from 0.5
        - p_new = 0.5 + sign(p - 0.5) * |p - 0.5|^0.8
        - This expands the tails while preserving order
        - Example: 0.52 → 0.53, 0.65 → 0.68, 0.48 → 0.47

        Returns:
            float: probability of class 1 (BUY direction) ∈ [0, 1]
            Returns 0.5 (neutral) if model not trained or error.
        """
        if self.model is None:
            logger.warning("[RF] Model not trained. Returning neutral 0.5.")
            return 0.5

        try:
            rf_df = engineer_rf_features(df_full)
            rf_df = rf_df.drop('Target', axis=1, errors='ignore')

            # Use only the latest row
            latest = rf_df.iloc[-1:].copy()

            # Fill any NaN from rolling calculations
            latest = latest.fillna(0)

            # Ensure column alignment with training
            for col in self.feature_names:
                if col not in latest.columns:
                    latest[col] = 0.0

            latest = latest[self.feature_names]

            X = self.scaler.transform(latest.values)
            proba = self.model.predict_proba(X)[0]

            # proba[1] = probability of class 1 (bullish)
            raw_prob = float(proba[1]) if len(proba) > 1 else 0.5
            
            # v6.0: Probability sharpening to reduce noise zone
            # Power transformation: pushes probabilities away from 0.5
            distance = raw_prob - 0.5
            sign = 1 if distance >= 0 else -1
            sharpened = 0.5 + sign * (abs(distance) ** 0.8)
            
            # Clamp to [0, 1]
            sharpened = float(np.clip(sharpened, 0.0, 1.0))
            
            # Log significant changes
            if abs(sharpened - raw_prob) > 0.02:
                logger.debug("[RF] Probability sharpening: %.4f → %.4f", raw_prob, sharpened)
            
            return sharpened

        except Exception as e:
            logger.error("[RF] Prediction failed: %s", e)
            return 0.5

    def get_top_features(self, n=5):
        """
        Get top N feature importances sorted by importance.
        Returns list of (feature_name, importance) tuples.
        """
        if not self.feature_importances:
            return []

        sorted_imp = sorted(self.feature_importances.items(),
                            key=lambda x: x[1], reverse=True)
        return sorted_imp[:n]

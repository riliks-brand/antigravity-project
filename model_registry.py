"""
model_registry.py — Elite Trading Bot v3.2
===========================================
Loads and serves per-symbol LSTM + RF models.
Falls back to universal model if symbol model missing.

Usage in main.py:
    from model_registry import ModelRegistry
    registry = ModelRegistry()
    lstm_prob = registry.predict_lstm("XAUUSD", df_processed)
    rf_prob   = registry.predict_rf("XAUUSD", df_processed)
"""

import os
import numpy as np
import logging
from joblib import load

logger = logging.getLogger("ModelRegistry")

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US30", "BTCUSD"]

def _lstm_path(sym):      return f"lstm_model_{sym}.h5"
def _scaler_path(sym):    return f"lstm_scaler_{sym}.joblib"
def _rf_path(sym):        return f"rf_model_{sym}.joblib"
def _rf_scaler_path(sym): return f"rf_scaler_{sym}.joblib"
def _rf_feat_path(sym):   return f"rf_features_{sym}.joblib"


class ModelRegistry:

    def __init__(self):
        self.lstm_models  = {}   # symbol → keras model
        self.lstm_scalers = {}   # symbol → scaler
        self.rf_models    = {}   # symbol → RF model
        self.rf_scalers   = {}   # symbol → scaler
        self.rf_features  = {}   # symbol → feature list
        self._load_all()

    def _load_all(self):
        try:
            from tensorflow.keras.models import load_model as keras_load
        except ImportError:
            keras_load = None

        for sym in SYMBOLS + ["universal"]:
            # LSTM
            lp = _lstm_path(sym)
            sp = _scaler_path(sym)
            if keras_load and os.path.exists(lp) and os.path.exists(sp):
                try:
                    self.lstm_models[sym]  = keras_load(lp)
                    self.lstm_scalers[sym] = load(sp)
                    logger.info("[Registry] Loaded LSTM for %s", sym)
                except Exception as e:
                    logger.warning("[Registry] Failed to load LSTM %s: %s", sym, e)

            # RF
            rp  = _rf_path(sym)
            rsp = _rf_scaler_path(sym)
            rfp = _rf_feat_path(sym)
            if os.path.exists(rp) and os.path.exists(rsp) and os.path.exists(rfp):
                try:
                    self.rf_models[sym]   = load(rp)
                    self.rf_scalers[sym]  = load(rsp)
                    self.rf_features[sym] = load(rfp)
                    logger.info("[Registry] Loaded RF for %s", sym)
                except Exception as e:
                    logger.warning("[Registry] Failed to load RF %s: %s", sym, e)

        logger.info("[Registry] Loaded LSTM for: %s", list(self.lstm_models.keys()))
        logger.info("[Registry] Loaded RF for:   %s", list(self.rf_models.keys()))

    def _resolve_sym(self, symbol, model_dict):
        """Returns symbol key — falls back to universal if missing."""
        if symbol in model_dict:
            return symbol
        if "universal" in model_dict:
            logger.debug("[Registry] %s not found, using universal fallback.", symbol)
            return "universal"
        return None

    def predict_lstm(self, symbol: str, df_processed, sequence_length=60) -> float:
        """
        Returns LSTM probability ∈ [0,1] for the given symbol.
        Falls back to 0.5 if model unavailable.
        """
        key = self._resolve_sym(symbol, self.lstm_models)
        if key is None:
            return 0.5

        try:
            model  = self.lstm_models[key]
            scaler = self.lstm_scalers[key]

            feature_cols = [c for c in df_processed.columns if c != "Target"]
            scaled = scaler.transform(df_processed[feature_cols].values)

            # Apply feature selection — the scaler stores selected_indices_
            # from training (SelectKBest: 106 → 25 features)
            selected_indices = getattr(scaler, 'selected_indices_', None)
            if selected_indices is not None:
                scaled = scaled[:, selected_indices]

            if len(scaled) < sequence_length:
                return 0.5

            n_features = scaled.shape[1]
            seq = scaled[-sequence_length:]
            seq = seq.reshape(1, sequence_length, n_features)
            prob = float(model.predict(seq, verbose=0)[0][0])
            return prob

        except Exception as e:
            logger.error("[Registry] LSTM predict error for %s: %s", symbol, e)
            return 0.5

    def predict_rf(self, symbol: str, df_processed) -> float:
        """
        Returns RF probability ∈ [0,1] for the given symbol.
        Falls back to 0.5 if model unavailable.
        """
        key = self._resolve_sym(symbol, self.rf_models)
        if key is None:
            return 0.5

        try:
            from rf_model import engineer_rf_features
            model    = self.rf_models[key]
            scaler   = self.rf_scalers[key]
            features = self.rf_features[key]

            rf_df = engineer_rf_features(df_processed)
            rf_df = rf_df.drop("Target", axis=1, errors="ignore")
            latest = rf_df.iloc[-1:].fillna(0)

            for col in features:
                if col not in latest.columns:
                    latest[col] = 0.0
            latest = latest[features]

            X = scaler.transform(latest.values)
            proba = model.predict_proba(X)[0]
            return float(proba[1]) if len(proba) > 1 else 0.5

        except Exception as e:
            logger.error("[Registry] RF predict error for %s: %s", symbol, e)
            return 0.5

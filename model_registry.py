"""
model_registry.py — Elite Trading Bot v3.2
===========================================
Loads and serves per-symbol LSTM + RF models.
Falls back to universal model if symbol model missing.

v3.2 Change:
  predict_lstm() now applies feature selection (selected_indices_)
  that was saved inside the scaler by lstm_model.py v3.2.
  Backward compatible — if scaler has no selected_indices_, uses all features.

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

def _lstm_path(sym):       return f"lstm_model_{sym}.h5"
def _scaler_path(sym):     return f"lstm_scaler_{sym}.joblib"
def _rf_path(sym):         return f"rf_model_{sym}.joblib"
def _rf_scaler_path(sym):  return f"rf_scaler_{sym}.joblib"
def _rf_feat_path(sym):    return f"rf_features_{sym}.joblib"

# Legacy single-model paths (backward compat)
_LEGACY_LSTM_PATH   = "lstm_model.h5"
_LEGACY_SCALER_PATH = "lstm_scaler.joblib"
_LEGACY_RF_PATH     = "rf_model.joblib"
_LEGACY_RF_SCALER   = "rf_scaler.joblib"
_LEGACY_RF_FEAT     = "rf_features.joblib"


class ModelRegistry:

    def __init__(self):
        self.lstm_models  = {}   # symbol → keras model
        self.lstm_scalers = {}   # symbol → scaler (may have .selected_indices_)
        self.rf_models    = {}   # symbol → RF model
        self.rf_scalers   = {}   # symbol → scaler
        self.rf_features  = {}   # symbol → feature list
        self._load_all()

    def _load_all(self):
        try:
            from tensorflow.keras.models import load_model as keras_load
        except ImportError:
            keras_load = None

        # ── Per-symbol + universal ──────────────────────────────
        for sym in SYMBOLS + ["universal"]:
            # LSTM
            lp = _lstm_path(sym)
            sp = _scaler_path(sym)
            if keras_load and os.path.exists(lp) and os.path.exists(sp):
                try:
                    self.lstm_models[sym]  = keras_load(lp)
                    self.lstm_scalers[sym] = load(sp)
                    has_sel = hasattr(self.lstm_scalers[sym], 'selected_indices_')
                    logger.info("[Registry] Loaded LSTM for %s (feature_selection=%s)",
                                sym, has_sel)
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

        # ── Legacy fallback (lstm_model.h5 / rf_model.joblib) ──
        # يُستخدم لو مفيش أي per-symbol models (قبل ما يتدرب)
        if keras_load and not self.lstm_models:
            if os.path.exists(_LEGACY_LSTM_PATH) and os.path.exists(_LEGACY_SCALER_PATH):
                try:
                    self.lstm_models["__legacy__"]  = keras_load(_LEGACY_LSTM_PATH)
                    self.lstm_scalers["__legacy__"] = load(_LEGACY_SCALER_PATH)
                    logger.info("[Registry] Loaded LEGACY LSTM as fallback.")
                except Exception as e:
                    logger.warning("[Registry] Legacy LSTM load failed: %s", e)

        if not self.rf_models:
            if os.path.exists(_LEGACY_RF_PATH) and os.path.exists(_LEGACY_RF_SCALER):
                try:
                    self.rf_models["__legacy__"]   = load(_LEGACY_RF_PATH)
                    self.rf_scalers["__legacy__"]  = load(_LEGACY_RF_SCALER)
                    self.rf_features["__legacy__"] = (
                        load(_LEGACY_RF_FEAT) if os.path.exists(_LEGACY_RF_FEAT) else []
                    )
                    logger.info("[Registry] Loaded LEGACY RF as fallback.")
                except Exception as e:
                    logger.warning("[Registry] Legacy RF load failed: %s", e)

        logger.info("[Registry] LSTM models available: %s", list(self.lstm_models.keys()))
        logger.info("[Registry] RF models available:   %s", list(self.rf_models.keys()))

    def _resolve_sym(self, symbol, model_dict):
        """
        3-tier resolution:
        1. Per-symbol model  (e.g. "XAUUSD")
        2. Universal model   ("universal")
        3. Legacy model      ("__legacy__")
        """
        if symbol in model_dict:
            return symbol
        if "universal" in model_dict:
            logger.debug("[Registry] %s → universal fallback", symbol)
            return "universal"
        if "__legacy__" in model_dict:
            logger.debug("[Registry] %s → legacy fallback", symbol)
            return "__legacy__"
        return None

    def has_model(self, symbol: str) -> dict:
        """Returns which models are available for a symbol."""
        return {
            "lstm": self._resolve_sym(symbol, self.lstm_models),
            "rf":   self._resolve_sym(symbol, self.rf_models),
        }

    # ─────────────────────────────────────────
    # LSTM PREDICTION
    # v3.2: applies selected_indices_ if present
    # ─────────────────────────────────────────
    def predict_lstm(self, symbol: str, df_processed, sequence_length=60) -> float:
        """
        Returns LSTM probability ∈ [0,1] for the given symbol.

        v3.2 feature selection:
          If the scaler has .selected_indices_ (set by lstm_model.py v3.2),
          only those columns are passed to the model.
          If not (old model), all features are used — backward compatible.
        """
        key = self._resolve_sym(symbol, self.lstm_models)
        if key is None:
            return 0.5

        try:
            model  = self.lstm_models[key]
            scaler = self.lstm_scalers[key]

            feature_cols = [c for c in df_processed.columns if c != "Target"]
            scaled = scaler.transform(df_processed[feature_cols].values)

            if len(scaled) < sequence_length:
                return 0.5

            # ── v3.2: Apply feature selection if available ──────
            selected_indices = getattr(scaler, 'selected_indices_', None)
            if selected_indices is not None:
                scaled = scaled[:, selected_indices]
            # ────────────────────────────────────────────────────

            seq = scaled[-sequence_length:]
            n_features = seq.shape[1]
            seq = seq.reshape(1, sequence_length, n_features)

            prob = float(model.predict(seq, verbose=0)[0][0])
            return prob

        except Exception as e:
            logger.error("[Registry] LSTM predict error for %s: %s", symbol, e)
            return 0.5

    # ─────────────────────────────────────────
    # RF PREDICTION (unchanged from v3.1)
    # ─────────────────────────────────────────
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

            # Align columns with training features
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

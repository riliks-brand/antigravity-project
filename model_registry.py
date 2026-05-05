"""
model_registry.py — Elite Trading Bot v4.0 (XGBoost Edition)
=============================================================
Loads and serves per-symbol XGBoost + RF models.
Falls back to universal model if symbol model missing.

Usage in main.py:
    from model_registry import ModelRegistry
    registry = ModelRegistry()
    xgb_prob = registry.predict_xgb("XAUUSD", df_processed)
    rf_prob   = registry.predict_rf("XAUUSD", df_processed)
"""

import os
import numpy as np
import logging
from joblib import load

logger = logging.getLogger("ModelRegistry")

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US30", "BTCUSD"]

def _xgb_path(sym):       return f"xgb_model_{sym}.joblib"
def _xgb_scaler_path(sym): return f"xgb_scaler_{sym}.joblib"
def _xgb_feat_path(sym):  return f"xgb_features_{sym}.joblib"
def _rf_path(sym):         return f"rf_model_{sym}.joblib"
def _rf_scaler_path(sym):  return f"rf_scaler_{sym}.joblib"
def _rf_feat_path(sym):    return f"rf_features_{sym}.joblib"


class ModelRegistry:

    def __init__(self):
        self.xgb_models   = {}   # symbol → XGBClassifier
        self.xgb_scalers  = {}   # symbol → scaler (with selected_indices_)
        self.xgb_features = {}   # symbol → feature list
        self.rf_models    = {}   # symbol → RF model
        self.rf_scalers   = {}   # symbol → scaler
        self.rf_features  = {}   # symbol → feature list
        self._load_all()

    def _load_all(self):
        for sym in SYMBOLS + ["universal"]:
            # XGBoost
            xp  = _xgb_path(sym)
            xsp = _xgb_scaler_path(sym)
            xfp = _xgb_feat_path(sym)
            if os.path.exists(xp) and os.path.exists(xsp) and os.path.exists(xfp):
                try:
                    self.xgb_models[sym]   = load(xp)
                    self.xgb_scalers[sym]  = load(xsp)
                    self.xgb_features[sym] = load(xfp)
                    logger.info("[Registry] Loaded XGB for %s", sym)
                except Exception as e:
                    logger.warning("[Registry] Failed to load XGB %s: %s", sym, e)

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

        logger.info("[Registry] Loaded XGB for: %s", list(self.xgb_models.keys()))
        logger.info("[Registry] Loaded RF for:   %s", list(self.rf_models.keys()))

    def _resolve_sym(self, symbol, model_dict):
        if symbol in model_dict:
            return symbol
        if "universal" in model_dict:
            logger.debug("[Registry] %s not found, using universal fallback.", symbol)
            return "universal"
        return None

    def predict_xgb(self, symbol: str, df_processed) -> float:
        """
        Returns XGBoost probability ∈ [0,1] for the given symbol.
        Falls back to 0.5 if model unavailable.
        """
        key = self._resolve_sym(symbol, self.xgb_models)
        if key is None:
            return 0.5

        try:
            from xgb_model import engineer_lagged_features
            model    = self.xgb_models[key]
            scaler   = self.xgb_scalers[key]
            features = self.xgb_features[key]

            df_lagged = engineer_lagged_features(df_processed)
            df_lagged = df_lagged.drop("Target", axis=1, errors="ignore")
            latest = df_lagged.iloc[-1:].fillna(0)

            all_cols = getattr(scaler, "all_feature_cols_", features)
            for col in all_cols:
                if col not in latest.columns:
                    latest[col] = 0.0

            selected_indices = getattr(scaler, "selected_indices_", None)
            if selected_indices is not None:
                X = latest[all_cols].values[:, selected_indices]
            else:
                for col in features:
                    if col not in latest.columns:
                        latest[col] = 0.0
                X = latest[features].values

            X_scaled = scaler.transform(X)
            proba = model.predict_proba(X_scaled)[0]
            return float(proba[1]) if len(proba) > 1 else 0.5

        except Exception as e:
            logger.error("[Registry] XGB predict error for %s: %s", symbol, e)
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

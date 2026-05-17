"""
train_offline.py — Elite Trading Bot v5.1
==========================================
MULTI-SYMBOL TRAINING EDITION — Large Dataset

التغيير الجوهري عن v3.2:
- القديم: 17,280 candle (~60 يوم) لكل symbol
- الجديد: 99,000 candle (~14 شهر) لكل symbol
  ده بيدي الموديل patterns أكتر بكتير ويحسن الـ accuracy بـ 3-5%

لماذا 99,000 وليس 100,000؟
- MT5 بيرجع max 99,000 M5 candle في request واحد
- M15 و H1 بيتحسبوا proportionally

التحسينات في v5.1:
1. زيادة البيانات: 17K → 99K candle (~6x أكبر)
2. XGBoost hyperparameters محسّنة للـ large dataset:
   - n_estimators: 500 → 1000 (أكتر trees = أدق)
   - early_stopping_rounds: 50 (يوقف لو مفيش تحسن)
   - learning_rate: 0.02 → 0.01 (أبطأ = أدق مع بيانات أكبر)
3. RF محسّن:
   - n_estimators: 200 → 500
   - min_samples_split: 10 → 20 (أقل overfitting مع بيانات أكبر)
4. Proportional MTF candles (M15 = M5/3, H1 = M5/12)
"""

import os
import sys
import numpy as np
import pandas as pd
import datetime
import logging
import MetaTrader5 as mt5
from joblib import dump, load
from pathlib import Path

from config import Config
from data_loader import fetch_mt5_ohlc
from features import feature_engineering_pipeline
from xgb_model import train_and_evaluate_xgb, engineer_lagged_features
from rf_model import RFModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────
# LOGGING — Rotating (preserves all lines)
# ─────────────────────────────────────────
from logging_setup import configure_root_logger
configure_root_logger(
    log_file="train_multisymbol.log",
    level=logging.INFO,
    fmt="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("TrainMultiSymbol")


# ─────────────────────────────────────────
# SYMBOL CONFIG
# كل symbol ليه إعدادات تدريب مختلفة
# بناءً على طبيعته وسرعة تحركه
# ─────────────────────────────────────────
SYMBOL_CONFIGS = {
    "EURUSD": {
        "m5_candles":  99_000,   # ~14 شهر — كل ما هو متاح في MT5
        "m15_candles": 33_000,   # proportional: M5/3
        "h1_candles":   8_250,   # proportional: M5/12
        "atr_lookahead_mult": 1.2,
        "description": "Baseline forex pair — stable"
    },
    "GBPUSD": {
        "m5_candles":  99_000,
        "m15_candles": 33_000,
        "h1_candles":   8_250,
        "atr_lookahead_mult": 1.2,
        "description": "GBP pair — slightly more volatile"
    },
    "USDJPY": {
        "m5_candles":  99_000,
        "m15_candles": 33_000,
        "h1_candles":   8_250,
        "atr_lookahead_mult": 1.2,
        "description": "JPY pair — range-bound tendency"
    },
    "XAUUSD": {
        "m5_candles":  99_000,
        "m15_candles": 33_000,
        "h1_candles":   8_250,
        "atr_lookahead_mult": 1.5,
        "description": "Gold — high volatility, $5-15/candle"
    },
    "US30": {
        "m5_candles":  99_000,
        "m15_candles": 33_000,
        "h1_candles":   8_250,
        "atr_lookahead_mult": 1.5,
        "description": "Dow Jones Index — news-driven"
    },
    "BTCUSD": {
        "m5_candles":  99_000,
        "m15_candles": 33_000,
        "h1_candles":   8_250,
        "atr_lookahead_mult": 1.8,
        "description": "Bitcoin — extreme volatility"
    },
}

# Symbols to train on (يمكن تغييرها وقت التشغيل)
TRAIN_SYMBOLS = list(SYMBOL_CONFIGS.keys())

# Model paths pattern: e.g. "xgb_model_EURUSD.joblib"
def xgb_path(symbol):          return f"xgb_model_{symbol}.joblib"
def xgb_scaler_path(symbol):   return f"xgb_scaler_{symbol}.joblib"
def xgb_features_path(symbol): return f"xgb_features_{symbol}.joblib"
def rf_path(symbol):            return f"rf_model_{symbol}.joblib"
def rf_scaler_path(symbol):     return f"rf_scaler_{symbol}.joblib"
def rf_features_path(symbol):   return f"rf_features_{symbol}.joblib"


# ─────────────────────────────────────────
# HELPER: Class Weights
# ─────────────────────────────────────────
def compute_class_weights(y):
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    weight_dict = dict(zip(classes, weights))
    sample_weights = np.zeros_like(y, dtype=float)
    for c, w in weight_dict.items():
        sample_weights[y == c] = w
    return sample_weights


# ─────────────────────────────────────────
# HELPER: Prediction Distribution Viewer
# ─────────────────────────────────────────
def visualize_predictions(y_pred, name="Model", symbol=""):
    bins = [0.0, 0.2, 0.4, 0.45, 0.55, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(y_pred, bins=bins)
    labels = [
        "Strong SELL (<0.2)",
        "Weak SELL   (0.2-0.4)",
        "Soft SELL   (0.4-0.45)",
        "NOISE       (0.45-0.55)",
        "Soft BUY    (0.55-0.6)",
        "Weak BUY    (0.6-0.8)",
        "Strong BUY  (>0.8)"
    ]
    print(f"\n{'='*50}")
    print(f"  {name} [{symbol}] Prediction Distribution")
    print(f"{'='*50}")
    max_count = max(hist) if max(hist) > 0 else 1
    for count, label in zip(hist, labels):
        bar = "█" * int(count / max_count * 35)
        print(f"  {label}: {count:5d}  {bar}")
    print(f"{'='*50}\n")


# ─────────────────────────────────────────
# CORE: Train Single Symbol
# ─────────────────────────────────────────
def train_symbol(symbol: str, sym_cfg: dict) -> dict:
    """
    Trains XGBoost + RF for a single symbol.
    Returns a result dict with accuracy and status.
    """
    logger.info("=" * 60)
    logger.info("Training: %s | %s", symbol, sym_cfg["description"])
    logger.info("Dataset: %d M5 candles (~%.0f months)",
                sym_cfg["m5_candles"], sym_cfg["m5_candles"] / (6 * 24 * 30))
    logger.info("=" * 60)

    result = {
        "symbol": symbol,
        "xgb_accuracy": 0.0,
        "rf_accuracy": 0.0,
        "train_rows": 0,
        "status": "FAILED",
        "error": None,
        "wfv_xgb": {},
        "wfv_rf": {},
    }

    try:
        # ── 1. Fetch Data ──────────────────────────────────────
        import time
        t0 = time.time()
        logger.info("[%s] Fetching M5 (%d candles)...", symbol, sym_cfg["m5_candles"])
        df_m5  = fetch_mt5_ohlc(symbol=symbol, timeframe=mt5.TIMEFRAME_M5,  count=sym_cfg["m5_candles"])
        df_m15 = fetch_mt5_ohlc(symbol=symbol, timeframe=mt5.TIMEFRAME_M15, count=sym_cfg["m15_candles"])
        df_h1  = fetch_mt5_ohlc(symbol=symbol, timeframe=mt5.TIMEFRAME_H1,  count=sym_cfg["h1_candles"])

        if df_m5 is None or df_m5.empty:
            raise ValueError(f"Failed to fetch M5 data for {symbol}. Symbol may not be available.")

        logger.info("[%s] Fetched %d M5 candles (%.1fs). Date range: %s to %s",
                    symbol, len(df_m5), time.time() - t0,
                    str(df_m5.index[0].date()), str(df_m5.index[-1].date()))

        # ── 2. Feature Engineering ─────────────────────────────
        # Temporarily override ATR lookahead multiplier per symbol
        original_atr_mult = getattr(Config, 'ATR_LOOKAHEAD_MULT', 1.2)
        Config.ATR_LOOKAHEAD_MULT = sym_cfg["atr_lookahead_mult"]

        logger.info("[%s] Running feature engineering pipeline...", symbol)
        t1 = time.time()
        df_processed = feature_engineering_pipeline(df_m5, df_confirm=df_m15, df_trend=df_h1, symbol=symbol)

        # Restore
        Config.ATR_LOOKAHEAD_MULT = original_atr_mult

        if df_processed.empty or len(df_processed) < 500:
            raise ValueError(f"Not enough data after feature engineering: {len(df_processed)} rows")

        result["train_rows"] = len(df_processed)
        target_counts = df_processed['Target'].value_counts()
        logger.info("[%s] Feature engineering done (%.1fs). Rows: %d | Labels: BUY=%d SELL=%d",
                    symbol, time.time() - t1, len(df_processed),
                    int(target_counts.get(1.0, 0)), int(target_counts.get(0.0, 0)))

        # ── 3. Train XGBoost ──────────────────────────────────
        logger.info("[%s] Training XGBoost (1000 trees, early stopping)...", symbol)
        t2 = time.time()
        xgb_model_obj, xgb_scaler, xgb_acc, xgb_features = train_and_evaluate_xgb(
            df_processed, symbol=symbol
        )

        if xgb_model_obj is None:
            raise ValueError(f"XGBoost training failed for {symbol}")

        # Save XGBoost
        from joblib import dump as jdump
        jdump(xgb_model_obj,  xgb_path(symbol))
        jdump(xgb_scaler,     xgb_scaler_path(symbol))
        jdump(xgb_features,   xgb_features_path(symbol))
        logger.info("[%s] XGB saved (acc: %.2f%% WFV, time: %.1fs)", symbol, xgb_acc * 100, time.time() - t2)

        # XGBoost prediction distribution
        from xgb_model import engineer_lagged_features
        df_lagged = engineer_lagged_features(df_processed).drop('Target', axis=1, errors='ignore')
        split_idx = int(len(df_lagged) * 0.8)
        df_test_xgb = df_lagged.iloc[split_idx:].fillna(0)
        xgb_selected = getattr(xgb_scaler, 'selected_indices_', None)
        all_cols = getattr(xgb_scaler, 'all_feature_cols_', xgb_features)
        for col in all_cols:
            if col not in df_test_xgb.columns:
                df_test_xgb[col] = 0.0
        if xgb_selected is not None:
            X_xgb_test = df_test_xgb[all_cols].values[:, xgb_selected]
        else:
            X_xgb_test = df_test_xgb[xgb_features].values
        X_xgb_test_s = xgb_scaler.transform(X_xgb_test)
        xgb_preds = xgb_model_obj.predict_proba(X_xgb_test_s)[:, 1]
        visualize_predictions(xgb_preds, "XGBoost", symbol)

        result["xgb_accuracy"] = round(xgb_acc * 100, 2)

        # ── 4. Train RF ────────────────────────────────────────
        logger.info("[%s] Training Random Forest (500 trees)...", symbol)
        t3 = time.time()

        # Create a per-symbol RF instance with custom save paths
        rf_engine = RFModelSymbol(symbol=symbol)
        rf_acc = rf_engine.train(df_processed)

        logger.info("[%s] RF saved (acc: %.2f%%, time: %.1fs)", symbol, rf_acc * 100, time.time() - t3)

        # RF prediction distribution
        feature_cols = [c for c in df_processed.columns if c != 'Target']
        split = int(len(df_processed) * 0.8)
        df_test = df_processed.iloc[split:]
        if not df_test.empty and rf_engine.model is not None:
            from rf_model import engineer_rf_features
            rf_df = engineer_rf_features(df_test).drop('Target', axis=1, errors='ignore').fillna(0)
            # align columns
            for col in rf_engine.feature_names:
                if col not in rf_df.columns:
                    rf_df[col] = 0.0
            rf_df = rf_df[rf_engine.feature_names]
            X_rf_test = rf_engine.scaler.transform(rf_df.values)
            rf_preds = rf_engine.model.predict_proba(X_rf_test)[:, 1]
            visualize_predictions(rf_preds, "Random Forest", symbol)

        result["rf_accuracy"] = round(rf_acc * 100, 2)
        result["status"] = "SUCCESS"

    except Exception as e:
        logger.error("[%s] ❌ Training failed: %s", symbol, e, exc_info=True)
        result["error"] = str(e)

    return result


# ─────────────────────────────────────────
# Per-Symbol RF Model
# نفس الـ RFModel بس بـ custom file paths
# ─────────────────────────────────────────
class RFModelSymbol:
    """RF model that saves to per-symbol paths."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.feature_importances = {}
        self.wfv_result = {}

        from sklearn.preprocessing import RobustScaler as RS
        self.scaler = RS()

    def _walk_forward_validate(self, X_all, y_all, n_folds=5):
        """
        Walk-Forward Validation للـ RF.
        نفس المنطق بتاع XGB لكن بـ RF model.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score as acc_score

        n = len(X_all)
        train_size = int(n * 0.60)
        test_size  = int(n * 0.10)
        step_size  = (n - train_size - test_size) // max(n_folds - 1, 1)

        fold_accuracies = []

        for fold in range(n_folds):
            train_start = fold * step_size
            train_end   = train_start + train_size
            test_start  = train_end
            test_end    = test_start + test_size

            if test_end > n:
                break

            X_tr = X_all[train_start:train_end]
            y_tr = y_all[train_start:train_end]
            X_te = X_all[test_start:test_end]
            y_te = y_all[test_start:test_end]

            sc = self.scaler.__class__()
            sc.fit(X_tr)
            X_tr_s = sc.transform(X_tr)
            X_te_s = sc.transform(X_te)

            # Lightweight RF for validation (fewer trees)
            fold_rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,                # v6.0: match final model
                min_samples_split=50,       # v6.0: match final model
                min_samples_leaf=25,        # v6.0: match final model
                max_features='log2',           # v6.0: match final model
                random_state=42 + fold,
                class_weght='balanced_subsample', # v6.0
                n_jobs=-1,
            )
            fold_rf.fit(X_tr_s, y_tr)
            fold_acc = acc_score(y_te, fold_rf.predict(X_te_s))
            fold_accuracies.append(fold_acc)

            logger.info(
                "[WFV-RF-%s] Fold %d/%d: acc=%.2f%%",
                self.symbol, fold + 1, n_folds, fold_acc * 100
            )

        if not fold_accuracies:
            return {"mean_accuracy": 0.0, "fold_accuracies": [], "skipped": True}

        mean_acc = float(np.mean(fold_accuracies))
        std_acc  = float(np.std(fold_accuracies))
        return {
            "fold_accuracies": [round(a * 100, 2) for a in fold_accuracies],
            "mean_accuracy":   round(mean_acc * 100, 2),
            "std_accuracy":    round(std_acc * 100, 2),
            "min_accuracy":    round(min(fold_accuracies) * 100, 2),
            "max_accuracy":    round(max(fold_accuracies) * 100, 2),
            "skipped":         False,
        }

    def train(self, df_full: pd.DataFrame) -> float:
        from rf_model import engineer_rf_features
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        rf_df = engineer_rf_features(df_full).dropna()

        if len(rf_df) < 100:
            logger.warning("[RF-%s] Not enough data.", self.symbol)
            return 0.0

        feature_cols = [c for c in rf_df.columns if c != 'Target']
        self.feature_names = feature_cols

        X = rf_df[feature_cols].values
        y = rf_df['Target'].values

        # ── Walk-Forward Validation ───────────────────────────
        logger.info("[RF-%s] Running Walk-Forward Validation (5 folds)...", self.symbol)
        self.wfv_result = self._walk_forward_validate(X, y, n_folds=5)
        if not self.wfv_result.get("skipped"):
            folds_str = " | ".join([f"{a:.1f}%" for a in self.wfv_result["fold_accuracies"]])
            print(f"  [RF-WFV-{self.symbol}] Folds: {folds_str}")
            print(f"  [RF-WFV-{self.symbol}] Mean: {self.wfv_result['mean_accuracy']:.2f}% | Std: {self.wfv_result['std_accuracy']:.2f}%")

        # ── Final Model Training (last 80% = most recent data) ─
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=6,                    # v6.0: shallower trees reduce noise memorization
            min_samples_split=50,           # v6.0: raised from 20
            min_samples_leaf=25,            # v6.0: raised from 10
            max_features='log2',            # v6.0: more feature diversity
            class_weight='balanced_subsample', # v6.0: per-tree balancing
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train_s, y_train)

        static_acc = accuracy_score(y_test, self.model.predict(X_test_s))

        # Save with symbol-specific paths
        dump(self.model,         rf_path(self.symbol))
        dump(self.scaler,        rf_scaler_path(self.symbol))
        dump(self.feature_names, rf_features_path(self.symbol))

        importances = self.model.feature_importances_
        self.feature_importances = dict(zip(self.feature_names, importances))
        top5 = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("[RF-%s] Top 5 features: %s", self.symbol, top5)

        # Return WFV mean if available, else static
        if not self.wfv_result.get("skipped"):
            return self.wfv_result["mean_accuracy"] / 100.0
        return static_acc


# ─────────────────────────────────────────
# UNIVERSAL MODEL
# نموذج مشترك من كل الـ symbols = fallback
# ─────────────────────────────────────────
def train_universal_model(all_dataframes: dict):
    """
    يجمع بيانات كل الـ symbols في نموذج واحد.
    ده بيشتغل كـ fallback لو symbol معينة مش عندها model.
    بيضيف feature اسمه 'symbol_id' عشان الموديل يعرف يفرق.
    """
    logger.info("=" * 60)
    logger.info("🌍 Training UNIVERSAL model (all symbols combined)...")
    logger.info("=" * 60)

    # Symbol encoding
    symbol_ids = {sym: i for i, sym in enumerate(TRAIN_SYMBOLS)}

    dfs = []
    for symbol, df in all_dataframes.items():
        if df is None or df.empty:
            continue
        df_copy = df.copy()
        # Normalize price features so symbols are comparable
        # الـ LSTM بيشوف prices مختلفة تماماً بين BTC وEURUSD
        # الحل: نحول كل شيء لـ % change بدل absolute price
        for col in ['open', 'high', 'low', 'close']:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].pct_change().fillna(0)

        # Add symbol identity feature
        df_copy['symbol_id'] = symbol_ids[symbol] / len(symbol_ids)  # normalize 0→1

        dfs.append(df_copy)
        logger.info("  Added %s: %d rows", symbol, len(df_copy))

    if not dfs:
        logger.error("No data available for universal model training.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    # Shuffle (بالنسبة للـ universal model، الـ time order أقل أهمية)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info("Universal combined dataset: %d rows", len(combined))

    # Train RF universal
    try:
        rf_univ = RFModelSymbol(symbol="universal")
        rf_acc = rf_univ.train(combined)
        logger.info("✅ Universal RF saved (acc: %.2f%%)", rf_acc * 100)
    except Exception as e:
        logger.error("Universal RF training failed: %s", e)


# ─────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────
def print_summary(results: list):
    print("\n")
    print("=" * 68)
    print("     MULTI-SYMBOL TRAINING SUMMARY v5.2 (Walk-Forward)")
    print("=" * 68)
    print(f"  {'Symbol':<10} {'XGB WFV':>10} {'RF WFV':>10} {'Rows':>8}  {'Status':<10}")
    print("=" * 68)

    total_xgb = 0
    total_rf   = 0
    success_count = 0

    for r in results:
        status_icon = "OK" if r["status"] == "SUCCESS" else ("--" if r["status"] == "SKIPPED" else "XX")
        print(f"  {r['symbol']:<10} {r['xgb_accuracy']:>9.1f}% {r['rf_accuracy']:>9.1f}% "
              f"{r['train_rows']:>8,}  {status_icon} {r['status']:<8}")
        if r["status"] == "SUCCESS":
            total_xgb += r["xgb_accuracy"]
            total_rf   += r["rf_accuracy"]
            success_count += 1

    print("=" * 68)
    if success_count > 0:
        print(f"  {'AVERAGE':<10} {total_xgb/success_count:>9.1f}% {total_rf/success_count:>9.1f}%"
              f"{'':>9}  {success_count}/{len(results)} OK")
    print("=" * 68)

    # Show WFV details per symbol
    print("\n  Walk-Forward Validation Details:")
    for r in results:
        if r["status"] == "SUCCESS" and r.get("wfv_xgb"):
            wfv = r["wfv_xgb"]
            folds = " | ".join([f"{a:.1f}%" for a in wfv.get("fold_accuracies", [])])
            print(f"  {r['symbol']} XGB: [{folds}] mean={wfv.get('mean_accuracy',0):.1f}% std={wfv.get('std_accuracy',0):.1f}%")

    # Show failed ones
    failed = [r for r in results if r["status"] not in ("SUCCESS", "SKIPPED")]
    if failed:
        print("\n  Failed symbols:")
        for r in failed:
            print(f"   - {r['symbol']}: {r['error']}")

    print(f"\n  Model files saved:")
    for r in results:
        if r["status"] == "SUCCESS":
            print(f"   - xgb_model_{r['symbol']}.joblib + xgb_scaler_{r['symbol']}.joblib")
            print(f"   - rf_model_{r['symbol']}.joblib  + rf_scaler_{r['symbol']}.joblib")

    print(f"\n  Universal fallback: rf_model_universal.joblib\n")


# ─────────────────────────────────────────
# INTEGRATION GUIDE
# كيف يعدل main.py عشان يستخدم الـ per-symbol models
# ─────────────────────────────────────────
INTEGRATION_NOTES = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO INTEGRATE IN main.py / ensemble_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ ALREADY DONE (v4.0):
   ─────────────────────────────────────────
   from model_registry import ModelRegistry
   registry = ModelRegistry()  # بيلود كل الـ models

   # في evaluation loop:
   xgb_prob = registry.predict_xgb(symbol, df_processed)
   rf_prob   = registry.predict_rf(symbol, df_processed)
   ─────────────────────────────────────────

   الـ ModelRegistry موجود في ملف model_registry.py
   اللي بيتعمل regenerate مع كل training run.
"""


# ─────────────────────────────────────────
# MODEL REGISTRY — يُشتغل في main.py
# ─────────────────────────────────────────
REGISTRY_CODE = '''"""
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
        logger.warning("[Registry] No model found for %s and no universal fallback. Will return 0.5.", symbol)
        return None

    def has_model(self, symbol: str) -> bool:
        """Returns True if XGB model exists for this symbol (or universal fallback)."""
        return self._resolve_sym(symbol, self.xgb_models) is not None

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
'''


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    import time
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("ELITE BOT v5.1 — LARGE DATASET TRAINING")
    logger.info("Dataset: 99,000 M5 candles per symbol (~14 months)")
    logger.info("XGBoost: 1000 trees + early stopping")
    logger.info("RF: 500 trees")
    logger.info("=" * 60)

    # ── Init MT5 ─────────────────────────────────
    logger.info("Initializing MT5...")
    if not mt5.initialize():
        logger.error("MT5 init failed. Make sure terminal is open.")
        return

    from config import Config
    if not mt5.login(Config.LOGIN, password=Config.PASSWORD, server=Config.SERVER):
        logger.error("MT5 login failed: %s", mt5.last_error())
        mt5.shutdown()
        return

    logger.info("MT5 connected. Starting training...")

    # ── Train each symbol ────────────────────────
    results = []
    all_dataframes = {}

    for symbol in TRAIN_SYMBOLS:
        sym_cfg = SYMBOL_CONFIGS[symbol]

        # Skip BTCUSD if not available (common on demo accounts)
        mt5.symbol_select(symbol, True)
        test_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 10)
        if test_rates is None or len(test_rates) == 0:
            logger.warning("[%s] Symbol not available in MT5 — skipping.", symbol)
            results.append({
                "symbol": symbol, "xgb_accuracy": 0.0, "rf_accuracy": 0.0,
                "train_rows": 0, "status": "SKIPPED", "error": "Symbol not available"
            })
            continue

        result = train_symbol(symbol, sym_cfg)
        results.append(result)

        # Collect for universal model
        if result["status"] == "SUCCESS":
            try:
                df_m5 = fetch_mt5_ohlc(symbol=symbol, timeframe=mt5.TIMEFRAME_M5,
                                        count=min(sym_cfg["m5_candles"], 20_000))
                if df_m5 is not None and not df_m5.empty:
                    df_proc = feature_engineering_pipeline(df_m5, symbol=symbol)
                    all_dataframes[symbol] = df_proc
            except Exception:
                pass

    # ── Train Universal Fallback ─────────────────
    if len(all_dataframes) >= 2:
        train_universal_model(all_dataframes)
    else:
        logger.warning("Not enough successful symbols for universal model.")

    # ── Save model_registry.py ────────────────────
    registry_path = "model_registry.py"
    with open(registry_path, "w", encoding="utf-8") as f:
        f.write(REGISTRY_CODE)
    logger.info("Saved model_registry.py")

    # ── Print Summary ─────────────────────────────
    total_time = time.time() - t_start
    print_summary(results)
    print(INTEGRATION_NOTES)
    logger.info("Total training time: %.1f minutes", total_time / 60)

    mt5.shutdown()
    logger.info("Multi-symbol training complete!")


if __name__ == "__main__":
    main()






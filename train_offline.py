"""
train_offline.py — Elite Trading Bot v3.2
==========================================
MULTI-SYMBOL TRAINING EDITION

التغيير الجوهري عن النسخة القديمة:
- القديم: بيتدرب على EURUSD فقط (50k candle)
- الجديد: بيتدرب على كل الـ 6 symbols بشكل منفصل
  كل symbol بيطلع: lstm_model_{symbol}.h5 + rf_model_{symbol}.joblib
  + نموذج مشترك (universal) كـ fallback

لماذا نموذج منفصل لكل symbol؟
- XAUUSD (Gold): يتحرك $5-15 per candle، طبيعة مختلفة تماماً
- BTCUSD: Crypto volatility، cycles مختلفة
- US30: Index، مرتبط بالأحداث الأمريكية
- USDJPY: JPY pair، حساس جداً للـ interest rate
- GBPUSD: مرتبط بـ Brexit/UK data
- EURUSD: Baseline pair، أكثر استقراراً

كيف يشتغل الـ main.py معاه؟
- كل symbol بيلود الـ model الخاص بيه
- لو مفيش model للـ symbol → يستخدم universal model كـ fallback
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
from lstm_model import prepare_sequential_data, train_and_evaluate
from rf_model import RFModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train_multisymbol.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("TrainMultiSymbol")


# ─────────────────────────────────────────
# SYMBOL CONFIG
# كل symbol ليه إعدادات تدريب مختلفة
# بناءً على طبيعته وسرعة تحركه
# ─────────────────────────────────────────
SYMBOL_CONFIGS = {
    "EURUSD": {
        "m5_candles":  17_280,   # ~6 شهور (6×20×24×6) — أحدث داتا بس
        "m15_candles":  5_760,   # proportional
        "h1_candles":   2_880,
        "atr_lookahead_mult": 1.2,   # threshold عادي
        "description": "Baseline forex pair — stable"
    },
    "GBPUSD": {
        "m5_candles":  17_280,
        "m15_candles":  5_760,
        "h1_candles":   2_880,
        "atr_lookahead_mult": 1.2,
        "description": "GBP pair — slightly more volatile"
    },
    "USDJPY": {
        "m5_candles":  17_280,
        "m15_candles":  5_760,
        "h1_candles":   2_880,
        "atr_lookahead_mult": 1.2,
        "description": "JPY pair — range-bound tendency"
    },
    "XAUUSD": {
        "m5_candles":  17_280,
        "m15_candles":  5_760,
        "h1_candles":   2_880,
        "atr_lookahead_mult": 1.5,   # threshold أعلى — تجنب noise
        "description": "Gold — high volatility, $5-15/candle"
    },
    "US30": {
        "m5_candles":  17_280,
        "m15_candles":  5_760,
        "h1_candles":   2_880,
        "atr_lookahead_mult": 1.5,
        "description": "Dow Jones Index — news-driven"
    },
    "BTCUSD": {
        "m5_candles":  17_280,
        "m15_candles":  5_760,
        "h1_candles":   2_880,
        "atr_lookahead_mult": 1.8,   # Crypto noise كتير — threshold أعلى
        "description": "Bitcoin — extreme volatility"
    },
}

# Symbols to train on (يمكن تغييرها وقت التشغيل)
TRAIN_SYMBOLS = list(SYMBOL_CONFIGS.keys())

# Model paths pattern: e.g. "lstm_model_EURUSD.h5"
def lstm_path(symbol):     return f"lstm_model_{symbol}.h5"
def lstm_scaler_path(symbol): return f"lstm_scaler_{symbol}.joblib"
def rf_path(symbol):       return f"rf_model_{symbol}.joblib"
def rf_scaler_path(symbol):  return f"rf_scaler_{symbol}.joblib"
def rf_features_path(symbol): return f"rf_features_{symbol}.joblib"


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
    Trains LSTM + RF for a single symbol.
    Returns a result dict with accuracy and status.
    """
    logger.info("=" * 60)
    logger.info("🚀 Training: %s | %s", symbol, sym_cfg["description"])
    logger.info("=" * 60)

    result = {
        "symbol": symbol,
        "lstm_accuracy": 0.0,
        "rf_accuracy": 0.0,
        "train_rows": 0,
        "status": "FAILED",
        "error": None,
    }

    try:
        # ── 1. Fetch Data ──────────────────────────────────────
        logger.info("[%s] Fetching M5 (%d candles)...", symbol, sym_cfg["m5_candles"])
        df_m5  = fetch_mt5_ohlc(symbol=symbol, timeframe=mt5.TIMEFRAME_M5,  count=sym_cfg["m5_candles"])
        df_m15 = fetch_mt5_ohlc(symbol=symbol, timeframe=mt5.TIMEFRAME_M15, count=sym_cfg["m15_candles"])
        df_h1  = fetch_mt5_ohlc(symbol=symbol, timeframe=mt5.TIMEFRAME_H1,  count=sym_cfg["h1_candles"])

        if df_m5 is None or df_m5.empty:
            raise ValueError(f"Failed to fetch M5 data for {symbol}. Symbol may not be available.")

        logger.info("[%s] Fetched %d M5 candles.", symbol, len(df_m5))

        # ── 2. Feature Engineering ─────────────────────────────
        # Temporarily override ATR lookahead multiplier per symbol
        original_atr_mult = getattr(Config, 'ATR_LOOKAHEAD_MULT', 1.2)
        Config.ATR_LOOKAHEAD_MULT = sym_cfg["atr_lookahead_mult"]

        logger.info("[%s] Running feature engineering pipeline...", symbol)
        df_processed = feature_engineering_pipeline(df_m5, df_confirm=df_m15, df_trend=df_h1)

        # Restore
        Config.ATR_LOOKAHEAD_MULT = original_atr_mult

        if df_processed.empty or len(df_processed) < 500:
            raise ValueError(f"Not enough data after feature engineering: {len(df_processed)} rows")

        result["train_rows"] = len(df_processed)
        target_counts = df_processed['Target'].value_counts()
        logger.info("[%s] Label distribution:\n%s", symbol, target_counts.to_string())

        # ── 3. Train LSTM ──────────────────────────────────────
        logger.info("[%s] Training LSTM...", symbol)
        X_train, X_test, y_train, y_test, scaler, base_weights = prepare_sequential_data(df_processed)

        # Balanced class weights (not just loss penalty)
        train_sample_weights = compute_class_weights(y_train)

        lstm_model, history, lstm_acc = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            sample_weights=train_sample_weights
        )

        # Save LSTM
        lstm_model.save(lstm_path(symbol))
        dump(scaler, lstm_scaler_path(symbol))
        logger.info("[%s] ✅ LSTM saved → %s (acc: %.2f%%)", symbol, lstm_path(symbol), lstm_acc * 100)

        # LSTM prediction distribution
        lstm_preds = lstm_model.predict(X_test, verbose=0).flatten()
        visualize_predictions(lstm_preds, "LSTM", symbol)

        result["lstm_accuracy"] = round(lstm_acc * 100, 2)

        # ── 4. Train RF ────────────────────────────────────────
        logger.info("[%s] Training Random Forest...", symbol)

        # Create a per-symbol RF instance with custom save paths
        rf_engine = RFModelSymbol(symbol=symbol)
        rf_acc = rf_engine.train(df_processed)

        logger.info("[%s] ✅ RF saved → %s (acc: %.2f%%)", symbol, rf_path(symbol), rf_acc * 100)

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

        from sklearn.preprocessing import RobustScaler as RS
        self.scaler = RS()

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

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            n_estimators=Config.RF_N_ESTIMATORS,
            max_depth=Config.RF_MAX_DEPTH,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train_s, y_train)

        acc = accuracy_score(y_test, self.model.predict(X_test_s))

        # Save with symbol-specific paths
        dump(self.model,         rf_path(self.symbol))
        dump(self.scaler,        rf_scaler_path(self.symbol))
        dump(self.feature_names, rf_features_path(self.symbol))

        importances = self.model.feature_importances_
        self.feature_importances = dict(zip(self.feature_names, importances))
        top5 = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("[RF-%s] Top 5 features: %s", self.symbol, top5)

        return acc


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
    print("╔" + "═" * 62 + "╗")
    print("║     MULTI-SYMBOL TRAINING SUMMARY                           ║")
    print("╠" + "═" * 62 + "╣")
    print(f"║  {'Symbol':<10} {'LSTM Acc':>10} {'RF Acc':>10} {'Rows':>8}  {'Status':<10} ║")
    print("╠" + "═" * 62 + "╣")

    total_lstm = 0
    total_rf   = 0
    success_count = 0

    for r in results:
        status_icon = "✅" if r["status"] == "SUCCESS" else "❌"
        print(f"║  {r['symbol']:<10} {r['lstm_accuracy']:>9.1f}% {r['rf_accuracy']:>9.1f}% "
              f"{r['train_rows']:>8,}  {status_icon} {r['status']:<8} ║")
        if r["status"] == "SUCCESS":
            total_lstm += r["lstm_accuracy"]
            total_rf   += r["rf_accuracy"]
            success_count += 1

    print("╠" + "═" * 62 + "╣")
    if success_count > 0:
        print(f"║  {'AVERAGE':<10} {total_lstm/success_count:>9.1f}% {total_rf/success_count:>9.1f}%"
              f"{'':>9}  {success_count}/{len(results)} OK      ║")
    print("╚" + "═" * 62 + "╝")

    # Show failed ones
    failed = [r for r in results if r["status"] != "SUCCESS"]
    if failed:
        print("\n⚠️  Failed symbols:")
        for r in failed:
            print(f"   - {r['symbol']}: {r['error']}")

    print(f"\n📁 Model files saved:")
    for r in results:
        if r["status"] == "SUCCESS":
            print(f"   - {lstm_path(r['symbol'])} + {lstm_scaler_path(r['symbol'])}")
            print(f"   - {rf_path(r['symbol'])} + {rf_scaler_path(r['symbol'])}")

    print(f"\n📁 Universal fallback:")
    print(f"   - rf_model_universal.joblib (all symbols combined)\n")


# ─────────────────────────────────────────
# INTEGRATION GUIDE
# كيف يعدل main.py عشان يستخدم الـ per-symbol models
# ─────────────────────────────────────────
INTEGRATION_NOTES = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO INTEGRATE IN main.py / ensemble_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. في main.py، غير load الـ models من:
   ─────────────────────────────────────────
   lstm_model = load_model("lstm_model.h5")
   lstm_scaler = joblib.load("lstm_scaler.joblib")
   rf_engine = RFModel()
   ─────────────────────────────────────────
   إلى:
   ─────────────────────────────────────────
   from model_registry import ModelRegistry
   registry = ModelRegistry()  # بيلود كل الـ models

   # في evaluation loop:
   lstm_prob = registry.predict_lstm(symbol, df_processed)
   rf_prob   = registry.predict_rf(symbol, df_processed)
   ─────────────────────────────────────────

2. الـ ModelRegistry موجود في ملف model_registry.py
   اللي اتنشأ مع الـ training.
"""


# ─────────────────────────────────────────
# MODEL REGISTRY — يُشتغل في main.py
# ─────────────────────────────────────────
REGISTRY_CODE = '''"""
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

            if len(scaled) < sequence_length:
                return 0.5

            seq = scaled[-sequence_length:]
            seq = seq.reshape(1, sequence_length, len(feature_cols))
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
'''


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  ELITE BOT v3.2 — MULTI-SYMBOL TRAINING      ║")
    logger.info("╚══════════════════════════════════════════════╝")

    # ── Init MT5 ─────────────────────────────────
    logger.info("Initializing MT5...")
    if not mt5.initialize():
        logger.error("MT5 init failed. Make sure terminal is open.")
        return

    # ── Train each symbol ────────────────────────
    results = []
    all_dataframes = {}  # for universal model

    for symbol in TRAIN_SYMBOLS:
        sym_cfg = SYMBOL_CONFIGS[symbol]
        result  = train_symbol(symbol, sym_cfg)
        results.append(result)

        # Collect processed df for universal model
        if result["status"] == "SUCCESS":
            # Re-fetch for universal (we already cleaned up GPU memory)
            try:
                df_m5 = fetch_mt5_ohlc(symbol=symbol, timeframe=mt5.TIMEFRAME_M5,
                                        count=min(sym_cfg["m5_candles"], 20_000))  # أقل للـ universal
                if df_m5 is not None and not df_m5.empty:
                    df_proc = feature_engineering_pipeline(df_m5)
                    all_dataframes[symbol] = df_proc
            except Exception:
                pass  # universal model is a bonus, not critical

    # ── Train Universal Fallback ─────────────────
    if len(all_dataframes) >= 2:
        train_universal_model(all_dataframes)
    else:
        logger.warning("Not enough successful symbols for universal model.")

    # ── Save model_registry.py ────────────────────
    registry_path = "model_registry.py"
    with open(registry_path, "w", encoding="utf-8") as f:
        f.write(REGISTRY_CODE)
    logger.info("✅ Saved model_registry.py")

    # ── Print Summary ─────────────────────────────
    print_summary(results)
    print(INTEGRATION_NOTES)

    mt5.shutdown()
    logger.info("Multi-symbol training complete! 🎉")


if __name__ == "__main__":
    main()

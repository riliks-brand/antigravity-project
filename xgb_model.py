"""
XGBoost Model — Elite v5.3
============================
استبدال LSTM بـ XGBoost بناءً على قرار المسار C — Hybrid.

v5.3 Changes (SHAP + RFE + Probability Calibration):
  - SHAP-based feature selection (بدل SelectKBest)
  - RFE (Recursive Feature Elimination) للتأكيد
  - Isotonic Calibration: يصلح الـ BUY-heavy distribution
    السوق كان في uptrend → XGB بيتعلم BUY كـ default
    الـ calibration بتعمل re-mapping للـ probabilities
    عشان تكون موزعة بشكل طبيعي (BUY ≈ SELL ≈ NOISE)

v5.2 Changes (Walk-Forward Validation):
  - بدل static 80/20 split: rolling walk-forward validation
  - 5 folds × rolling window = accuracy estimate أدق بكتير
"""

import numpy as np
import pandas as pd
import os
import datetime
import logging
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.calibration import CalibratedClassifierCV
from config import Config

from logging_setup import setup_module_logger
logger = setup_module_logger("XGB_Model", Config.LOG_FILE, console_color="\033[96m")

# Model persistence paths (generic — يتغيروا لو per-symbol)
XGB_MODEL_PATH = "xgb_model.joblib"
XGB_SCALER_PATH = "xgb_scaler.joblib"
XGB_FEATURES_PATH = "xgb_features.joblib"

# عدد الـ features بعد الـ selection
# v5.1: 50 → 80 — مع 99K candle عندنا بيانات كافية لـ features أكتر
TOP_K_FEATURES = 80


# =========================================
# LAGGED FEATURE ENGINEERING
# XGBoost مش بيشوف sequences — نعوض بـ lag features
# =========================================

def engineer_lagged_features(df: pd.DataFrame, lags: list = None) -> pd.DataFrame:
    """
    يضيف lagged versions من الـ features المهمة.
    ده بيعوض عن انعدام الـ sequence memory في XGBoost.

    الـ strategy:
    - Lag 1, 3, 5 لأهم الـ indicators
    - Rolling stats (mean, std) على windows قصيرة
    - Delta (rate of change) بين consecutive candles
    """
    if lags is None:
        lags = [1, 3, 5]

    df_out = df.copy()
    new_features = {}

    # الـ features اللي بنعمل لها lag — أهم الـ indicators
    lag_cols = [
        'RSI', 'MACD', 'MACD_hist', 'ATR', 'ADX',
        'BB_position', 'close', 'Volatility',
        'DI_plus', 'DI_minus', 'EMA_50', 'EMA_200',
        'ROC_5', 'ROC_10', 'body_size', 'candle_range_atr',
    ]

    for col in lag_cols:
        if col not in df_out.columns:
            continue
        for lag in lags:
            new_features[f'{col}_lag{lag}'] = df_out[col].shift(lag)

        # Rolling stats على window 5
        new_features[f'{col}_roll5_mean'] = df_out[col].rolling(5).mean()
        new_features[f'{col}_roll5_std'] = df_out[col].rolling(5).std()

        # Delta (momentum)
        new_features[f'{col}_delta1'] = df_out[col].diff(1)
        new_features[f'{col}_delta3'] = df_out[col].diff(3)

    if new_features:
        new_df = pd.DataFrame(new_features, index=df_out.index)
        df_out = pd.concat([df_out, new_df], axis=1)

    logger.info("[XGB] Lagged feature engineering complete. Shape: %s", df_out.shape)
    return df_out


def prepare_tabular_data(df: pd.DataFrame):
    """
    يحضر البيانات لـ XGBoost (tabular بدل sequential).
    
    Pipeline:
    1. Engineer lagged features
    2. Drop NaN rows
    3. Chronological train/test split (80/20)
    4. Feature selection (SelectKBest)
    5. Scale features
    6. Return train/test splits + scaler + selected feature names
    """
    logger.info("[XGB] Preparing tabular data...")

    # Step 1: Add lagged features
    df_lagged = engineer_lagged_features(df)

    # Step 2: Drop rows without Target or with NaN features
    df_valid = df_lagged.dropna(subset=['Target'])
    feature_cols = [c for c in df_valid.columns if c != 'Target']
    df_valid = df_valid.dropna(subset=feature_cols)

    logger.info("[XGB] Valid rows after NaN drop: %d", len(df_valid))

    if len(df_valid) < 200:
        raise ValueError(f"Not enough data: {len(df_valid)} rows. Need at least 200.")

    X = df_valid[feature_cols].values
    y = df_valid['Target'].values.astype(int)

    # Remove constant features before selection (fixes sklearn UserWarning)
    col_std = X.std(axis=0)
    non_constant_mask = col_std > 0
    if not non_constant_mask.all():
        n_removed = (~non_constant_mask).sum()
        logger.info("[XGB] Removed %d constant features before selection.", n_removed)
        X = X[:, non_constant_mask]
        feature_cols = [f for f, keep in zip(feature_cols, non_constant_mask) if keep]

    # Step 3: Chronological split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    logger.info("[XGB] Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    # Step 4: Feature selection on training data only
    k = min(TOP_K_FEATURES, X_train.shape[1])
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X_train, y_train)
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_cols[i] for i in selected_indices]

    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]

    logger.info("[XGB] Feature selection: %d → %d features", X_train.shape[1], len(selected_features))

    # Step 5: Scale
    scaler = RobustScaler()
    scaler.fit(X_train_sel)
    X_train_scaled = scaler.transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    # نحتفظ بـ selected_indices في الـ scaler للاستخدام وقت الـ inference
    scaler.selected_indices_ = selected_indices
    scaler.selected_features_ = selected_features
    scaler.all_feature_cols_ = feature_cols

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, selected_features


# =========================================
# SHAP + RFE FEATURE SELECTION — v5.3
# بيحدد الـ features الحقيقية المؤثرة
# =========================================

def shap_feature_selection(
    model: XGBClassifier,
    X_train: np.ndarray,
    feature_names: list,
    top_k: int = None,
    shap_threshold: float = 0.001,
    symbol: str = "",
) -> tuple:
    """
    SHAP-based feature selection.

    Strategy:
    1. حساب SHAP values على الـ training data
    2. حساب mean(|SHAP|) لكل feature = الأهمية الحقيقية
    3. حذف الـ features اللي mean(|SHAP|) < threshold (noise)
    4. الاحتفاظ بـ top_k features الأعلى أهمية

    v5.3 fix: نستخدم balanced sample (50% BUY, 50% SELL) عشان
    نتجنب الـ SHAP bias ناحية الـ majority class.

    Returns:
        (selected_indices, selected_features, shap_importance_dict)
    """
    try:
        import shap

        # v5.3 fix: balanced sampling to avoid directional bias
        # نأخذ sample متوازن من الـ BUY و SELL predictions
        sample_size = min(len(X_train), 2000)

        # Get model predictions to identify BUY/SELL samples
        try:
            preds = model.predict(X_train)
            buy_idx  = np.where(preds == 1)[0]
            sell_idx = np.where(preds == 0)[0]
            half = sample_size // 2
            buy_sample  = buy_idx[np.random.choice(len(buy_idx),  min(half, len(buy_idx)),  replace=False)]
            sell_sample = sell_idx[np.random.choice(len(sell_idx), min(half, len(sell_idx)), replace=False)]
            balanced_idx = np.concatenate([buy_sample, sell_sample])
            np.random.shuffle(balanced_idx)
            X_sample = X_train[balanced_idx]
            logger.info("[SHAP-%s] Balanced sample: %d BUY + %d SELL = %d total",
                        symbol, len(buy_sample), len(sell_sample), len(balanced_idx))
        except Exception:
            # Fallback to random sample
            idx = np.random.choice(len(X_train), sample_size, replace=False)
            X_sample = X_train[idx]
            logger.info("[SHAP-%s] Random sample: %d samples", symbol, sample_size)

        logger.info("[SHAP-%s] Computing SHAP values on %d samples...", symbol, len(X_sample))

        # TreeExplainer — الأسرع والأدق لـ XGBoost
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # لو binary classification، shap_values ممكن يكون list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 (BUY)

        # mean(|SHAP|) لكل feature — absolute value يتجنب الـ directional bias
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Build importance dict
        shap_importance = {
            feat: float(imp)
            for feat, imp in zip(feature_names, mean_abs_shap)
        }

        # Sort by importance
        sorted_features = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)

        # Filter: remove features below threshold
        above_threshold = [(f, v) for f, v in sorted_features if v >= shap_threshold]

        # Apply top_k limit
        if top_k is not None:
            above_threshold = above_threshold[:top_k]

        selected_features = [f for f, _ in above_threshold]
        selected_indices = [feature_names.index(f) for f in selected_features]

        # Log top 10
        logger.info("[SHAP-%s] Top 10 features by SHAP importance:", symbol)
        for i, (feat, val) in enumerate(sorted_features[:10], 1):
            logger.info("[SHAP-%s]   %2d. %-35s %.6f", symbol, i, feat, val)

        removed = len(feature_names) - len(selected_features)
        logger.info(
            "[SHAP-%s] Feature selection: %d → %d (removed %d noise features, threshold=%.4f)",
            symbol, len(feature_names), len(selected_features), removed, shap_threshold
        )

        return selected_indices, selected_features, shap_importance

    except Exception as e:
        logger.warning("[SHAP-%s] SHAP selection failed: %s. Falling back to all features.", symbol, e)
        # Fallback: return all features
        return list(range(len(feature_names))), feature_names, {}


def rfe_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    n_features_to_select: int,
    symbol: str = "",
) -> tuple:
    """
    RFE (Recursive Feature Elimination) للتأكيد على SHAP.

    بيستخدم XGBoost خفيف (100 trees) عشان يكون سريع.
    بيحذف الـ features الأضعف واحدة واحدة.

    Returns:
        (selected_indices, selected_features)
    """
    try:
        logger.info("[RFE-%s] Running RFE: %d → %d features...", symbol, len(feature_names), n_features_to_select)

        # XGB خفيف للـ RFE
        rfe_model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        rfe = RFE(
            estimator=rfe_model,
            n_features_to_select=n_features_to_select,
            step=0.1,  # حذف 10% في كل iteration
            verbose=0,
        )
        rfe.fit(X_train, y_train)

        selected_mask = rfe.support_
        selected_features = [f for f, s in zip(feature_names, selected_mask) if s]
        selected_indices = [i for i, s in enumerate(selected_mask) if s]

        logger.info("[RFE-%s] RFE complete: %d features selected.", symbol, len(selected_features))
        return selected_indices, selected_features

    except Exception as e:
        logger.warning("[RFE-%s] RFE failed: %s. Skipping RFE.", symbol, e)
        return list(range(len(feature_names))), feature_names


def smart_feature_selection(
    model: XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    symbol: str = "",
) -> tuple:
    """
    v5.3: Smart Feature Selection = SHAP + RFE combined.

    Pipeline:
    1. SHAP: حذف الـ noise features (mean|SHAP| < threshold)
    2. من الـ SHAP survivors، نأخذ top 60
    3. RFE: من الـ 60، نختار أفضل 40
    4. النتيجة: 40 feature نظيفة ومؤثرة فعلاً

    لماذا 40 بدل 80؟
    - أقل features = أقل overfitting
    - الـ SHAP بيثبت إن معظم الـ 80 features بتضيف noise
    - 40 feature كافية لـ XGBoost مع 99K candle

    Returns:
        (selected_indices, selected_features, shap_importance)
    """
    logger.info("[SmartFS-%s] Starting Smart Feature Selection (SHAP + RFE)...", symbol)

    # Step 1: SHAP — حذف noise، الاحتفاظ بـ top 60
    shap_top_k = min(60, len(feature_names))
    shap_indices, shap_features, shap_importance = shap_feature_selection(
        model, X_train, feature_names,
        top_k=shap_top_k,
        shap_threshold=0.0005,
        symbol=symbol,
    )

    if len(shap_features) <= 20:
        # مفيش كفاية features بعد SHAP — نرجع الـ SHAP results مباشرة
        logger.info("[SmartFS-%s] Few features after SHAP (%d). Skipping RFE.", symbol, len(shap_features))
        return shap_indices, shap_features, shap_importance

    # Step 2: RFE على الـ SHAP survivors
    X_shap = X_train[:, shap_indices]
    rfe_target = min(40, len(shap_features))

    rfe_indices_local, rfe_features = rfe_feature_selection(
        X_shap, y_train, shap_features,
        n_features_to_select=rfe_target,
        symbol=symbol,
    )

    # Map back to original indices
    final_indices = [shap_indices[i] for i in rfe_indices_local]
    final_features = rfe_features

    logger.info(
        "[SmartFS-%s] Final: %d → SHAP(%d) → RFE(%d) features",
        symbol, len(feature_names), len(shap_features), len(final_features)
    )

    # Print top 10 final features with SHAP importance
    print(f"\n  Smart Feature Selection [{symbol}]")
    print(f"  {'─'*50}")
    print(f"  Total features  : {len(feature_names)}")
    print(f"  After SHAP      : {len(shap_features)}")
    print(f"  After RFE       : {len(final_features)}")
    print(f"  Top 10 selected :")
    for i, feat in enumerate(final_features[:10], 1):
        shap_val = shap_importance.get(feat, 0)
        print(f"    {i:2d}. {feat:<35} SHAP={shap_val:.6f}")
    print(f"  {'─'*50}")

    return final_indices, final_features, shap_importance


def calibrate_model(
    model: XGBClassifier,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    symbol: str = "",
) -> object:
    """
    Isotonic Calibration — يصلح الـ BUY-heavy probability distribution.

    المشكلة:
    - السوق كان في uptrend خلال الـ 14 شهر الأخيرة
    - XGBoost تعلم إن BUY هو الـ dominant pattern
    - النتيجة: 99% من الـ predictions > 0.6 (BUY zone)

    الحل:
    - Isotonic Regression: بتعمل monotonic mapping للـ probabilities
    - بتحول [0.6, 0.99] → [0.3, 0.7] بشكل proportional
    - بتحافظ على الـ relative ordering (أعلى prob = أقوى signal)
    - بتستخدم validation set منفصل (مش training data)

    لماذا Isotonic وليس Platt Scaling؟
    - Platt: linear sigmoid — مش كافي للـ heavy skew
    - Isotonic: non-parametric — بيتكيف مع أي distribution

    Returns:
        CalibratedClassifierCV wrapper around the model
    """
    logger.info("[CAL-%s] Calibrating probabilities on %d samples...", symbol, len(X_cal))

    # Check distribution before calibration
    raw_probs = model.predict_proba(X_cal)[:, 1]
    pct_above_60 = (raw_probs > 0.6).mean() * 100
    pct_below_40 = (raw_probs < 0.4).mean() * 100
    pct_noise = ((raw_probs >= 0.4) & (raw_probs <= 0.6)).mean() * 100

    logger.info(
        "[CAL-%s] Before calibration: BUY>0.6=%.1f%% | SELL<0.4=%.1f%% | NOISE=%.1f%%",
        symbol, pct_above_60, pct_below_40, pct_noise
    )

    # Apply isotonic calibration
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method='isotonic',
        cv='prefit',  # model already fitted — use X_cal as calibration set
    )
    calibrated.fit(X_cal, y_cal)

    # Check distribution after calibration
    cal_probs = calibrated.predict_proba(X_cal)[:, 1]
    pct_above_60_after = (cal_probs > 0.6).mean() * 100
    pct_below_40_after = (cal_probs < 0.4).mean() * 100
    pct_noise_after = ((cal_probs >= 0.4) & (cal_probs <= 0.6)).mean() * 100

    logger.info(
        "[CAL-%s] After calibration:  BUY>0.6=%.1f%% | SELL<0.4=%.1f%% | NOISE=%.1f%%",
        symbol, pct_above_60_after, pct_below_40_after, pct_noise_after
    )

    improvement = pct_noise_after - pct_noise
    logger.info("[CAL-%s] NOISE zone improvement: %+.1f%%", symbol, improvement)

    return calibrated


# =========================================
# WALK-FORWARD VALIDATION — v5.2
# Rolling window cross-validation للـ financial time series
# =========================================

def walk_forward_validate(
    df: pd.DataFrame,
    n_folds: int = 5,
    train_window_pct: float = 0.60,
    test_window_pct: float = 0.10,
    symbol: str = "",
) -> dict:
    """
    Walk-Forward Validation بـ Rolling Window.

    Strategy:
    - كل fold: train على window ثابتة الحجم، test على الـ window التالية
    - الـ window بتتحرك للأمام (rolling) — مش expanding
    - ده بيحاكي الـ real-world scenario: الموديل بيتدرب على بيانات حديثة

    مثال مع 99K rows، 5 folds، train=60%، test=10%:
      Fold 1: Train[0-59K]    → Test[59K-69K]
      Fold 2: Train[6K-65K]   → Test[65K-75K]
      Fold 3: Train[12K-71K]  → Test[71K-81K]
      Fold 4: Train[18K-77K]  → Test[77K-87K]
      Fold 5: Train[24K-83K]  → Test[83K-93K]

    Returns:
        dict with:
          - fold_accuracies: list of per-fold accuracy
          - mean_accuracy: average across folds
          - std_accuracy: standard deviation (consistency measure)
          - min_accuracy: worst fold (downside risk)
          - max_accuracy: best fold
          - stability_score: 1 - (std/mean) — higher = more consistent
    """
    logger.info("[WFV-%s] Starting Walk-Forward Validation (%d folds)...", symbol, n_folds)

    # Step 1: Prepare lagged features
    df_lagged = engineer_lagged_features(df)
    df_valid = df_lagged.dropna(subset=['Target'])
    feature_cols = [c for c in df_valid.columns if c != 'Target']
    df_valid = df_valid.dropna(subset=feature_cols)

    if len(df_valid) < 1000:
        logger.warning("[WFV-%s] Not enough data for walk-forward (%d rows). Skipping.", symbol, len(df_valid))
        return {"mean_accuracy": 0.0, "fold_accuracies": [], "skipped": True}

    X_all = df_valid[feature_cols].values
    y_all = df_valid['Target'].values.astype(int)
    n = len(X_all)

    train_size = int(n * train_window_pct)
    test_size  = int(n * test_window_pct)
    step_size  = (n - train_size - test_size) // max(n_folds - 1, 1)

    fold_accuracies = []

    for fold in range(n_folds):
        train_start = fold * step_size
        train_end   = train_start + train_size
        test_start  = train_end
        test_end    = test_start + test_size

        # Safety check
        if test_end > n:
            logger.debug("[WFV-%s] Fold %d: test_end=%d > n=%d, stopping.", symbol, fold + 1, test_end, n)
            break

        X_train = X_all[train_start:train_end]
        y_train = y_all[train_start:train_end]
        X_test  = X_all[test_start:test_end]
        y_test  = y_all[test_start:test_end]

        # Feature selection on this fold's train data only
        k = min(TOP_K_FEATURES, X_train.shape[1])
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X_train, y_train)
        sel_idx = selector.get_support(indices=True)

        X_tr_sel = X_train[:, sel_idx]
        X_te_sel = X_test[:, sel_idx]

        # Scale
        scaler = RobustScaler()
        scaler.fit(X_tr_sel)
        X_tr_s = scaler.transform(X_tr_sel)
        X_te_s = scaler.transform(X_te_sel)

        # Train a lightweight XGB for validation (fewer trees = faster)
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        spw = min(n_neg / max(n_pos, 1), 1.5)  # v6.0: cap at 1.5 to prevent extreme BUY bias

        fold_model = XGBClassifier(
            n_estimators=300,          # أقل من الـ final model — للسرعة
            max_depth=4,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=30,          # v6.0: raised from 20 - more regularization
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=spw,
            use_label_encoder=False,
            eval_metric='logloss',
            early_stopping_rounds=30,
            random_state=42 + fold,   # different seed per fold
            n_jobs=-1,
            verbosity=0,
        )
        fold_model.fit(
            X_tr_s, y_train,
            eval_set=[(X_te_s, y_test)],
            verbose=False,
        )

        y_pred = fold_model.predict(X_te_s)
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)

        logger.info(
            "[WFV-%s] Fold %d/%d: train[%d-%d] test[%d-%d] | acc=%.2f%%",
            symbol, fold + 1, n_folds,
            train_start, train_end, test_start, test_end,
            acc * 100
        )

    if not fold_accuracies:
        return {"mean_accuracy": 0.0, "fold_accuracies": [], "skipped": True}

    mean_acc = float(np.mean(fold_accuracies))
    std_acc  = float(np.std(fold_accuracies))
    stability = float(1.0 - (std_acc / mean_acc)) if mean_acc > 0 else 0.0

    result = {
        "fold_accuracies": [round(a * 100, 2) for a in fold_accuracies],
        "mean_accuracy":   round(mean_acc * 100, 2),
        "std_accuracy":    round(std_acc * 100, 2),
        "min_accuracy":    round(min(fold_accuracies) * 100, 2),
        "max_accuracy":    round(max(fold_accuracies) * 100, 2),
        "stability_score": round(stability * 100, 1),
        "n_folds":         len(fold_accuracies),
        "skipped":         False,
    }

    logger.info(
        "[WFV-%s] Results: mean=%.2f%% std=%.2f%% min=%.2f%% max=%.2f%% stability=%.1f%%",
        symbol, result["mean_accuracy"], result["std_accuracy"],
        result["min_accuracy"], result["max_accuracy"], result["stability_score"]
    )

    return result


def print_wfv_report(wfv_result: dict, symbol: str = ""):
    """Prints a formatted Walk-Forward Validation report."""
    if wfv_result.get("skipped"):
        print(f"  [WFV-{symbol}] Skipped (not enough data)")
        return

    folds_str = " | ".join([f"{a:.1f}%" for a in wfv_result["fold_accuracies"]])
    print(f"\n  Walk-Forward Validation [{symbol}]")
    print(f"  {'─'*45}")
    print(f"  Folds      : {folds_str}")
    print(f"  Mean       : {wfv_result['mean_accuracy']:.2f}%")
    print(f"  Std Dev    : {wfv_result['std_accuracy']:.2f}% (lower = more consistent)")
    print(f"  Range      : {wfv_result['min_accuracy']:.2f}% - {wfv_result['max_accuracy']:.2f}%")
    print(f"  Stability  : {wfv_result['stability_score']:.1f}% (higher = better)")

    # Interpretation
    mean = wfv_result["mean_accuracy"]
    std  = wfv_result["std_accuracy"]
    if mean >= 58 and std <= 3:
        verdict = "EXCELLENT — Deploy with confidence"
    elif mean >= 55 and std <= 5:
        verdict = "GOOD — Profitable with proper risk management"
    elif mean >= 52:
        verdict = "MARGINAL — Monitor closely, consider retraining"
    else:
        verdict = "POOR — Model needs improvement before deployment"
    print(f"  Verdict    : {verdict}")
    print(f"  {'─'*45}")


# =========================================
# MODEL
# =========================================

class XGBModel:
    """
    XGBoost model wrapper — بديل LSTM في الـ ensemble.
    
    Hyperparameters مُضبوطة لـ financial time series:
    - n_estimators=500: trees كافية بدون overfitting
    - max_depth=4: shallow trees = generalization أحسن
    - learning_rate=0.02: بطيء = learning أدق
    - subsample=0.8: row sampling = variance reduction
    - colsample_bytree=0.8: feature sampling = diversification
    - scale_pos_weight: handles class imbalance
    """

    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.feature_names = []
        self.feature_importances = {}
        self.last_train_time = None
        self.train_count = 0
        self.accuracy = 0.0

        self._load_model()

    def _load_model(self):
        if (os.path.exists(XGB_MODEL_PATH) and
                os.path.exists(XGB_SCALER_PATH) and
                os.path.exists(XGB_FEATURES_PATH)):
            try:
                self.model = joblib.load(XGB_MODEL_PATH)
                self.scaler = joblib.load(XGB_SCALER_PATH)
                self.feature_names = joblib.load(XGB_FEATURES_PATH)
                logger.info("[XGB] Loaded saved model (%d features).", len(self.feature_names))
            except Exception as e:
                logger.warning("[XGB] Failed to load saved model: %s", e)
                self.model = None

    def _save_model(self):
        try:
            joblib.dump(self.model, XGB_MODEL_PATH)
            joblib.dump(self.scaler, XGB_SCALER_PATH)
            joblib.dump(self.feature_names, XGB_FEATURES_PATH)
            logger.info("[XGB] Model saved to disk.")
        except Exception as e:
            logger.warning("[XGB] Failed to save model: %s", e)

    def needs_retraining(self, candle_count=0):
        if self.model is None:
            return True
        if self.last_train_time is None:
            return True
        hours_since = (datetime.datetime.utcnow() - self.last_train_time).total_seconds() / 3600
        if hours_since >= Config.RF_RETRAIN_EVERY_HOURS:
            logger.info("[XGB] Retraining triggered: %.1f hours since last train.", hours_since)
            return True
        if candle_count > 0 and (candle_count - self.train_count) >= Config.RF_RETRAIN_EVERY_CANDLES:
            logger.info("[XGB] Retraining triggered: %d candles since last train.",
                        candle_count - self.train_count)
            return True
        return False

    def train(self, df_full: pd.DataFrame) -> float:
        """
        Train XGBoost on tabular + lagged features.
        
        Returns:
            float: test set accuracy
        """
        logger.info("[XGB] Starting training...")

        try:
            X_train, X_test, y_train, y_test, scaler, selected_features = prepare_tabular_data(df_full)
        except ValueError as e:
            logger.warning("[XGB] %s", e)
            return 0.0

        self.scaler = scaler
        self.feature_names = selected_features

        # Class balance
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / max(n_pos, 1)

        self.model = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,    # regularization — يمنع overfitting على noise
            reg_alpha=0.1,          # L1 regularization
            reg_lambda=1.0,         # L2 regularization
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        # Early stopping على الـ test set
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)

        # Feature importance
        imp = self.model.feature_importances_
        self.feature_importances = dict(zip(self.feature_names, imp))
        top5 = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]

        # Report
        print(f"\n\033[96m{'='*55}\033[0m")
        print(f"\033[96m        XGBOOST MODEL REPORT\033[0m")
        print(f"\033[96m{'='*55}\033[0m")
        print(f"\033[96m  Accuracy       : {self.accuracy * 100:.2f}%\033[0m")
        print(f"\033[96m  Trees (final)  : {self.model.n_estimators}\033[0m")
        print(f"\033[96m  Max Depth      : {self.model.max_depth}\033[0m")
        print(f"\033[96m  Train Size     : {len(X_train)}\033[0m")
        print(f"\033[96m  Test Size      : {len(X_test)}\033[0m")
        print(f"\033[96m  Features       : {len(selected_features)}\033[0m")
        print(f"\033[96m  Top Feature    : {top5[0][0]} ({top5[0][1]:.4f})\033[0m")
        print(f"\033[96m{'='*55}\033[0m\n")

        logger.info("[XGB] ✅ Training complete. Accuracy: %.2f%%", self.accuracy * 100)
        logger.info("[XGB] Top 5 features:")
        for i, (feat, imp_val) in enumerate(top5, 1):
            logger.info("[XGB]   %d. %s: %.4f", i, feat, imp_val)

        self.last_train_time = datetime.datetime.utcnow()
        self._save_model()

        return self.accuracy

    def predict_proba(self, df_full: pd.DataFrame) -> float:
        """
        Returns XGB probability for the latest data point.
        
        Returns:
            float: probability of class 1 (BUY) ∈ [0, 1]
            Returns 0.5 if model not trained or error.
        """
        if self.model is None:
            logger.warning("[XGB] Model not trained. Returning neutral 0.5.")
            return 0.5

        try:
            # Engineer lagged features
            df_lagged = engineer_lagged_features(df_full)
            df_lagged = df_lagged.drop('Target', axis=1, errors='ignore')

            # Use latest row
            latest = df_lagged.iloc[-1:].copy().fillna(0)

            # Align to training features via stored indices
            # نحتاج كل الـ feature columns الأصلية أولاً
            all_cols = getattr(self.scaler, 'all_feature_cols_', self.feature_names)
            for col in all_cols:
                if col not in latest.columns:
                    latest[col] = 0.0

            # Apply feature selection
            selected_indices = getattr(self.scaler, 'selected_indices_', None)
            if selected_indices is not None:
                X_full = latest[all_cols].values
                X_sel = X_full[:, selected_indices]
            else:
                # Fallback: use feature_names directly
                for col in self.feature_names:
                    if col not in latest.columns:
                        latest[col] = 0.0
                X_sel = latest[self.feature_names].values

            X_scaled = self.scaler.transform(X_sel)
            proba = self.model.predict_proba(X_scaled)[0]
            return float(proba[1]) if len(proba) > 1 else 0.5

        except Exception as e:
            logger.error("[XGB] Prediction failed: %s", e)
            return 0.5

    def get_top_features(self, n=5):
        """Returns top N features by importance."""
        if not self.feature_importances:
            return []
        return sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:n]


# =========================================
# TRAIN & EVALUATE (للاستخدام في train_offline.py)
# نفس الـ interface الـ LSTM كان بيعرضه
# =========================================

def train_and_evaluate_xgb(df_full: pd.DataFrame, symbol: str = ""):
    """
    Interface بديل لـ train_and_evaluate من lstm_model.py.
    يُستدعى من train_offline.py بنفس الطريقة.

    v5.2: Walk-Forward Validation قبل الـ final training.
    الـ WFV بيدي accuracy estimate أدق من الـ static split.
    الـ final model بيتدرب على آخر 80% (الأحدث).

    Returns:
        (model, scaler, accuracy, selected_features)
        accuracy = walk-forward mean accuracy (أدق من static)
    """
    logger.info("[XGB-%s] Starting training with Walk-Forward Validation...", symbol)

    # ── Step 1: Walk-Forward Validation ──────────────────────
    # بيشتغل على كل البيانات عشان يقيس الـ real performance
    wfv_result = walk_forward_validate(df_full, n_folds=5, symbol=symbol)
    print_wfv_report(wfv_result, symbol)

    # ── Step 2: Final Model Training ─────────────────────────
    # بيتدرب على آخر 80% من البيانات (الأحدث = الأهم للـ live trading)
    try:
        X_train, X_test, y_train, y_test, scaler, selected_features = prepare_tabular_data(df_full)
    except ValueError as e:
        logger.error("[XGB-%s] %s", symbol, e)
        return None, None, 0.0, []

    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = min(n_neg / max(n_pos, 1), 1.5)  # v6.0: cap at 1.5

    # v6.0: Optimized hyperparameters + increased regularization to reduce BUY bias
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=30,       # v6.0: raised from 20 - more regularization
        reg_alpha=0.3,             # v6.0: raised from 0.1 - stronger L1
        reg_lambda=2.0,            # v6.0: raised from 1.0 - stronger L2
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    best_iter = getattr(model, 'best_iteration', model.n_estimators)

    # Static test accuracy (for reference)
    y_pred = model.predict(X_test)
    static_accuracy = accuracy_score(y_test, y_pred)

    # ── Step 3: SHAP + RFE Smart Feature Selection ───────────
    logger.info("[XGB-%s] Running SHAP + RFE feature selection...", symbol)
    smart_indices, smart_features, shap_importance = smart_feature_selection(
        model, X_train, y_train, selected_features, symbol=symbol
    )

    # ── Step 4: Retrain on smart features if reduced ─────────
    if len(smart_features) < len(selected_features):
        logger.info("[XGB-%s] Retraining on %d smart features (was %d)...",
                    symbol, len(smart_features), len(selected_features))

        X_train_smart = X_train[:, smart_indices]
        X_test_smart  = X_test[:, smart_indices]

        smart_scaler = RobustScaler()
        smart_scaler.fit(X_train_smart)
        X_tr_s = smart_scaler.transform(X_train_smart)
        X_te_s = smart_scaler.transform(X_test_smart)

        # Map smart_indices back to original all_feature_cols indices
        original_selected = scaler.selected_indices_
        final_original_indices = np.array([original_selected[i] for i in smart_indices])
        smart_scaler.selected_indices_ = final_original_indices
        smart_scaler.selected_features_ = smart_features
        smart_scaler.all_feature_cols_ = scaler.all_feature_cols_

        n_pos2 = np.sum(y_train == 1)
        n_neg2 = np.sum(y_train == 0)
        spw2 = min(n_neg2 / max(n_pos2, 1), 1.5)  # v6.0: cap at 1.5
        final_model = XGBClassifier(
            n_estimators=1000, max_depth=4, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=30,
            reg_alpha=0.3, reg_lambda=2.0,
            scale_pos_weight=spw2,
            use_label_encoder=False, eval_metric='logloss',
            early_stopping_rounds=50, random_state=42, n_jobs=-1, verbosity=0,
        )
        
        final_model.fit(X_tr_s, y_train, eval_set=[(X_te_s, y_test)], verbose=False)

        smart_acc = accuracy_score(y_test, final_model.predict(X_te_s))

        # Accept smart model if accuracy within 0.5% of original
        if smart_acc >= static_accuracy - 0.005:
            model = final_model
            scaler = smart_scaler
            selected_features = smart_features
            static_accuracy = smart_acc
            best_iter = getattr(final_model, 'best_iteration', final_model.n_estimators)
            logger.info("[XGB-%s] Smart model accepted: %.2f%%", symbol, smart_acc * 100)
        else:
            logger.info("[XGB-%s] Smart model rejected (%.2f%% < %.2f%%). Keeping original.",
                        symbol, smart_acc * 100, static_accuracy * 100)

    # Use WFV mean as the reported accuracy (more realistic)
    reported_accuracy = wfv_result["mean_accuracy"] / 100.0 if not wfv_result.get("skipped") else static_accuracy

    # Feature importance
    imp = model.feature_importances_
    feat_imp = dict(zip(selected_features, imp))
    top5 = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:5]

    print(f"\n  XGBOOST [{symbol}] FINAL TRAINING REPORT v5.3")
    print(f"  {'='*50}")
    print(f"  WFV Mean Accuracy  : {wfv_result.get('mean_accuracy', 0):.2f}% (realistic)")
    print(f"  Static Test Acc    : {static_accuracy * 100:.2f}% (reference)")
    print(f"  Best Iteration     : {best_iter} / 1000")
    print(f"  Features Selected  : {len(selected_features)} (SHAP+RFE filtered)")
    print(f"  Train Size         : {len(X_train):,}")
    print(f"  Test Size          : {len(X_test):,}")
    print(f"  Top Feature        : {top5[0][0]} ({top5[0][1]:.4f})")
    print(f"  {'='*50}")

    logger.info(
        "[XGB-%s] WFV=%.2f%% | Static=%.2f%% | Best iter=%d | Features=%d",
        symbol, wfv_result.get("mean_accuracy", 0), static_accuracy * 100,
        best_iter, len(selected_features)
    )
    logger.info("[XGB-%s] Top 5 features: %s", symbol, top5)



    # v6.0 FIX: Isotonic calibration to fix BUY-heavy distribution
    try:
        cal_model = calibrate_model(model, X_test, y_test, symbol=symbol)
        cal_probs = cal_model.predict_proba(X_test)[:, 1]
        pct_buy   = float((cal_probs > 0.6).mean() * 100)
        pct_sell  = float((cal_probs < 0.4).mean() * 100)
        pct_noise = float(((cal_probs >= 0.4) & (cal_probs <= 0.6)).mean() * 100)
        logger.info("[XGB-"+pct+s+"] Post-cal BUY="+pct+".1f"+pct+pct+" SELL="+pct+".1f"+pct+pct+" NOISE="+pct+".1f"+pct+pct, symbol, pct_buy, pct_sell, pct_noise)
        if pct_noise > 20.0:
            model = cal_model
            logger.info("[XGB-"+pct+s+"] Calibration ACCEPTED noise="+pct+".1f"+pct+pct, symbol, pct_noise)
        else:
            logger.warning("[XGB-"+pct+s+"] Cal skipped noise="+pct+".1f"+pct+pct, symbol, pct_noise)
    except Exception as cal_err:
        logger.warning("[XGB-"+pct+s+"] Cal failed: "+pct+s, symbol, cal_err)


    return model, scaler, reported_accuracy, selected_features





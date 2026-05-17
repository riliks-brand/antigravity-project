"""
Test Bug Exploration — XGBoost BUY Bias Detection
==================================================
هذا الاختبار يوثق الخلل الأصلي (95% BUY bias) قبل تطبيق الإصلاحات.

الهدف:
1. تشغيل XGBoost على البيانات الحقيقية بدون إصلاحات
2. قياس توزيع التنبؤات (BUY vs SELL vs NOISE)
3. توثيق الأمثلة المضادة (counterexamples)
4. إثبات أن الإصلاحات حلت المشكلة

الاستخدام:
    python test_bug_exploration.py --mode before  # قبل الإصلاحات
    python test_bug_exploration.py --mode after   # بعد الإصلاحات
"""

import numpy as np
import pandas as pd
import argparse
import logging
import sys
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('test_bug_exploration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_real_data(symbol: str):
    """
    تحميل البيانات الحقيقية من MT5 لرمز معين.
    
    Args:
        symbol: اسم الرمز (GBPUSD, XAUUSD, US30, EURUSD, USDJPY)
    
    Returns:
        DataFrame مع الـ features والـ Target
    """
    logger.info(f"Loading real data for {symbol}...")
    
    try:
        # Import data loader and feature engineering
        from data_loader import fetch_mt5_ohlc, fetch_mtf_data
        from features import feature_engineering_pipeline
        import MetaTrader5 as mt5
        
        # Initialize MT5
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return None
        
        # Fetch multi-timeframe data
        mtf_data = fetch_mtf_data(symbol)
        if mtf_data is None or mtf_data['M5'].empty:
            logger.error(f"Failed to fetch data for {symbol}")
            mt5.shutdown()
            return None
        
        # Feature engineering
        df = feature_engineering_pipeline(
            mtf_data['M5'],
            df_confirm=mtf_data.get('M15'),
            df_trend=mtf_data.get('H1'),
            symbol=symbol
        )
        
        mt5.shutdown()
        
        logger.info(f"Loaded {len(df)} rows for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        return None


def train_xgb_before_fix(df: pd.DataFrame, symbol: str):
    """
    تدريب XGBoost بدون الإصلاحات (الكود القديم).
    
    هذا يحاكي السلوك القديم:
    - بدون Isotonic Calibration
    - scale_pos_weight غير محدود
    - regularization ضعيف
    
    Returns:
        (model, scaler, test_probs, y_test)
    """
    logger.info(f"[{symbol}] Training XGBoost BEFORE fixes...")
    
    # Prepare data (using old 80/20 split for comparison)
    from xgb_model import engineer_lagged_features
    
    df_lagged = engineer_lagged_features(df)
    df_valid = df_lagged.dropna(subset=['Target'])
    feature_cols = [c for c in df_valid.columns if c != 'Target']
    df_valid = df_valid.dropna(subset=feature_cols)
    
    X = df_valid[feature_cols].values
    y = df_valid['Target'].values.astype(int)
    
    # Remove constant features
    col_std = X.std(axis=0)
    non_constant_mask = col_std > 0
    X = X[:, non_constant_mask]
    feature_cols = [f for f, keep in zip(feature_cols, non_constant_mask) if keep]
    
    # OLD SPLIT: 80/20 (no separate calibration set)
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    # Feature selection
    k = min(80, X_train.shape[1])
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X_train, y_train)
    selected_indices = selector.get_support(indices=True)
    
    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]
    
    # Scale
    scaler = RobustScaler()
    scaler.fit(X_train_sel)
    X_train_scaled = scaler.transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    
    # OLD HYPERPARAMETERS (before v6.0 fixes)
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = n_neg / max(n_pos, 1)  # NO CAP (old behavior)
    
    model = XGBClassifier(
        n_estimators=500,           # OLD: 500 trees
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,        # OLD: 20 (weaker regularization)
        reg_alpha=0.1,              # OLD: 0.1 (weaker L1)
        reg_lambda=1.0,             # OLD: 1.0 (weaker L2)
        scale_pos_weight=scale_pos_weight,  # NO CAP
        max_delta_step=0,           # OLD: 0 (no limit on predictions)
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    
    model.fit(X_train_scaled, y_train, verbose=False)
    
    # NO CALIBRATION (old behavior)
    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    
    logger.info(f"[{symbol}] Training complete (BEFORE fixes)")
    return model, scaler, test_probs, y_test


def train_xgb_after_fix(df: pd.DataFrame, symbol: str):
    """
    تدريب XGBoost بعد الإصلاحات (الكود الجديد).
    
    الإصلاحات المطبقة:
    - Isotonic Calibration على calibration set منفصل
    - scale_pos_weight محدود عند 1.2
    - regularization أقوى (min_child_weight=30, reg_lambda=1.5)
    - max_delta_step=1
    
    Returns:
        (model, scaler, test_probs, y_test)
    """
    logger.info(f"[{symbol}] Training XGBoost AFTER fixes...")
    
    from xgb_model import prepare_tabular_data, calibrate_model
    
    # NEW: 60/20/20 split with separate calibration set
    X_train, X_cal, X_test, y_train, y_cal, y_test, scaler, selected_features = prepare_tabular_data(df)
    
    # NEW HYPERPARAMETERS (v6.0 fixes)
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = min(n_neg / max(n_pos, 1), 1.2)  # CAPPED at 1.2
    
    model = XGBClassifier(
        n_estimators=300,           # NEW: 300 trees (aligned with WFV)
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=30,        # NEW: 30 (stronger regularization)
        reg_alpha=0.2,              # NEW: 0.2 (stronger L1)
        reg_lambda=1.5,             # NEW: 1.5 (stronger L2)
        scale_pos_weight=scale_pos_weight,  # CAPPED
        max_delta_step=1,           # NEW: 1 (limits extreme predictions)
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # NEW: Isotonic Calibration on UNSEEN calibration set
    model = calibrate_model(model, X_cal, y_cal, symbol=symbol)
    
    test_probs = model.predict_proba(X_test)[:, 1]
    
    logger.info(f"[{symbol}] Training complete (AFTER fixes)")
    return model, scaler, test_probs, y_test


def analyze_distribution(probs: np.ndarray, y_true: np.ndarray, mode: str, symbol: str):
    """
    تحليل توزيع التنبؤات وتوثيق الأمثلة المضادة.
    
    Args:
        probs: احتماليات التنبؤ (0-1)
        y_true: القيم الحقيقية (0=SELL, 1=BUY)
        mode: "before" أو "after"
        symbol: اسم الرمز
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Distribution Analysis [{symbol}] - {mode.upper()} fixes")
    logger.info(f"{'='*60}")
    
    # Calculate distribution
    pct_buy = (probs > 0.6).mean() * 100
    pct_sell = (probs < 0.4).mean() * 100
    pct_noise = ((probs >= 0.4) & (probs <= 0.6)).mean() * 100
    
    logger.info(f"BUY zone (>0.6):     {pct_buy:.1f}%")
    logger.info(f"SELL zone (<0.4):    {pct_sell:.1f}%")
    logger.info(f"NOISE zone (0.4-0.6): {pct_noise:.1f}%")
    
    # Predictions based on thresholds
    y_pred = np.where(probs > 0.6, 1, np.where(probs < 0.4, 0, -1))
    
    # Filter out NOISE predictions for accuracy calculation
    valid_mask = y_pred != -1
    if valid_mask.sum() > 0:
        acc = accuracy_score(y_true[valid_mask], y_pred[valid_mask])
        logger.info(f"Accuracy (excluding NOISE): {acc*100:.2f}%")
        
        # Confusion matrix
        cm = confusion_matrix(y_true[valid_mask], y_pred[valid_mask])
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"              Predicted")
        logger.info(f"              SELL  BUY")
        logger.info(f"Actual SELL   {cm[0,0]:4d}  {cm[0,1]:4d}")
        logger.info(f"       BUY    {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Counterexamples (high confidence wrong predictions)
    high_conf_buy = (probs > 0.8) & (y_true == 0)  # Predicted BUY, actual SELL
    high_conf_sell = (probs < 0.2) & (y_true == 1)  # Predicted SELL, actual BUY
    
    n_wrong_buy = high_conf_buy.sum()
    n_wrong_sell = high_conf_sell.sum()
    
    logger.info(f"\nCounterexamples (high confidence errors):")
    logger.info(f"  Wrong BUY (prob>0.8, actual SELL):  {n_wrong_buy}")
    logger.info(f"  Wrong SELL (prob<0.2, actual BUY):  {n_wrong_sell}")
    
    # Expected behavior
    if mode == "before":
        logger.info(f"\n⚠️  EXPECTED BUG: BUY zone should be ~95% (heavy bias)")
        if pct_buy > 90:
            logger.info(f"✅ Bug confirmed: {pct_buy:.1f}% BUY bias detected")
        else:
            logger.warning(f"❌ Bug NOT reproduced: expected >90% BUY, got {pct_buy:.1f}%")
    else:
        logger.info(f"\n✅ EXPECTED FIX: Distribution should be balanced")
        if 25 <= pct_buy <= 45 and 25 <= pct_sell <= 45 and pct_noise >= 20:
            logger.info(f"✅ Fix successful: balanced distribution achieved")
        else:
            logger.warning(f"⚠️  Distribution still skewed, may need further tuning")
    
    logger.info(f"{'='*60}\n")
    
    return {
        'pct_buy': pct_buy,
        'pct_sell': pct_sell,
        'pct_noise': pct_noise,
        'n_wrong_buy': n_wrong_buy,
        'n_wrong_sell': n_wrong_sell,
        'accuracy': acc if valid_mask.sum() > 0 else 0.0
    }


def main():
    parser = argparse.ArgumentParser(description='XGBoost BUY Bias Bug Exploration')
    parser.add_argument('--mode', type=str, choices=['before', 'after', 'both'], default='both',
                        help='Test mode: before fixes, after fixes, or both')
    parser.add_argument('--symbols', type=str, nargs='+', 
                        default=['GBPUSD', 'XAUUSD', 'US30', 'EURUSD', 'USDJPY'],
                        help='Symbols to test')
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("XGBoost BUY Bias Bug Exploration Test")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info("="*60)
    
    results = {}
    
    for symbol in args.symbols:
        logger.info(f"\n{'#'*60}")
        logger.info(f"Testing {symbol}")
        logger.info(f"{'#'*60}")
        
        # Load real data
        df = load_real_data(symbol)
        if df is None or len(df) < 1000:
            logger.error(f"Insufficient data for {symbol}, skipping...")
            continue
        
        results[symbol] = {}
        
        # Test BEFORE fixes
        if args.mode in ['before', 'both']:
            try:
                model_before, scaler_before, probs_before, y_test_before = train_xgb_before_fix(df, symbol)
                results[symbol]['before'] = analyze_distribution(probs_before, y_test_before, 'before', symbol)
            except Exception as e:
                logger.error(f"Error testing BEFORE fixes for {symbol}: {e}")
        
        # Test AFTER fixes
        if args.mode in ['after', 'both']:
            try:
                model_after, scaler_after, probs_after, y_test_after = train_xgb_after_fix(df, symbol)
                results[symbol]['after'] = analyze_distribution(probs_after, y_test_after, 'after', symbol)
            except Exception as e:
                logger.error(f"Error testing AFTER fixes for {symbol}: {e}")
    
    # Summary report
    logger.info("\n" + "="*60)
    logger.info("SUMMARY REPORT")
    logger.info("="*60)
    
    for symbol, symbol_results in results.items():
        logger.info(f"\n{symbol}:")
        
        if 'before' in symbol_results:
            before = symbol_results['before']
            logger.info(f"  BEFORE: BUY={before['pct_buy']:.1f}% SELL={before['pct_sell']:.1f}% NOISE={before['pct_noise']:.1f}%")
        
        if 'after' in symbol_results:
            after = symbol_results['after']
            logger.info(f"  AFTER:  BUY={after['pct_buy']:.1f}% SELL={after['pct_sell']:.1f}% NOISE={after['pct_noise']:.1f}%")
        
        if 'before' in symbol_results and 'after' in symbol_results:
            improvement = symbol_results['after']['pct_noise'] - symbol_results['before']['pct_noise']
            logger.info(f"  IMPROVEMENT: NOISE zone +{improvement:.1f}%")
    
    logger.info("\n" + "="*60)
    logger.info("Test complete. Results saved to test_bug_exploration.log")
    logger.info("="*60)


if __name__ == '__main__':
    main()

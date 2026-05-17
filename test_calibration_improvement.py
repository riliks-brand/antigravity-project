"""
Calibration Improvement Test — XGBoost v6.0
============================================
اختبار لقياس تحسين المعايرة (Isotonic Calibration).

الهدف:
1. قياس توزيع التنبؤات قبل وبعد المعايرة
2. التأكد من أن المعايرة تحسن التوزيع (أكثر توازناً)
3. التأكد من أن المعايرة لا تضر بالدقة
4. قياس Calibration Error (ECE - Expected Calibration Error)

الاستخدام:
    python test_calibration_improvement.py
    python test_calibration_improvement.py --symbols XAUUSD GBPUSD
"""

import numpy as np
import pandas as pd
import argparse
import logging
import sys
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import calibration_curve

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('test_calibration_improvement.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    حساب Expected Calibration Error (ECE).
    
    ECE يقيس الفرق بين الاحتمالات المتوقعة والدقة الفعلية.
    ECE منخفض = calibration أفضل.
    
    Args:
        y_true: القيم الحقيقية (0 أو 1)
        y_prob: الاحتمالات المتوقعة (0-1)
        n_bins: عدد الـ bins للتقسيم
    
    Returns:
        float: ECE value (0 = perfect calibration)
    """
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        
        bin_prob = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        bin_size = mask.sum()
        
        ece += (bin_size / total_samples) * abs(bin_prob - bin_acc)
    
    return ece


def analyze_calibration(y_true, y_prob_before, y_prob_after, symbol: str):
    """
    تحليل شامل للمعايرة قبل وبعد.
    
    Args:
        y_true: القيم الحقيقية
        y_prob_before: الاحتمالات قبل المعايرة
        y_prob_after: الاحتمالات بعد المعايرة
        symbol: اسم الرمز
    
    Returns:
        dict مع نتائج التحليل
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Calibration Analysis [{symbol}]")
    logger.info(f"{'='*60}")
    
    results = {}
    
    # 1. Distribution Analysis
    logger.info("\n1. Distribution Analysis:")
    
    # Before
    buy_before = (y_prob_before > 0.6).mean() * 100
    sell_before = (y_prob_before < 0.4).mean() * 100
    noise_before = ((y_prob_before >= 0.4) & (y_prob_before <= 0.6)).mean() * 100
    
    logger.info(f"  BEFORE: BUY={buy_before:.1f}% SELL={sell_before:.1f}% NOISE={noise_before:.1f}%")
    
    # After
    buy_after = (y_prob_after > 0.6).mean() * 100
    sell_after = (y_prob_after < 0.4).mean() * 100
    noise_after = ((y_prob_after >= 0.4) & (y_prob_after <= 0.6)).mean() * 100
    
    logger.info(f"  AFTER:  BUY={buy_after:.1f}% SELL={sell_after:.1f}% NOISE={noise_after:.1f}%")
    logger.info(f"  CHANGE: BUY={buy_after-buy_before:+.1f}% SELL={sell_after-sell_before:+.1f}% NOISE={noise_after-noise_before:+.1f}%")
    
    results['distribution'] = {
        'buy_before': buy_before,
        'sell_before': sell_before,
        'noise_before': noise_before,
        'buy_after': buy_after,
        'sell_after': sell_after,
        'noise_after': noise_after,
    }
    
    # 2. Accuracy Analysis
    logger.info("\n2. Accuracy Analysis:")
    
    # Predictions (using 0.6/0.4 thresholds)
    y_pred_before = np.where(y_prob_before > 0.6, 1, np.where(y_prob_before < 0.4, 0, -1))
    y_pred_after = np.where(y_prob_after > 0.6, 1, np.where(y_prob_after < 0.4, 0, -1))
    
    # Accuracy (excluding NOISE)
    valid_before = y_pred_before != -1
    valid_after = y_pred_after != -1
    
    acc_before = accuracy_score(y_true[valid_before], y_pred_before[valid_before]) if valid_before.sum() > 0 else 0
    acc_after = accuracy_score(y_true[valid_after], y_pred_after[valid_after]) if valid_after.sum() > 0 else 0
    
    logger.info(f"  BEFORE: {acc_before*100:.2f}% (on {valid_before.sum()} predictions)")
    logger.info(f"  AFTER:  {acc_after*100:.2f}% (on {valid_after.sum()} predictions)")
    logger.info(f"  CHANGE: {(acc_after-acc_before)*100:+.2f}%")
    
    results['accuracy'] = {
        'before': acc_before * 100,
        'after': acc_after * 100,
        'change': (acc_after - acc_before) * 100,
    }
    
    # 3. Log Loss (Calibration Quality)
    logger.info("\n3. Log Loss (lower = better calibration):")
    
    ll_before = log_loss(y_true, y_prob_before)
    ll_after = log_loss(y_true, y_prob_after)
    
    logger.info(f"  BEFORE: {ll_before:.4f}")
    logger.info(f"  AFTER:  {ll_after:.4f}")
    logger.info(f"  CHANGE: {ll_after-ll_before:+.4f} ({'better' if ll_after < ll_before else 'worse'})")
    
    results['log_loss'] = {
        'before': ll_before,
        'after': ll_after,
        'improved': ll_after < ll_before,
    }
    
    # 4. Expected Calibration Error (ECE)
    logger.info("\n4. Expected Calibration Error (lower = better):")
    
    ece_before = expected_calibration_error(y_true, y_prob_before)
    ece_after = expected_calibration_error(y_true, y_prob_after)
    
    logger.info(f"  BEFORE: {ece_before:.4f}")
    logger.info(f"  AFTER:  {ece_after:.4f}")
    logger.info(f"  CHANGE: {ece_after-ece_before:+.4f} ({'better' if ece_after < ece_before else 'worse'})")
    
    results['ece'] = {
        'before': ece_before,
        'after': ece_after,
        'improved': ece_after < ece_before,
    }
    
    # 5. Calibration Curve
    logger.info("\n5. Calibration Curve Analysis:")
    
    try:
        # Before
        prob_true_before, prob_pred_before = calibration_curve(y_true, y_prob_before, n_bins=10)
        mse_before = np.mean((prob_true_before - prob_pred_before) ** 2)
        
        # After
        prob_true_after, prob_pred_after = calibration_curve(y_true, y_prob_after, n_bins=10)
        mse_after = np.mean((prob_true_after - prob_pred_after) ** 2)
        
        logger.info(f"  Calibration MSE BEFORE: {mse_before:.4f}")
        logger.info(f"  Calibration MSE AFTER:  {mse_after:.4f}")
        logger.info(f"  IMPROVEMENT: {mse_before-mse_after:+.4f} ({'better' if mse_after < mse_before else 'worse'})")
        
        results['calibration_curve'] = {
            'mse_before': mse_before,
            'mse_after': mse_after,
            'improved': mse_after < mse_before,
        }
    except Exception as e:
        logger.warning(f"  Calibration curve analysis failed: {e}")
        results['calibration_curve'] = None
    
    # 6. Overall Assessment
    logger.info("\n6. Overall Assessment:")
    
    improvements = []
    if results['distribution']['noise_after'] > results['distribution']['noise_before']:
        improvements.append("✅ Distribution more balanced")
    else:
        improvements.append("❌ Distribution not improved")
    
    if results['accuracy']['change'] >= -1.0:  # Allow 1% accuracy drop
        improvements.append("✅ Accuracy preserved")
    else:
        improvements.append("❌ Accuracy dropped significantly")
    
    if results['log_loss']['improved']:
        improvements.append("✅ Log loss improved")
    else:
        improvements.append("⚠️  Log loss not improved")
    
    if results['ece']['improved']:
        improvements.append("✅ ECE improved")
    else:
        improvements.append("⚠️  ECE not improved")
    
    for imp in improvements:
        logger.info(f"  {imp}")
    
    passed = sum(1 for imp in improvements if imp.startswith("✅"))
    total = len(improvements)
    
    logger.info(f"\n  Score: {passed}/{total} checks passed")
    
    if passed >= 3:
        logger.info("  ✅ Calibration SUCCESSFUL")
        results['overall'] = 'PASS'
    else:
        logger.info("  ❌ Calibration FAILED")
        results['overall'] = 'FAIL'
    
    logger.info(f"{'='*60}\n")
    
    return results


def test_calibration_on_symbol(symbol: str):
    """
    اختبار المعايرة على رمز واحد.
    
    Args:
        symbol: اسم الرمز (XAUUSD, GBPUSD, etc.)
    
    Returns:
        dict مع نتائج الاختبار
    """
    logger.info(f"\n{'#'*60}")
    logger.info(f"Testing Calibration for {symbol}")
    logger.info(f"{'#'*60}")
    
    try:
        from xgb_model import prepare_tabular_data, calibrate_model
        from xgboost import XGBClassifier
        from data_loader import fetch_mtf_data
        from features import feature_engineering_pipeline
        import MetaTrader5 as mt5
        
        # Initialize MT5
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return None
        
        # Fetch data
        logger.info(f"Fetching data for {symbol}...")
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
        
        # Prepare data
        X_train, X_cal, X_test, y_train, y_cal, y_test, scaler, selected_features = prepare_tabular_data(df)
        
        # Train model
        logger.info(f"Training model for {symbol}...")
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = min(n_neg / max(n_pos, 1), 1.2)
        
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=30,
            reg_alpha=0.2,
            reg_lambda=1.5,
            scale_pos_weight=scale_pos_weight,
            max_delta_step=1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        
        model.fit(X_train, y_train, verbose=False)
        
        # Get predictions BEFORE calibration
        y_prob_before = model.predict_proba(X_test)[:, 1]
        
        # Apply calibration
        logger.info(f"Applying calibration for {symbol}...")
        calibrated_model = calibrate_model(model, X_cal, y_cal, symbol=symbol)
        
        # Get predictions AFTER calibration
        y_prob_after = calibrated_model.predict_proba(X_test)[:, 1]
        
        # Analyze
        results = analyze_calibration(y_true=y_test, 
                                     y_prob_before=y_prob_before,
                                     y_prob_after=y_prob_after,
                                     symbol=symbol)
        
        return results
        
    except Exception as e:
        logger.error(f"Error testing {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Calibration Improvement Test')
    parser.add_argument('--symbols', type=str, nargs='+',
                       default=['XAUUSD', 'GBPUSD', 'US30'],
                       help='Symbols to test')
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Calibration Improvement Test — XGBoost v6.0")
    logger.info("="*60)
    logger.info(f"Testing symbols: {args.symbols}")
    logger.info("="*60)
    
    all_results = {}
    
    for symbol in args.symbols:
        result = test_calibration_on_symbol(symbol)
        if result:
            all_results[symbol] = result
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    for symbol, result in all_results.items():
        status = result.get('overall', 'UNKNOWN')
        noise_improvement = result['distribution']['noise_after'] - result['distribution']['noise_before']
        acc_change = result['accuracy']['change']
        
        logger.info(f"\n{symbol}: {status}")
        logger.info(f"  NOISE improvement: {noise_improvement:+.1f}%")
        logger.info(f"  Accuracy change: {acc_change:+.2f}%")
        logger.info(f"  Log loss improved: {result['log_loss']['improved']}")
        logger.info(f"  ECE improved: {result['ece']['improved']}")
    
    passed = sum(1 for r in all_results.values() if r.get('overall') == 'PASS')
    total = len(all_results)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Overall: {passed}/{total} symbols passed")
    logger.info(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        logger.info("✅ All calibration tests passed!")
    else:
        logger.warning(f"⚠️  {total-passed} symbol(s) failed calibration test")
    
    logger.info("="*60)
    logger.info("Results saved to test_calibration_improvement.log")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())

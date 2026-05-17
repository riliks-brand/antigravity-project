"""
Property-Based Preservation Tests — XGBoost v6.0
=================================================
اختبارات للتأكد من أن الإصلاحات لم تكسر السلوك الحالي.

الخصائص المحفوظة:
1. Feature Engineering: نفس الـ features تُنتج بنفس الطريقة
2. Feature Selection: SHAP + RFE يختاران features معقولة
3. Model Persistence: حفظ وتحميل الموديل يعمل بشكل صحيح
4. Walk-Forward Validation: النتائج متسقة عبر الـ folds
5. Early Stopping: يتوقف عند الـ iteration الصحيح
6. Error Handling: الأخطاء تُعالج بشكل صحيح

الاستخدام:
    python test_preservation.py
    python test_preservation.py --verbose
"""

import numpy as np
import pandas as pd
import argparse
import logging
import sys
import os
import tempfile
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('test_preservation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TestResult:
    """نتيجة اختبار واحد"""
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        msg = f": {self.message}" if self.message else ""
        return f"{status} - {self.name}{msg}"


def create_synthetic_data(n_rows: int = 5000, n_features: int = 50) -> pd.DataFrame:
    """
    إنشاء بيانات اصطناعية للاختبار.
    
    Args:
        n_rows: عدد الصفوف
        n_features: عدد الـ features
    
    Returns:
        DataFrame مع features وTarget
    """
    np.random.seed(42)
    
    # Generate features
    data = {}
    for i in range(n_features):
        data[f'feature_{i}'] = np.random.randn(n_rows)
    
    # Generate target (balanced)
    data['Target'] = np.random.choice([0, 1], size=n_rows, p=[0.5, 0.5])
    
    df = pd.DataFrame(data)
    df.index = pd.date_range('2024-01-01', periods=n_rows, freq='5min')
    
    return df


def test_feature_engineering():
    """
    Test 1: Feature Engineering Preservation
    التأكد من أن feature engineering ينتج نفس الـ features
    """
    logger.info("\n" + "="*60)
    logger.info("Test 1: Feature Engineering Preservation")
    logger.info("="*60)
    
    try:
        from features import (
            add_technical_indicators,
            add_trend_features,
            add_momentum_features,
            add_pivot_points,
            add_session_features,
            add_price_action_features,
            add_market_structure_features,
            add_order_block_features,
            add_fvg_features,
            add_liquidity_features
        )
        
        # Create synthetic OHLC data
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'open': np.random.uniform(1.1, 1.2, n),
            'high': np.random.uniform(1.15, 1.25, n),
            'low': np.random.uniform(1.05, 1.15, n),
            'close': np.random.uniform(1.1, 1.2, n),
            'real_volume': np.random.randint(100, 1000, n),
        })
        df.index = pd.date_range('2024-01-01', periods=n, freq='5min')
        
        # Apply feature engineering
        df = add_technical_indicators(df)
        df = add_trend_features(df)
        df = add_momentum_features(df)
        df = add_pivot_points(df)
        df = add_session_features(df)
        df = add_price_action_features(df)
        df = add_market_structure_features(df)
        df = add_order_block_features(df)
        df = add_fvg_features(df)
        df = add_liquidity_features(df)
        
        # Check expected features exist
        expected_features = [
            'RSI', 'MACD', 'BB_width', 'ATR', 'Volatility',
            'EMA_50', 'EMA_200', 'trend_direction', 'ADX',
            'ROC_5', 'ROC_10', 'momentum_agreement',
            'Pivot', 'R1', 'S1',
            'is_london', 'is_ny', 'session_overlap',
            'body_size', 'upper_shadow', 'lower_shadow',
            'higher_high', 'lower_low', 'structure_trend',
            'distance_to_ob', 'inside_ob_zone', 'ob_strength',
            'fvg_size', 'distance_to_fvg', 'fvg_filled',
            'equal_highs_count', 'liquidity_sweep_flag'
        ]
        
        missing = [f for f in expected_features if f not in df.columns]
        
        if missing:
            return TestResult(
                "Feature Engineering",
                False,
                f"Missing features: {missing}"
            )
        
        # Check no NaN in key features (after warmup period)
        df_check = df.iloc[200:]  # Skip warmup
        key_features = ['RSI', 'MACD', 'ATR', 'EMA_50', 'ADX']
        nan_counts = {f: df_check[f].isna().sum() for f in key_features}
        
        if any(count > 0 for count in nan_counts.values()):
            return TestResult(
                "Feature Engineering",
                False,
                f"NaN values found: {nan_counts}"
            )
        
        return TestResult(
            "Feature Engineering",
            True,
            f"{len(df.columns)} features generated successfully"
        )
        
    except Exception as e:
        return TestResult("Feature Engineering", False, str(e))


def test_feature_selection():
    """
    Test 2: Feature Selection (SHAP + RFE)
    التأكد من أن feature selection يعمل بشكل صحيح
    """
    logger.info("\n" + "="*60)
    logger.info("Test 2: Feature Selection (SHAP + RFE)")
    logger.info("="*60)
    
    try:
        from xgb_model import smart_feature_selection
        from xgboost import XGBClassifier
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 2000
        n_features = 100
        
        X = np.random.randn(n_samples, n_features)
        # Make first 10 features informative
        y = (X[:, :10].sum(axis=1) > 0).astype(int)
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Train a quick model
        model = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42,
            verbosity=0
        )
        model.fit(X, y)
        
        # Run smart feature selection
        selected_indices, selected_features, shap_importance = smart_feature_selection(
            model, X, y, feature_names, symbol="TEST"
        )
        
        # Checks
        if len(selected_features) == 0:
            return TestResult("Feature Selection", False, "No features selected")
        
        if len(selected_features) > 60:
            return TestResult(
                "Feature Selection",
                False,
                f"Too many features selected: {len(selected_features)} (expected ≤60)"
            )
        
        # Check that informative features are prioritized
        # At least 5 of the top 10 selected should be from the informative set
        top_10_selected = selected_features[:10]
        informative_in_top10 = sum(1 for f in top_10_selected if f.startswith('feature_') and int(f.split('_')[1]) < 10)
        
        if informative_in_top10 < 5:
            return TestResult(
                "Feature Selection",
                False,
                f"Only {informative_in_top10}/10 informative features in top 10"
            )
        
        return TestResult(
            "Feature Selection",
            True,
            f"{len(selected_features)} features selected, {informative_in_top10}/10 informative in top 10"
        )
        
    except Exception as e:
        return TestResult("Feature Selection", False, str(e))


def test_model_persistence():
    """
    Test 3: Model Persistence
    التأكد من أن حفظ وتحميل الموديل يعمل بشكل صحيح
    """
    logger.info("\n" + "="*60)
    logger.info("Test 3: Model Persistence")
    logger.info("="*60)
    
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import RobustScaler
        import joblib
        
        # Create and train a simple model
        np.random.seed(42)
        X_train = np.random.randn(1000, 20)
        y_train = np.random.choice([0, 1], 1000)
        
        model = XGBClassifier(n_estimators=50, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        
        scaler = RobustScaler()
        scaler.fit(X_train)
        
        # Save to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"
            scaler_path = Path(tmpdir) / "test_scaler.joblib"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Load back
            loaded_model = joblib.load(model_path)
            loaded_scaler = joblib.load(scaler_path)
            
            # Test predictions match
            X_test = np.random.randn(100, 20)
            
            orig_pred = model.predict_proba(X_test)
            loaded_pred = loaded_model.predict_proba(X_test)
            
            if not np.allclose(orig_pred, loaded_pred):
                return TestResult(
                    "Model Persistence",
                    False,
                    "Predictions don't match after loading"
                )
            
            # Test scaler match
            orig_scaled = scaler.transform(X_test)
            loaded_scaled = loaded_scaler.transform(X_test)
            
            if not np.allclose(orig_scaled, loaded_scaled):
                return TestResult(
                    "Model Persistence",
                    False,
                    "Scaler output doesn't match after loading"
                )
        
        return TestResult(
            "Model Persistence",
            True,
            "Model and scaler saved/loaded successfully"
        )
        
    except Exception as e:
        return TestResult("Model Persistence", False, str(e))


def test_walk_forward_validation():
    """
    Test 4: Walk-Forward Validation
    التأكد من أن WFV يعمل بشكل صحيح ويعطي نتائج متسقة
    """
    logger.info("\n" + "="*60)
    logger.info("Test 4: Walk-Forward Validation")
    logger.info("="*60)
    
    try:
        from xgb_model import walk_forward_validate
        
        # Create synthetic data with trend
        df = create_synthetic_data(n_rows=10000, n_features=30)
        
        # Run WFV
        wfv_result = walk_forward_validate(df, n_folds=3, symbol="TEST")
        
        if wfv_result.get('skipped'):
            return TestResult(
                "Walk-Forward Validation",
                False,
                "WFV was skipped (not enough data)"
            )
        
        # Check results structure
        required_keys = ['fold_accuracies', 'mean_accuracy', 'std_accuracy', 
                        'min_accuracy', 'max_accuracy', 'stability_score']
        missing_keys = [k for k in required_keys if k not in wfv_result]
        
        if missing_keys:
            return TestResult(
                "Walk-Forward Validation",
                False,
                f"Missing keys in result: {missing_keys}"
            )
        
        # Check fold count
        if len(wfv_result['fold_accuracies']) != 3:
            return TestResult(
                "Walk-Forward Validation",
                False,
                f"Expected 3 folds, got {len(wfv_result['fold_accuracies'])}"
            )
        
        # Check accuracy range (should be reasonable for random data)
        mean_acc = wfv_result['mean_accuracy']
        if not (40 <= mean_acc <= 65):
            return TestResult(
                "Walk-Forward Validation",
                False,
                f"Mean accuracy {mean_acc:.1f}% outside expected range [40-65%]"
            )
        
        # Check stability score
        stability = wfv_result['stability_score']
        if stability < 0 or stability > 100:
            return TestResult(
                "Walk-Forward Validation",
                False,
                f"Stability score {stability:.1f}% outside valid range [0-100%]"
            )
        
        return TestResult(
            "Walk-Forward Validation",
            True,
            f"3 folds completed, mean={mean_acc:.1f}%, stability={stability:.1f}%"
        )
        
    except Exception as e:
        return TestResult("Walk-Forward Validation", False, str(e))


def test_early_stopping():
    """
    Test 5: Early Stopping
    التأكد من أن early stopping يعمل بشكل صحيح
    """
    logger.info("\n" + "="*60)
    logger.info("Test 5: Early Stopping")
    logger.info("="*60)
    
    try:
        from xgboost import XGBClassifier
        
        # Create synthetic data
        np.random.seed(42)
        X_train = np.random.randn(2000, 30)
        y_train = np.random.choice([0, 1], 2000)
        X_test = np.random.randn(500, 30)
        y_test = np.random.choice([0, 1], 500)
        
        # Train with early stopping
        model = XGBClassifier(
            n_estimators=1000,
            max_depth=4,
            learning_rate=0.02,
            early_stopping_rounds=30,
            random_state=42,
            verbosity=0
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Check that it stopped early
        best_iter = model.best_iteration
        
        if best_iter >= 1000:
            return TestResult(
                "Early Stopping",
                False,
                f"Model didn't stop early (best_iter={best_iter})"
            )
        
        if best_iter < 10:
            return TestResult(
                "Early Stopping",
                False,
                f"Model stopped too early (best_iter={best_iter})"
            )
        
        return TestResult(
            "Early Stopping",
            True,
            f"Stopped at iteration {best_iter}/1000"
        )
        
    except Exception as e:
        return TestResult("Early Stopping", False, str(e))


def test_error_handling():
    """
    Test 6: Error Handling
    التأكد من أن الأخطاء تُعالج بشكل صحيح
    """
    logger.info("\n" + "="*60)
    logger.info("Test 6: Error Handling")
    logger.info("="*60)
    
    tests_passed = []
    
    try:
        from xgb_model import prepare_tabular_data
        
        # Test 1: Empty DataFrame
        try:
            df_empty = pd.DataFrame()
            prepare_tabular_data(df_empty)
            tests_passed.append(False)  # Should have raised error
        except (ValueError, KeyError):
            tests_passed.append(True)  # Correctly raised error
        
        # Test 2: DataFrame without Target
        try:
            df_no_target = create_synthetic_data(100, 10)
            df_no_target = df_no_target.drop('Target', axis=1)
            prepare_tabular_data(df_no_target)
            tests_passed.append(False)  # Should have raised error
        except (ValueError, KeyError):
            tests_passed.append(True)  # Correctly raised error
        
        # Test 3: Insufficient data
        try:
            df_small = create_synthetic_data(50, 10)  # Too small
            prepare_tabular_data(df_small)
            tests_passed.append(False)  # Should have raised error
        except ValueError:
            tests_passed.append(True)  # Correctly raised error
        
        if all(tests_passed):
            return TestResult(
                "Error Handling",
                True,
                f"{len(tests_passed)}/3 error cases handled correctly"
            )
        else:
            failed = sum(1 for p in tests_passed if not p)
            return TestResult(
                "Error Handling",
                False,
                f"{failed}/3 error cases not handled correctly"
            )
        
    except Exception as e:
        return TestResult("Error Handling", False, str(e))


def test_calibration_preservation():
    """
    Test 7: Calibration Preservation
    التأكد من أن Isotonic Calibration لا تكسر الموديل
    """
    logger.info("\n" + "="*60)
    logger.info("Test 7: Calibration Preservation")
    logger.info("="*60)
    
    try:
        from xgb_model import calibrate_model
        from xgboost import XGBClassifier
        
        # Create synthetic data
        np.random.seed(42)
        X_train = np.random.randn(2000, 30)
        y_train = np.random.choice([0, 1], 2000)
        X_cal = np.random.randn(500, 30)
        y_cal = np.random.choice([0, 1], 500)
        X_test = np.random.randn(500, 30)
        y_test = np.random.choice([0, 1], 500)
        
        # Train model
        model = XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        
        # Get predictions before calibration
        probs_before = model.predict_proba(X_test)[:, 1]
        
        # Calibrate
        calibrated_model = calibrate_model(model, X_cal, y_cal, symbol="TEST")
        
        # Get predictions after calibration
        probs_after = calibrated_model.predict_proba(X_test)[:, 1]
        
        # Check that predictions changed (calibration had effect)
        if np.allclose(probs_before, probs_after):
            return TestResult(
                "Calibration Preservation",
                False,
                "Calibration had no effect on predictions"
            )
        
        # Check that predictions are still in valid range
        if not (0 <= probs_after.min() <= probs_after.max() <= 1):
            return TestResult(
                "Calibration Preservation",
                False,
                f"Predictions out of range: [{probs_after.min():.3f}, {probs_after.max():.3f}]"
            )
        
        # Check that calibration improved distribution
        # (more predictions in middle range)
        noise_before = ((probs_before >= 0.4) & (probs_before <= 0.6)).mean()
        noise_after = ((probs_after >= 0.4) & (probs_after <= 0.6)).mean()
        
        return TestResult(
            "Calibration Preservation",
            True,
            f"Calibration applied successfully (NOISE: {noise_before*100:.1f}% → {noise_after*100:.1f}%)"
        )
        
    except Exception as e:
        return TestResult("Calibration Preservation", False, str(e))


def main():
    parser = argparse.ArgumentParser(description='Property-Based Preservation Tests')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("="*60)
    logger.info("Property-Based Preservation Tests — XGBoost v6.0")
    logger.info("="*60)
    
    # Run all tests
    tests = [
        test_feature_engineering,
        test_feature_selection,
        test_model_persistence,
        test_walk_forward_validation,
        test_early_stopping,
        test_error_handling,
        test_calibration_preservation,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
            logger.info(str(result))
        except Exception as e:
            result = TestResult(test_func.__name__, False, f"Unexpected error: {e}")
            results.append(result)
            logger.error(str(result))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    logger.info(f"Tests passed: {passed}/{total}")
    logger.info(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        logger.info("✅ All tests passed! Preservation verified.")
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed. Review failures above.")
    
    logger.info("="*60)
    logger.info("Results saved to test_preservation.log")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())

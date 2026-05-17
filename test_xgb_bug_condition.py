"""
Bug Condition Exploration Test for XGBoost 95%+ BUY Bias
==========================================================

**Property 1: Bug Condition** - XGBoost 95%+ BUY Bias Detection

**Validates: Requirements 1.1, 1.2, 1.7, 1.8, 1.9**

CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.
DO NOT attempt to fix the test or the code when it fails.

This test encodes the expected behavior - it will validate the fix when it passes
after implementation.

GOAL: Surface counterexamples that demonstrate the 95%+ BUY bias exists across
all trading pairs.

Expected Counterexamples (on unfixed code):
- GBPUSD: ~95% BUY bias (7,590 Weak BUY + 11,195 Strong BUY out of ~19,000 predictions)
- XAUUSD: ~95% BUY bias (7,559 Weak BUY + 11,176 Strong BUY out of ~19,000 predictions)
- US30: ~96% BUY bias (11,649 Weak BUY + 7,448 Strong BUY out of ~19,000 predictions)

The test assertions match the Expected Behavior Properties from design:
- 30-40% BUY (>0.6)
- 30-40% SELL (<0.4)
- 20-40% NOISE (0.4-0.6)
"""

import sys
import os
import numpy as np
import pandas as pd
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xgb_model import XGBModel
from data_loader import fetch_mt5_ohlc
from features import feature_engineering_pipeline
from config import Config
from mt5_engine import connect_to_exness

# Disable logging during tests to reduce noise
logging.disable(logging.CRITICAL)

# Initialize MT5 connection
if not connect_to_exness():
    raise RuntimeError("Failed to connect to MT5. Tests cannot run without MT5 connection.")


def prepare_test_data(symbol: str, min_candles: int = 99000):
    """
    Fetch and prepare test data for a trading pair.
    
    Returns:
        tuple: (df_full, X_test, y_test) where df_full is the full dataset
               and X_test, y_test are the test split
    """
    print(f"\n[TEST] Preparing data for {symbol}...")
    
    # Fetch data
    df = fetch_mt5_ohlc(symbol, count=min_candles)
    if df is None or len(df) < 1000:
        raise ValueError(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} rows")
    
    print(f"[TEST] Fetched {len(df)} candles for {symbol}")
    
    # Add features
    df = feature_engineering_pipeline(df, symbol=symbol)
    
    print(f"[TEST] After feature engineering: {len(df)} rows")
    
    return df


def test_xgb_predictions_distribution(df: pd.DataFrame, symbol: str):
    """
    Test that XGBoost model predictions show balanced distribution.
    
    Expected behavior (after fix):
    - 30-40% BUY predictions (probability > 0.6)
    - 30-40% SELL predictions (probability < 0.4)
    - 20-40% NOISE predictions (0.4 <= probability <= 0.6)
    
    Bug condition (before fix):
    - 95%+ BUY predictions (probability > 0.6)
    - <5% SELL predictions (probability < 0.4)
    - <10% NOISE predictions
    """
    print(f"\n[TEST] Training XGBoost model on {symbol}...")
    
    # Prepare data manually to avoid the feature_importances_ bug in train()
    from xgb_model import prepare_tabular_data, calibrate_model
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
    
    try:
        X_train, X_test, y_train, y_test, scaler, selected_features = prepare_tabular_data(df)
    except ValueError as e:
        print(f"[TEST ERROR] {e}")
        raise
    
    print(f"[TEST] Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Train XGBoost model (matching the current unfixed code configuration)
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    raw_spw = n_neg / max(n_pos, 1)
    scale_pos_weight = min(raw_spw, 1.2)
    
    print(f"[TEST] Class balance: {n_pos} BUY, {n_neg} SELL, scale_pos_weight={scale_pos_weight:.3f}")
    
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
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
    
    # Train
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    
    # Apply calibration (this is in the current code)
    cal_split = int(len(X_train) * 0.8)
    X_cal = X_train[cal_split:]
    y_cal = y_train[cal_split:]
    
    print(f"[TEST] Applying isotonic calibration on {len(X_cal)} samples...")
    calibrated_model = calibrate_model(model, X_cal, y_cal, symbol=symbol)
    
    # Get predictions
    y_pred = calibrated_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"[TEST] Model trained. Accuracy: {accuracy * 100:.2f}%")
    print(f"[TEST] Test set size: {len(X_test)} samples")
    
    # Get probability predictions
    predictions = calibrated_model.predict_proba(X_test)[:, 1]  # Probability of BUY class
    
    # Calculate distribution
    pct_buy = (predictions > 0.6).mean() * 100
    pct_sell = (predictions < 0.4).mean() * 100
    pct_noise = ((predictions >= 0.4) & (predictions <= 0.6)).mean() * 100
    
    print(f"\n[TEST] Prediction Distribution for {symbol}:")
    print(f"  BUY zone (>0.6):     {pct_buy:.1f}%")
    print(f"  SELL zone (<0.4):    {pct_sell:.1f}%")
    print(f"  NOISE zone (0.4-0.6): {pct_noise:.1f}%")
    
    # Count predictions in different ranges for detailed analysis
    strong_buy = (predictions > 0.8).sum()
    weak_buy = ((predictions > 0.6) & (predictions <= 0.8)).sum()
    weak_sell = ((predictions < 0.4) & (predictions >= 0.2)).sum()
    strong_sell = (predictions < 0.2).sum()
    
    print(f"\n[TEST] Detailed Breakdown:")
    print(f"  Strong BUY (>0.8):   {strong_buy} predictions ({strong_buy/len(predictions)*100:.1f}%)")
    print(f"  Weak BUY (0.6-0.8):  {weak_buy} predictions ({weak_buy/len(predictions)*100:.1f}%)")
    print(f"  Weak SELL (0.2-0.4): {weak_sell} predictions ({weak_sell/len(predictions)*100:.1f}%)")
    print(f"  Strong SELL (<0.2):  {strong_sell} predictions ({strong_sell/len(predictions)*100:.1f}%)")
    
    # Store results for assertion
    return {
        'symbol': symbol,
        'pct_buy': pct_buy,
        'pct_sell': pct_sell,
        'pct_noise': pct_noise,
        'strong_buy': strong_buy,
        'weak_buy': weak_buy,
        'weak_sell': weak_sell,
        'strong_sell': strong_sell,
        'total_predictions': len(predictions),
        'predictions': predictions
    }


def is_bug_condition(result: dict) -> bool:
    """
    Check if the bug condition is present.
    
    Bug condition:
    - pct_buy > 90% AND pct_sell < 5% AND pct_noise < 10%
    
    Returns:
        True if bug condition is present, False otherwise
    """
    return (result['pct_buy'] > 90.0 and 
            result['pct_sell'] < 5.0 and 
            result['pct_noise'] < 10.0)


def test_property_1_bug_condition_gbpusd():
    """
    **Property 1: Bug Condition** - XGBoost 95%+ BUY Bias Detection (GBPUSD)
    
    **Validates: Requirements 1.1, 1.2, 1.7**
    
    Test that XGBoost model predictions on GBPUSD test data show balanced
    distribution (30-40% BUY, 30-40% SELL, 20-40% NOISE).
    
    EXPECTED OUTCOME ON UNFIXED CODE: Test FAILS
    - Shows ~95% BUY bias (7,590 Weak BUY + 11,195 Strong BUY)
    - This confirms the bug exists
    
    EXPECTED OUTCOME ON FIXED CODE: Test PASSES
    - Shows balanced distribution
    - This confirms the bug is fixed
    """
    print("\n" + "="*70)
    print("TEST: Property 1 - Bug Condition Detection (GBPUSD)")
    print("="*70)
    
    df = prepare_test_data('GBPUSD')
    result = test_xgb_predictions_distribution(df, 'GBPUSD')
    
    # Check if bug condition is present
    if is_bug_condition(result):
        print(f"\n[COUNTEREXAMPLE FOUND] GBPUSD shows 95%+ BUY bias:")
        print(f"  - BUY zone: {result['pct_buy']:.1f}% (expected 30-40%)")
        print(f"  - SELL zone: {result['pct_sell']:.1f}% (expected 30-40%)")
        print(f"  - NOISE zone: {result['pct_noise']:.1f}% (expected 20-40%)")
        print(f"  - Strong BUY: {result['strong_buy']} predictions")
        print(f"  - Weak BUY: {result['weak_buy']} predictions")
        print(f"\n[EXPECTED] This test SHOULD FAIL on unfixed code - bug confirmed!")
    
    # Assert expected behavior (will fail on unfixed code)
    assert 30.0 <= result['pct_buy'] <= 40.0, \
        f"BUY predictions should be 30-40%, got {result['pct_buy']:.1f}%"
    assert 30.0 <= result['pct_sell'] <= 40.0, \
        f"SELL predictions should be 30-40%, got {result['pct_sell']:.1f}%"
    assert 20.0 <= result['pct_noise'] <= 40.0, \
        f"NOISE predictions should be 20-40%, got {result['pct_noise']:.1f}%"
    
    print(f"\n[PASS] GBPUSD shows balanced distribution - bug is fixed!")


def test_property_1_bug_condition_xauusd():
    """
    **Property 1: Bug Condition** - XGBoost 95%+ BUY Bias Detection (XAUUSD)
    
    **Validates: Requirements 1.1, 1.2, 1.8**
    
    Test that XGBoost model predictions on XAUUSD test data show balanced
    distribution (30-40% BUY, 30-40% SELL, 20-40% NOISE).
    
    EXPECTED OUTCOME ON UNFIXED CODE: Test FAILS
    - Shows ~95% BUY bias (7,559 Weak BUY + 11,176 Strong BUY)
    - This confirms the bug exists
    
    EXPECTED OUTCOME ON FIXED CODE: Test PASSES
    - Shows balanced distribution
    - This confirms the bug is fixed
    """
    print("\n" + "="*70)
    print("TEST: Property 1 - Bug Condition Detection (XAUUSD)")
    print("="*70)
    
    df = prepare_test_data('XAUUSD')
    result = test_xgb_predictions_distribution(df, 'XAUUSD')
    
    # Check if bug condition is present
    if is_bug_condition(result):
        print(f"\n[COUNTEREXAMPLE FOUND] XAUUSD shows 95%+ BUY bias:")
        print(f"  - BUY zone: {result['pct_buy']:.1f}% (expected 30-40%)")
        print(f"  - SELL zone: {result['pct_sell']:.1f}% (expected 30-40%)")
        print(f"  - NOISE zone: {result['pct_noise']:.1f}% (expected 20-40%)")
        print(f"  - Strong BUY: {result['strong_buy']} predictions")
        print(f"  - Weak BUY: {result['weak_buy']} predictions")
        print(f"\n[EXPECTED] This test SHOULD FAIL on unfixed code - bug confirmed!")
    
    # Assert expected behavior (will fail on unfixed code)
    assert 30.0 <= result['pct_buy'] <= 40.0, \
        f"BUY predictions should be 30-40%, got {result['pct_buy']:.1f}%"
    assert 30.0 <= result['pct_sell'] <= 40.0, \
        f"SELL predictions should be 30-40%, got {result['pct_sell']:.1f}%"
    assert 20.0 <= result['pct_noise'] <= 40.0, \
        f"NOISE predictions should be 20-40%, got {result['pct_noise']:.1f}%"
    
    print(f"\n[PASS] XAUUSD shows balanced distribution - bug is fixed!")


def test_property_1_bug_condition_us30():
    """
    **Property 1: Bug Condition** - XGBoost 95%+ BUY Bias Detection (US30)
    
    **Validates: Requirements 1.1, 1.2, 1.9**
    
    Test that XGBoost model predictions on US30 test data show balanced
    distribution (30-40% BUY, 30-40% SELL, 20-40% NOISE).
    
    EXPECTED OUTCOME ON UNFIXED CODE: Test FAILS
    - Shows ~96% BUY bias (11,649 Weak BUY + 7,448 Strong BUY)
    - This confirms the bug exists
    
    EXPECTED OUTCOME ON FIXED CODE: Test PASSES
    - Shows balanced distribution
    - This confirms the bug is fixed
    """
    print("\n" + "="*70)
    print("TEST: Property 1 - Bug Condition Detection (US30)")
    print("="*70)
    
    df = prepare_test_data('US30')
    result = test_xgb_predictions_distribution(df, 'US30')
    
    # Check if bug condition is present
    if is_bug_condition(result):
        print(f"\n[COUNTEREXAMPLE FOUND] US30 shows 95%+ BUY bias:")
        print(f"  - BUY zone: {result['pct_buy']:.1f}% (expected 30-40%)")
        print(f"  - SELL zone: {result['pct_sell']:.1f}% (expected 30-40%)")
        print(f"  - NOISE zone: {result['pct_noise']:.1f}% (expected 20-40%)")
        print(f"  - Strong BUY: {result['strong_buy']} predictions")
        print(f"  - Weak BUY: {result['weak_buy']} predictions")
        print(f"\n[EXPECTED] This test SHOULD FAIL on unfixed code - bug confirmed!")
    
    # Assert expected behavior (will fail on unfixed code)
    assert 30.0 <= result['pct_buy'] <= 40.0, \
        f"BUY predictions should be 30-40%, got {result['pct_buy']:.1f}%"
    assert 30.0 <= result['pct_sell'] <= 40.0, \
        f"SELL predictions should be 30-40%, got {result['pct_sell']:.1f}%"
    assert 20.0 <= result['pct_noise'] <= 40.0, \
        f"NOISE predictions should be 20-40%, got {result['pct_noise']:.1f}%"
    
    print(f"\n[PASS] US30 shows balanced distribution - bug is fixed!")


def test_property_1_bug_condition_eurusd():
    """
    **Property 1: Bug Condition** - XGBoost 95%+ BUY Bias Detection (EURUSD)
    
    **Validates: Requirements 1.1, 1.2**
    
    Test that XGBoost model predictions on EURUSD test data show balanced
    distribution (30-40% BUY, 30-40% SELL, 20-40% NOISE).
    
    EXPECTED OUTCOME ON UNFIXED CODE: Test FAILS
    - Shows 95%+ BUY bias
    - This confirms the bug exists
    
    EXPECTED OUTCOME ON FIXED CODE: Test PASSES
    - Shows balanced distribution
    - This confirms the bug is fixed
    """
    print("\n" + "="*70)
    print("TEST: Property 1 - Bug Condition Detection (EURUSD)")
    print("="*70)
    
    df = prepare_test_data('EURUSD')
    result = test_xgb_predictions_distribution(df, 'EURUSD')
    
    # Check if bug condition is present
    if is_bug_condition(result):
        print(f"\n[COUNTEREXAMPLE FOUND] EURUSD shows 95%+ BUY bias:")
        print(f"  - BUY zone: {result['pct_buy']:.1f}% (expected 30-40%)")
        print(f"  - SELL zone: {result['pct_sell']:.1f}% (expected 30-40%)")
        print(f"  - NOISE zone: {result['pct_noise']:.1f}% (expected 20-40%)")
        print(f"  - Strong BUY: {result['strong_buy']} predictions")
        print(f"  - Weak BUY: {result['weak_buy']} predictions")
        print(f"\n[EXPECTED] This test SHOULD FAIL on unfixed code - bug confirmed!")
    
    # Assert expected behavior (will fail on unfixed code)
    assert 30.0 <= result['pct_buy'] <= 40.0, \
        f"BUY predictions should be 30-40%, got {result['pct_buy']:.1f}%"
    assert 30.0 <= result['pct_sell'] <= 40.0, \
        f"SELL predictions should be 30-40%, got {result['pct_sell']:.1f}%"
    assert 20.0 <= result['pct_noise'] <= 40.0, \
        f"NOISE predictions should be 20-40%, got {result['pct_noise']:.1f}%"
    
    print(f"\n[PASS] EURUSD shows balanced distribution - bug is fixed!")


def test_property_1_bug_condition_usdjpy():
    """
    **Property 1: Bug Condition** - XGBoost 95%+ BUY Bias Detection (USDJPY)
    
    **Validates: Requirements 1.1, 1.2**
    
    Test that XGBoost model predictions on USDJPY test data show balanced
    distribution (30-40% BUY, 30-40% SELL, 20-40% NOISE).
    
    EXPECTED OUTCOME ON UNFIXED CODE: Test FAILS
    - Shows 95%+ BUY bias
    - This confirms the bug exists
    
    EXPECTED OUTCOME ON FIXED CODE: Test PASSES
    - Shows balanced distribution
    - This confirms the bug is fixed
    """
    print("\n" + "="*70)
    print("TEST: Property 1 - Bug Condition Detection (USDJPY)")
    print("="*70)
    
    df = prepare_test_data('USDJPY')
    result = test_xgb_predictions_distribution(df, 'USDJPY')
    
    # Check if bug condition is present
    if is_bug_condition(result):
        print(f"\n[COUNTEREXAMPLE FOUND] USDJPY shows 95%+ BUY bias:")
        print(f"  - BUY zone: {result['pct_buy']:.1f}% (expected 30-40%)")
        print(f"  - SELL zone: {result['pct_sell']:.1f}% (expected 30-40%)")
        print(f"  - NOISE zone: {result['pct_noise']:.1f}% (expected 20-40%)")
        print(f"  - Strong BUY: {result['strong_buy']} predictions")
        print(f"  - Weak BUY: {result['weak_buy']} predictions")
        print(f"\n[EXPECTED] This test SHOULD FAIL on unfixed code - bug confirmed!")
    
    # Assert expected behavior (will fail on unfixed code)
    assert 30.0 <= result['pct_buy'] <= 40.0, \
        f"BUY predictions should be 30-40%, got {result['pct_buy']:.1f}%"
    assert 30.0 <= result['pct_sell'] <= 40.0, \
        f"SELL predictions should be 30-40%, got {result['pct_sell']:.1f}%"
    assert 20.0 <= result['pct_noise'] <= 40.0, \
        f"NOISE predictions should be 20-40%, got {result['pct_noise']:.1f}%"
    
    print(f"\n[PASS] USDJPY shows balanced distribution - bug is fixed!")


if __name__ == "__main__":
    """
    Run all bug condition exploration tests.
    
    EXPECTED OUTCOME ON UNFIXED CODE:
    - All tests FAIL with counterexamples showing 95%+ BUY bias
    - This confirms the bug exists across all trading pairs
    
    EXPECTED OUTCOME ON FIXED CODE:
    - All tests PASS showing balanced distributions
    - This confirms the bug is fixed
    """
    print("\n" + "="*70)
    print("BUG CONDITION EXPLORATION TEST SUITE")
    print("XGBoost 95%+ BUY Bias Detection")
    print("="*70)
    print("\nCRITICAL: These tests are EXPECTED TO FAIL on unfixed code.")
    print("Failure confirms the bug exists. Success confirms the bug is fixed.")
    print("="*70)
    
    # Test all trading pairs
    symbols = ['GBPUSD', 'XAUUSD', 'US30', 'EURUSD', 'USDJPY']
    results = []
    
    for symbol in symbols:
        try:
            if symbol == 'GBPUSD':
                test_property_1_bug_condition_gbpusd()
            elif symbol == 'XAUUSD':
                test_property_1_bug_condition_xauusd()
            elif symbol == 'US30':
                test_property_1_bug_condition_us30()
            elif symbol == 'EURUSD':
                test_property_1_bug_condition_eurusd()
            elif symbol == 'USDJPY':
                test_property_1_bug_condition_usdjpy()
            
            results.append((symbol, 'PASS'))
        except AssertionError as e:
            results.append((symbol, 'FAIL', str(e)))
        except Exception as e:
            results.append((symbol, 'ERROR', str(e)))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for result in results:
        if len(result) == 2:
            symbol, status = result
            print(f"  {symbol:10s} : {status}")
        else:
            symbol, status, msg = result
            print(f"  {symbol:10s} : {status}")
            print(f"    {msg}")
    print("="*70)
    
    # Final verdict
    failed_count = sum(1 for r in results if r[1] == 'FAIL')
    error_count = sum(1 for r in results if r[1] == 'ERROR')
    
    if failed_count > 0:
        print(f"\n[EXPECTED ON UNFIXED CODE] {failed_count} test(s) failed - bug confirmed!")
        print("These failures are counterexamples demonstrating the 95%+ BUY bias.")
    elif error_count > 0:
        print(f"\n[ERROR] {error_count} test(s) encountered errors - check logs.")
    else:
        print(f"\n[SUCCESS] All tests passed - bug is fixed!")

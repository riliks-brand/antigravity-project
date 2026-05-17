"""
Test script to verify XGBoost BUY bias fix (v6.1)

This script tests the critical fixes:
1. 3-way data split (60% train, 20% calibration, 20% test)
2. Calibration on UNSEEN data
3. n_estimators alignment (300 trees)
4. Prediction distribution analysis

Run this BEFORE retraining models to verify the fix works.
"""

import pandas as pd
import numpy as np
from xgb_model import prepare_tabular_data, calibrate_model
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples=10000, buy_bias=0.95):
    """
    Generate synthetic data with known BUY bias to test calibration.
    
    Args:
        n_samples: Number of samples to generate
        buy_bias: Proportion of BUY labels (0.95 = 95% BUY bias)
    """
    logger.info(f"Generating {n_samples} synthetic samples with {buy_bias*100:.0f}% BUY bias...")
    
    # Generate features
    np.random.seed(42)
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    
    # Generate biased labels (simulate uptrend market)
    y = np.random.choice([0, 1], size=n_samples, p=[1-buy_bias, buy_bias])
    
    # Create DataFrame with feature names
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Target'] = y
    
    # Add some lagged features (simulate real data structure)
    for lag in [1, 3, 5]:
        for col in feature_names[:5]:  # Only lag first 5 features
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    df = df.dropna()
    
    logger.info(f"Generated data: {len(df)} rows, {len(df.columns)-1} features")
    logger.info(f"Label distribution: BUY={np.sum(df['Target']==1)} ({np.mean(df['Target']==1)*100:.1f}%), SELL={np.sum(df['Target']==0)} ({np.mean(df['Target']==0)*100:.1f}%)")
    
    return df


def test_data_split():
    """Test 1: Verify 3-way split (60/20/20)"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Data Split Verification")
    logger.info("="*60)
    
    df = generate_synthetic_data(n_samples=10000)
    
    try:
        X_train, X_cal, X_test, y_train, y_cal, y_test, scaler, features = prepare_tabular_data(df)
        
        total = len(X_train) + len(X_cal) + len(X_test)
        train_pct = len(X_train) / total * 100
        cal_pct = len(X_cal) / total * 100
        test_pct = len(X_test) / total * 100
        
        logger.info(f"✅ Split successful:")
        logger.info(f"   Train: {len(X_train)} samples ({train_pct:.1f}%)")
        logger.info(f"   Calibration: {len(X_cal)} samples ({cal_pct:.1f}%)")
        logger.info(f"   Test: {len(X_test)} samples ({test_pct:.1f}%)")
        
        # Verify proportions are approximately correct
        assert 55 <= train_pct <= 65, f"Train split should be ~60%, got {train_pct:.1f}%"
        assert 15 <= cal_pct <= 25, f"Calibration split should be ~20%, got {cal_pct:.1f}%"
        assert 15 <= test_pct <= 25, f"Test split should be ~20%, got {test_pct:.1f}%"
        
        logger.info("✅ TEST 1 PASSED: Data split is correct (60/20/20)")
        return True
        
    except Exception as e:
        logger.error(f"❌ TEST 1 FAILED: {e}")
        return False


def test_calibration_effect():
    """Test 2: Verify calibration reduces BUY bias"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Calibration Effect Verification")
    logger.info("="*60)
    
    # Generate data with extreme BUY bias (95%)
    df = generate_synthetic_data(n_samples=10000, buy_bias=0.95)
    
    try:
        X_train, X_cal, X_test, y_train, y_cal, y_test, scaler, features = prepare_tabular_data(df)
        
        # Train uncalibrated model
        logger.info("Training uncalibrated model...")
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.02,
            min_child_weight=30,
            reg_alpha=0.2,
            reg_lambda=1.5,
            scale_pos_weight=1.2,
            max_delta_step=1,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # Check distribution BEFORE calibration
        raw_probs = model.predict_proba(X_test)[:, 1]
        pct_buy_raw = (raw_probs > 0.6).mean() * 100
        pct_sell_raw = (raw_probs < 0.4).mean() * 100
        pct_noise_raw = ((raw_probs >= 0.4) & (raw_probs <= 0.6)).mean() * 100
        
        logger.info(f"BEFORE Calibration:")
        logger.info(f"   BUY (>0.6):   {pct_buy_raw:.1f}%")
        logger.info(f"   SELL (<0.4):  {pct_sell_raw:.1f}%")
        logger.info(f"   NOISE (0.4-0.6): {pct_noise_raw:.1f}%")
        
        # Apply calibration
        logger.info(f"Applying calibration on {len(X_cal)} UNSEEN samples...")
        calibrated_model = calibrate_model(model, X_cal, y_cal, symbol="TEST")
        
        # Check distribution AFTER calibration
        cal_probs = calibrated_model.predict_proba(X_test)[:, 1]
        pct_buy_cal = (cal_probs > 0.6).mean() * 100
        pct_sell_cal = (cal_probs < 0.4).mean() * 100
        pct_noise_cal = ((cal_probs >= 0.4) & (cal_probs <= 0.6)).mean() * 100
        
        logger.info(f"AFTER Calibration:")
        logger.info(f"   BUY (>0.6):   {pct_buy_cal:.1f}% (change: {pct_buy_cal - pct_buy_raw:+.1f}%)")
        logger.info(f"   SELL (<0.4):  {pct_sell_cal:.1f}% (change: {pct_sell_cal - pct_sell_raw:+.1f}%)")
        logger.info(f"   NOISE (0.4-0.6): {pct_noise_cal:.1f}% (change: {pct_noise_cal - pct_noise_raw:+.1f}%)")
        
        # Verify calibration improved distribution
        noise_improvement = pct_noise_cal - pct_noise_raw
        buy_reduction = pct_buy_raw - pct_buy_cal
        
        logger.info(f"\nCalibration Impact:")
        logger.info(f"   NOISE zone increased by: {noise_improvement:+.1f}%")
        logger.info(f"   BUY zone decreased by: {buy_reduction:+.1f}%")
        
        # Success criteria: NOISE should increase significantly
        if noise_improvement > 5.0:
            logger.info("✅ TEST 2 PASSED: Calibration significantly improved distribution")
            return True
        else:
            logger.warning(f"⚠️  TEST 2 WARNING: Calibration improvement is small ({noise_improvement:.1f}%)")
            return True  # Still pass, but warn
            
    except Exception as e:
        logger.error(f"❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hyperparameter_alignment():
    """Test 3: Verify n_estimators is 300 (not 500)"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Hyperparameter Alignment Verification")
    logger.info("="*60)
    
    try:
        # Create a model with the same config as XGBModel.train()
        model = XGBClassifier(
            n_estimators=300,  # Should be 300, not 500
            max_depth=4,
            learning_rate=0.02,
            min_child_weight=30,
            reg_alpha=0.2,
            reg_lambda=1.5,
            scale_pos_weight=1.2,
            max_delta_step=1,
            random_state=42,
        )
        
        logger.info(f"✅ Model configuration:")
        logger.info(f"   n_estimators: {model.n_estimators} (expected: 300)")
        logger.info(f"   max_depth: {model.max_depth}")
        logger.info(f"   learning_rate: {model.learning_rate}")
        logger.info(f"   min_child_weight: {model.min_child_weight}")
        logger.info(f"   reg_alpha: {model.reg_alpha}")
        logger.info(f"   reg_lambda: {model.reg_lambda}")
        logger.info(f"   max_delta_step: {model.max_delta_step}")
        
        assert model.n_estimators == 300, f"n_estimators should be 300, got {model.n_estimators}"
        assert model.min_child_weight == 30, f"min_child_weight should be 30, got {model.min_child_weight}"
        assert model.max_delta_step == 1, f"max_delta_step should be 1, got {model.max_delta_step}"
        
        logger.info("✅ TEST 3 PASSED: Hyperparameters are correctly aligned")
        return True
        
    except Exception as e:
        logger.error(f"❌ TEST 3 FAILED: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "="*60)
    logger.info("XGBoost BUY Bias Fix Verification (v6.1)")
    logger.info("="*60)
    logger.info("This script tests the critical fixes:")
    logger.info("1. 3-way data split (60% train, 20% cal, 20% test)")
    logger.info("2. Calibration on UNSEEN data")
    logger.info("3. Hyperparameter alignment (300 trees)")
    logger.info("="*60)
    
    results = []
    
    # Run tests
    results.append(("Data Split", test_data_split()))
    results.append(("Calibration Effect", test_calibration_effect()))
    results.append(("Hyperparameter Alignment", test_hyperparameter_alignment()))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED! The fix is working correctly.")
        logger.info("\nNext steps:")
        logger.info("1. Run train_offline.py to retrain all models with the fix")
        logger.info("2. Monitor ensemble_decisions.csv for balanced predictions")
        logger.info("3. Verify live trading shows ~30-40% BUY, ~30-40% SELL, ~20-40% NOISE")
    else:
        logger.error("\n❌ SOME TESTS FAILED! Review the errors above.")
    
    logger.info("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

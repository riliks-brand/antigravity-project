# XGBoost BUY Bias Bugfix Design

## Overview

The XGBoost model exhibits severe directional bias toward BUY predictions (95%+ of all predictions are BUY), causing consistent losing trades across all trading pairs. This bug stems from training data temporal bias (uptrend period Jan 2025 - May 2026), uncalibrated probability outputs, and configuration inconsistencies.

The fix strategy combines multiple approaches:
1. **Isotonic Calibration**: Apply isotonic regression to correct probability distributions and eliminate temporal bias
2. **Hyperparameter Alignment**: Match final model configuration with Walk-Forward Validation (300 trees instead of 500)
3. **Regularization Strengthening**: Increase min_child_weight, reg_alpha, reg_lambda, and add max_delta_step to prevent overfitting to directional patterns
4. **scale_pos_weight Capping**: Limit scale_pos_weight to 1.2 maximum to prevent extreme directional bias
5. **Data Consistency**: Ensure all symbols fetch minimum 99,000 candles for consistent model quality
6. **SHAP Balanced Sampling**: Use 50% BUY / 50% SELL sampling when computing SHAP values to avoid directional bias in feature importance

This is a targeted fix that preserves all existing functionality while correcting the probability distribution to reflect actual market conditions (30-40% BUY, 30-40% SELL, 20-40% NOISE).

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug - when XGBoost outputs 95%+ predictions in the BUY zone (probability > 0.6) regardless of actual market conditions
- **Property (P)**: The desired behavior - XGBoost should output balanced predictions reflecting actual market conditions with approximately 30-40% BUY, 30-40% SELL, and 20-40% NOISE zones
- **Preservation**: Existing training pipeline, feature engineering, model persistence, and prediction interface that must remain unchanged by the fix
- **Isotonic Calibration**: Non-parametric probability calibration using isotonic regression that maps uncalibrated probabilities to calibrated probabilities while preserving monotonicity
- **Temporal Bias**: Model learning directional patterns from training period market regime (uptrend) rather than learning from feature relationships
- **scale_pos_weight**: XGBoost hyperparameter that adjusts the balance between positive and negative classes (BUY vs SELL)
- **Calibration Set**: A held-out subset of training data (20%) used exclusively for probability calibration, not for model training
- **SHAP Values**: SHapley Additive exPlanations - feature importance values that explain individual predictions
- **handleKeyPress**: The function in `xgb_model.py` that trains the XGBoost model and applies calibration
- **predict_proba**: The method that outputs probability values between 0.0 and 1.0 for the BUY class

## Bug Details

### Bug Condition

The bug manifests when the XGBoost model makes predictions on test data from any trading pair (GBPUSD, XAUUSD, US30, EURUSD, USDJPY). The model outputs 95%+ predictions in the BUY zone (probability > 0.6), with almost zero SELL predictions (probability < 0.4), regardless of actual market conditions or feature values.

The root cause is a combination of:
1. Training data from Jan 2025 - May 2026 captured an uptrend period, causing the model to learn "BUY = profit" as a dominant pattern
2. XGBoost raw probabilities are uncalibrated and skewed toward BUY
3. Configuration mismatch between final model (500 trees) and Walk-Forward Validation (300 trees)
4. Insufficient regularization allows overfitting to temporal directional patterns

**Formal Specification:**
```
FUNCTION isBugCondition(predictions)
  INPUT: predictions of type array[float] (XGBoost probability outputs on test data)
  OUTPUT: boolean
  
  pct_buy_zone = COUNT(predictions > 0.6) / LENGTH(predictions)
  pct_sell_zone = COUNT(predictions < 0.4) / LENGTH(predictions)
  pct_noise_zone = COUNT(0.4 <= predictions <= 0.6) / LENGTH(predictions)
  
  RETURN (pct_buy_zone > 0.90)
         AND (pct_sell_zone < 0.05)
         AND (pct_noise_zone < 0.10)
END FUNCTION
```

### Examples

- **GBPUSD Test Data**: Weak BUY (0.6-0.8): 7,590 predictions, Strong BUY (>0.8): 11,195 predictions, SELL (<0.4): ~0 predictions → 95%+ BUY bias
- **XAUUSD Test Data**: Weak BUY (0.6-0.8): 7,559 predictions, Strong BUY (>0.8): 11,176 predictions, SELL (<0.4): ~0 predictions → 95%+ BUY bias
- **US30 Test Data**: Weak BUY (0.6-0.8): 11,649 predictions, Strong BUY (>0.8): 7,448 predictions, SELL (<0.4): ~0 predictions → 96%+ BUY bias
- **Expected Behavior**: For balanced test data, approximately 30-40% BUY (>0.6), 30-40% SELL (<0.4), 20-40% NOISE (0.4-0.6)

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Model training pipeline must continue to successfully complete training and produce a trained model
- Feature engineering (lagged features, rolling statistics, delta features) must remain unchanged
- Feature selection using SelectKBest, SHAP, and RFE must continue to work
- Model persistence (saving/loading from joblib files) must remain unchanged
- Walk-Forward Validation with 5 folds must continue to report per-fold accuracies
- Early stopping monitoring must continue to work
- Ensemble engine prediction interface must remain unchanged (returns probability 0.0-1.0)
- Model retraining triggers (time-based and candle-based) must continue to work
- Error handling (insufficient data, constant features) must remain unchanged

**Scope:**
All inputs and operations that do NOT involve probability calibration or hyperparameter configuration should be completely unaffected by this fix. This includes:
- Feature engineering and selection logic
- Data loading and preprocessing
- Model persistence and loading
- Walk-Forward Validation logic
- SHAP and RFE feature selection
- Ensemble engine integration

## Hypothesized Root Cause

Based on the bug description and code analysis, the most likely issues are:

1. **Temporal Bias in Training Data**: The training period (Jan 2025 - May 2026) captured a strong uptrend across all trading pairs. XGBoost learned that BUY predictions are profitable regardless of feature values, creating a directional bias that persists even when market conditions change.

2. **Uncalibrated Probability Outputs**: XGBoost raw probabilities are not properly calibrated. The model outputs high confidence BUY predictions (>0.8) even when the actual probability should be closer to 0.5 (noise zone). This is a known issue with gradient boosting models.

3. **Configuration Mismatch**: The final model uses n_estimators=500 while Walk-Forward Validation uses n_estimators=300. This creates a train-test mismatch where the validation results don't accurately reflect final model performance.

4. **Insufficient Regularization**: Current regularization parameters (min_child_weight=20, reg_alpha=0.1, reg_lambda=1.0) are too weak to prevent overfitting to temporal directional patterns. The model needs stronger regularization to focus on feature relationships rather than directional trends.

5. **Ineffective scale_pos_weight**: Despite balanced training labels (50/50 BUY/SELL), scale_pos_weight ≈ 1.0 has no effect. The bias comes from temporal patterns, not class imbalance. However, uncapped scale_pos_weight could amplify bias if training data becomes slightly imbalanced.

6. **EURUSD Data Insufficiency**: EURUSD is trained with only 20,000 candles while other symbols use 99,000 candles, creating inconsistent model quality across trading pairs.

7. **SHAP Directional Bias**: When computing SHAP values on training data dominated by BUY predictions, feature importance becomes biased toward features that predict BUY, creating a feedback loop that reinforces the bias.

## Correctness Properties

Property 1: Bug Condition - Balanced Prediction Distribution

_For any_ test data from any trading pair where the XGBoost model makes predictions after applying the fix, the fixed model SHALL output approximately 30-40% predictions in the BUY zone (>0.6), 30-40% predictions in the SELL zone (<0.4), and 20-40% predictions in the NOISE zone (0.4-0.6), reflecting actual market conditions rather than temporal training bias.

**Validates: Requirements 2.1, 2.2, 2.7, 2.8, 2.9**

Property 2: Preservation - Training Pipeline Functionality

_For any_ valid feature data with proper Target labels, the fixed training pipeline SHALL continue to successfully complete all existing operations (feature engineering, feature selection, scaling, model training, model persistence, Walk-Forward Validation) and produce the same outputs as the original pipeline, preserving all existing functionality except probability calibration.

**Validates: Requirements 3.1, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12, 3.14, 3.15, 3.16**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `xgb_model.py`

**Function**: `train()` method in `XGBModel` class

**Specific Changes**:

1. **Isotonic Calibration Implementation**:
   - After training the base XGBoost model, split training data: 80% for model training, 20% for calibration
   - Apply `sklearn.calibration.CalibratedClassifierCV` with `method='isotonic'` and `cv='prefit'`
   - Use the calibration set (20% of training data) to fit the isotonic regression
   - Replace `self.model` with the calibrated wrapper before saving
   - Log probability distribution before and after calibration (% in BUY/SELL/NOISE zones)

2. **Hyperparameter Alignment**:
   - Change `n_estimators` from 500 to 300 in the final model to match Walk-Forward Validation
   - This ensures validation results accurately reflect final model performance

3. **Regularization Strengthening**:
   - Increase `min_child_weight` from 20 to 30 (prevents overfitting to small sample patterns)
   - Increase `reg_alpha` from 0.2 to 0.2 (already correct in code, ensure it's applied)
   - Increase `reg_lambda` from 1.0 to 1.5 (stronger L2 penalty)
   - Add `max_delta_step=1` (limits extreme predictions, prevents overconfident BUY predictions)

4. **scale_pos_weight Capping**:
   - Calculate scale_pos_weight as usual: `n_neg / n_pos`
   - Apply cap: `scale_pos_weight = min(calculated_value, 1.2)`
   - Log when capping occurs to track its effect

5. **Calibration Set Split**:
   - After `self.model.fit()` completes, create calibration set from training data
   - Use `cal_split = int(len(X_train) * 0.8)` to get 80/20 split
   - Extract `X_cal = X_train[cal_split:]` and `y_cal = y_train[cal_split:]`
   - Call `calibrate_model(self.model, X_cal, y_cal, symbol="TRAIN")`
   - The `calibrate_model()` function already exists in the code and implements isotonic calibration correctly

**File**: `data_loader.py`

**Function**: `fetch_mt5_ohlc()`

**Specific Changes**:

1. **EURUSD Data Consistency Fix**:
   - In `fetch_mt5_ohlc()`, ensure minimum 99,000 candles for all symbols
   - Change: `count = max(Config.DATA_POINTS, 99000)` if count is None
   - Also apply when count is provided: `count = max(count, 99000)`
   - This ensures EURUSD gets the same data volume as other symbols

**File**: `xgb_model.py`

**Function**: `shap_feature_selection()`

**Specific Changes**:

1. **SHAP Balanced Sampling** (already implemented in code):
   - The code already implements balanced sampling (50% BUY, 50% SELL) when computing SHAP values
   - Verify this logic is working correctly and not being bypassed
   - The implementation at lines 234-247 correctly samples equal numbers of BUY and SELL predictions

**File**: `xgb_model.py`

**Function**: `walk_forward_validate()`

**Specific Changes**:

1. **Walk-Forward Validation Hyperparameter Sync**:
   - Ensure WFV uses the same hyperparameters as the final model
   - Already uses n_estimators=300 ✓
   - Update min_child_weight from 30 to 30 (already correct)
   - Update reg_alpha to 0.2 (already correct)
   - Update reg_lambda to 1.5 (already correct)
   - Update scale_pos_weight capping to 1.2 (already correct)
   - Add max_delta_step=1 (already correct)

**Summary of Code Changes**:
- `xgb_model.py` line ~680: Change `n_estimators=500` to `n_estimators=300`
- `xgb_model.py` line ~684: Change `min_child_weight=20` to `min_child_weight=30`
- `xgb_model.py` line ~686: Change `reg_lambda=1.5` to `reg_lambda=1.5` (already correct)
- `xgb_model.py` line ~687: Ensure `scale_pos_weight=scale_pos_weight` uses capped value (already correct)
- `xgb_model.py` line ~688: Ensure `max_delta_step=1` is present (already correct)
- `xgb_model.py` line ~700: Add calibration call after model.fit() (already exists at line ~703)
- `data_loader.py` line ~40: Add `count = max(count, 99000)` to ensure minimum candles

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Train the XGBoost model on historical data (Jan 2025 - May 2026) WITHOUT isotonic calibration, then evaluate predictions on test data from each trading pair. Measure the distribution of predictions across BUY/SELL/NOISE zones. Run these tests on the UNFIXED code to observe the 95%+ BUY bias and understand the root cause.

**Test Cases**:
1. **GBPUSD Distribution Test**: Train model on GBPUSD data, evaluate on test set, measure % in each zone (will show 95%+ BUY on unfixed code)
2. **XAUUSD Distribution Test**: Train model on XAUUSD data, evaluate on test set, measure % in each zone (will show 95%+ BUY on unfixed code)
3. **US30 Distribution Test**: Train model on US30 data, evaluate on test set, measure % in each zone (will show 96%+ BUY on unfixed code)
4. **EURUSD Data Volume Test**: Check candle count fetched for EURUSD (will show 20K on unfixed code, should be 99K)
5. **Probability Calibration Test**: Compute calibration curve (predicted probability vs actual frequency) on test data (will show poor calibration on unfixed code)
6. **Walk-Forward Stability Test**: Run WFV on USDJPY, check fold accuracies (will show Fold 3 at 53% on unfixed code)

**Expected Counterexamples**:
- Test data predictions will show 90-96% in BUY zone (>0.6), <5% in SELL zone (<0.4)
- Calibration curve will show predicted probabilities 0.6-0.9 corresponding to actual frequencies 0.5-0.6 (overconfident)
- EURUSD will fetch only 20,000 candles instead of 99,000
- Walk-Forward Validation will show high variance across folds (std > 3%)

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds (test data from any trading pair), the fixed function produces the expected behavior (balanced prediction distribution).

**Pseudocode:**
```
FOR ALL test_data FROM [GBPUSD, XAUUSD, US30, EURUSD, USDJPY] DO
  predictions := xgb_model_fixed.predict_proba(test_data)
  pct_buy := COUNT(predictions > 0.6) / LENGTH(predictions)
  pct_sell := COUNT(predictions < 0.4) / LENGTH(predictions)
  pct_noise := COUNT(0.4 <= predictions <= 0.6) / LENGTH(predictions)
  
  ASSERT 0.30 <= pct_buy <= 0.40
  ASSERT 0.30 <= pct_sell <= 0.40
  ASSERT 0.20 <= pct_noise <= 0.40
  ASSERT NOT isBugCondition(predictions)
END FOR
```

**Test Cases**:
1. **GBPUSD Balanced Distribution**: Train fixed model on GBPUSD, verify test predictions show 30-40% BUY, 30-40% SELL, 20-40% NOISE
2. **XAUUSD Balanced Distribution**: Train fixed model on XAUUSD, verify test predictions show 30-40% BUY, 30-40% SELL, 20-40% NOISE
3. **US30 Balanced Distribution**: Train fixed model on US30, verify test predictions show 30-40% BUY, 30-40% SELL, 20-40% NOISE
4. **EURUSD Data Volume**: Verify EURUSD fetches minimum 99,000 candles
5. **Calibration Improvement**: Verify calibration curve shows predicted probabilities match actual frequencies (well-calibrated)
6. **Walk-Forward Stability**: Verify all fold accuracies within 3-5% of mean (stable performance)

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold (all other operations in the training pipeline), the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL operation IN [feature_engineering, feature_selection, scaling, model_persistence, wfv] DO
  result_original := operation_original(input)
  result_fixed := operation_fixed(input)
  ASSERT result_original == result_fixed
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Observe behavior on UNFIXED code first for feature engineering, feature selection, and model persistence, then write property-based tests capturing that behavior. Verify the fixed code produces identical outputs.

**Test Cases**:
1. **Feature Engineering Preservation**: Verify lagged features, rolling statistics, and delta features are computed identically
2. **Feature Selection Preservation**: Verify SelectKBest, SHAP, and RFE select the same features (given same random seed)
3. **Scaling Preservation**: Verify RobustScaler produces identical scaled values
4. **Model Persistence Preservation**: Verify saved model files (joblib) have identical structure and can be loaded
5. **Walk-Forward Validation Preservation**: Verify WFV logic (fold splitting, per-fold training) works identically
6. **Early Stopping Preservation**: Verify early stopping triggers at the same iteration
7. **Error Handling Preservation**: Verify insufficient data and constant feature errors still raise correctly

### Unit Tests

- Test isotonic calibration on synthetic data with known bias (e.g., all predictions >0.8 should be calibrated to ~0.5-0.7)
- Test scale_pos_weight capping with various class imbalance ratios (1:1, 1:2, 1:5)
- Test EURUSD data fetching to verify minimum 99,000 candles
- Test hyperparameter configuration matches between final model and WFV
- Test calibration set split (80/20) produces correct sizes
- Test SHAP balanced sampling produces equal BUY/SELL samples

### Property-Based Tests

- Generate random training data with various class distributions and verify calibrated predictions fall within expected ranges
- Generate random feature matrices and verify feature engineering produces consistent output shapes
- Generate random model configurations and verify all hyperparameters are applied correctly
- Test that calibration preserves monotonicity (higher raw probability → higher calibrated probability)
- Test that calibration preserves relative ordering of predictions

### Integration Tests

- Train full model on each trading pair (GBPUSD, XAUUSD, US30, EURUSD, USDJPY) and verify balanced prediction distributions
- Run complete training pipeline (data fetch → feature engineering → training → calibration → persistence) and verify all steps complete successfully
- Load saved calibrated model and verify predictions match in-memory model
- Run Walk-Forward Validation with calibrated model and verify stable performance across folds
- Test ensemble engine integration: verify calibrated XGBoost predictions integrate correctly with Random Forest predictions

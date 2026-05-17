# Bugfix Requirements Document

## Introduction

The XGBoost model in the trading bot exhibits severe directional bias toward BUY predictions (95%+ of all predictions are BUY), causing consistent losing trades across all trading pairs. This critical bug affects the core prediction engine and results in the bot opening losing BUY trades continuously, regardless of actual market conditions.

The root cause is a combination of:
1. **Training data temporal bias**: Training data from Jan 2025 - May 2026 captured an uptrend period, causing the model to learn "BUY = profit" as a dominant pattern
2. **Uncalibrated probability outputs**: XGBoost raw probabilities are not properly calibrated, leading to skewed distributions
3. **Ineffective class balancing**: Despite balanced training labels (50/50 BUY/SELL), the model learned directional bias from the temporal patterns in the data
4. **Configuration inconsistency**: Final model uses 500 trees while Walk-Forward Validation uses 300 trees, creating train-test mismatch

This bug impacts all trading pairs (GBPUSD, XAUUSD, US30, EURUSD, USDJPY) and makes the bot unprofitable in non-uptrend market conditions.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN the XGBoost model makes predictions on test data THEN the system outputs 95%+ predictions in the BUY zone (probability > 0.6)

1.2 WHEN the XGBoost model is trained on data from Jan 2025 - May 2026 (uptrend period) THEN the system learns BUY as the dominant profitable pattern regardless of actual feature values

1.3 WHEN the final XGBoost model is configured with n_estimators=500 THEN the system uses a different configuration than Walk-Forward Validation (300 trees), creating train-test mismatch

1.4 WHEN scale_pos_weight is calculated as n_neg/n_pos on balanced training data (50/50) THEN the system sets scale_pos_weight ≈ 1.0, which has no effect on preventing directional bias

1.5 WHEN XGBoost outputs raw probabilities without calibration THEN the system produces uncalibrated probability distributions that do not reflect true class probabilities

1.6 WHEN the model is trained without temporal validation THEN the system fails to detect distribution shift between training period (uptrend) and live trading (mixed market conditions)

1.7 WHEN GBPUSD test data is evaluated THEN the system outputs Weak BUY (0.6-0.8): 7,590 predictions and Strong BUY (>0.8): 11,195 predictions with almost zero SELL predictions

1.8 WHEN XAUUSD test data is evaluated THEN the system outputs Weak BUY (0.6-0.8): 7,559 predictions and Strong BUY (>0.8): 11,176 predictions with almost zero SELL predictions

1.9 WHEN US30 test data is evaluated THEN the system outputs Weak BUY (0.6-0.8): 11,649 predictions and Strong BUY (>0.8): 7,448 predictions with almost zero SELL predictions

1.10 WHEN the Random Forest model outputs 68% predictions in the noise zone (0.45-0.55) THEN the system blocks most signals, forcing reliance on the biased XGBoost model

1.11 WHEN EURUSD is trained with only 20,000 candles THEN the system produces a weaker model compared to other symbols trained on 99,000 candles

1.12 WHEN USDJPY Walk-Forward Validation shows fold accuracies of 58.3%, 57.6%, 53.0%, 55.5%, 60.6% THEN the system demonstrates model instability with Fold 3 dropping to near-random performance (53%)

### Expected Behavior (Correct)

2.1 WHEN the XGBoost model makes predictions on test data THEN the system SHALL output balanced predictions reflecting actual market conditions with approximately 30-40% BUY, 30-40% SELL, and 20-40% NOISE zones

2.2 WHEN the XGBoost model is trained on historical data THEN the system SHALL apply isotonic calibration to correct temporal bias and ensure probability outputs reflect true class distributions independent of training period market regime

2.3 WHEN the final XGBoost model is configured THEN the system SHALL use consistent hyperparameters matching Walk-Forward Validation configuration (n_estimators=300)

2.4 WHEN scale_pos_weight is configured THEN the system SHALL cap the value at 1.2 maximum to prevent extreme directional bias even when training data appears balanced

2.5 WHEN XGBoost outputs probabilities THEN the system SHALL apply isotonic calibration using a held-out calibration set (20% of training data) to produce well-calibrated probability distributions

2.6 WHEN the model is trained THEN the system SHALL perform temporal validation using Walk-Forward Validation with rolling windows to detect and measure distribution shift across different market regimes

2.7 WHEN GBPUSD test data is evaluated THEN the system SHALL output approximately 30-40% BUY predictions, 30-40% SELL predictions, and 20-40% NOISE predictions

2.8 WHEN XAUUSD test data is evaluated THEN the system SHALL output approximately 30-40% BUY predictions, 30-40% SELL predictions, and 20-40% NOISE predictions

2.9 WHEN US30 test data is evaluated THEN the system SHALL output approximately 30-40% BUY predictions, 30-40% SELL predictions, and 20-40% NOISE predictions

2.10 WHEN the Random Forest model is trained THEN the system SHALL reduce noise zone predictions from 68% to approximately 20-30% by improving feature engineering and threshold calibration

2.11 WHEN EURUSD is trained THEN the system SHALL fetch minimum 99,000 candles matching other symbols to ensure consistent model quality across all trading pairs

2.12 WHEN USDJPY Walk-Forward Validation is performed THEN the system SHALL demonstrate stable performance with all fold accuracies within 3-5% of the mean, indicating consistent generalization across market conditions

2.13 WHEN regularization parameters are configured THEN the system SHALL use min_child_weight=30 (raised from 20), reg_alpha=0.2 (raised from 0.1), reg_lambda=1.5 (raised from 1.0), and max_delta_step=1 to prevent overfitting to directional patterns

2.14 WHEN the model training pipeline executes THEN the system SHALL apply SHAP-based feature selection with balanced sampling (50% BUY, 50% SELL) to avoid directional bias in feature importance calculation

### Unchanged Behavior (Regression Prevention)

3.1 WHEN the XGBoost model is trained on valid feature data with proper Target labels THEN the system SHALL CONTINUE TO successfully complete training and produce a trained model

3.2 WHEN the model achieves accuracy above 55% on test data THEN the system SHALL CONTINUE TO save the model to disk for production use

3.3 WHEN Walk-Forward Validation is performed with 5 folds THEN the system SHALL CONTINUE TO report per-fold accuracies, mean accuracy, standard deviation, and stability score

3.4 WHEN lagged features are engineered (lag 1, 3, 5) THEN the system SHALL CONTINUE TO create rolling statistics and delta features for key indicators (RSI, MACD, ATR, ADX)

3.5 WHEN feature selection is performed using SelectKBest THEN the system SHALL CONTINUE TO select top K features based on f_classif scoring

3.6 WHEN RobustScaler is applied THEN the system SHALL CONTINUE TO scale features using robust statistics (median, IQR) to handle outliers

3.7 WHEN the model is saved to disk THEN the system SHALL CONTINUE TO persist model, scaler, and feature names to joblib files (xgb_model.joblib, xgb_scaler.joblib, xgb_features.joblib)

3.8 WHEN the model is loaded from disk THEN the system SHALL CONTINUE TO restore model, scaler, and feature names from saved joblib files

3.9 WHEN early stopping is enabled during training THEN the system SHALL CONTINUE TO monitor eval_set performance and stop training when validation loss stops improving

3.10 WHEN feature importance is calculated THEN the system SHALL CONTINUE TO extract and store feature_importances_ from the trained XGBoost model

3.11 WHEN SHAP feature selection is performed THEN the system SHALL CONTINUE TO compute SHAP values using TreeExplainer and rank features by mean absolute SHAP value

3.12 WHEN RFE (Recursive Feature Elimination) is performed THEN the system SHALL CONTINUE TO iteratively remove weak features using a lightweight XGBoost estimator

3.13 WHEN the ensemble engine calls XGBoost for predictions THEN the system SHALL CONTINUE TO return probability values between 0.0 and 1.0 for the BUY class

3.14 WHEN model retraining is triggered based on time or candle count THEN the system SHALL CONTINUE TO check needs_retraining() and initiate training when thresholds are exceeded

3.15 WHEN training data has fewer than 200 valid rows after NaN removal THEN the system SHALL CONTINUE TO raise ValueError and abort training

3.16 WHEN constant features are detected (std=0) THEN the system SHALL CONTINUE TO remove them before feature selection to prevent sklearn warnings

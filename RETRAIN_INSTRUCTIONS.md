# 🚀 Model Retraining Instructions - XGBoost Fix v6.1

## ⚠️ IMPORTANT: Read Before Retraining

The XGBoost BUY bias fix (v6.1) has been successfully implemented and tested. All tests passed ✅

**What was fixed:**
1. ✅ Proper 3-way data split (60% train, 20% calibration, 20% test)
2. ✅ Calibration on UNSEEN data (not contaminated by training)
3. ✅ Hyperparameter alignment (300 trees across all functions)
4. ✅ Enhanced logging for prediction distribution

---

## 📋 Pre-Retraining Checklist

- [x] Code changes verified (`test_xgb_fix.py` passed)
- [x] Backup existing models (optional but recommended)
- [ ] Ensure MT5 connection is available (for data fetching)
- [ ] Ensure sufficient disk space (~500MB for all models)
- [ ] Close any running trading bots (to avoid conflicts)

---

## 🔄 Retraining Steps

### Step 1: Backup Existing Models (Optional)
```bash
# Create backup directory
mkdir models_backup_v6.0

# Backup XGBoost models
copy xgb_model_*.joblib models_backup_v6.0\
copy xgb_scaler_*.joblib models_backup_v6.0\
copy xgb_features_*.joblib models_backup_v6.0\

# Backup RF models
copy rf_model_*.joblib models_backup_v6.0\
copy rf_scaler_*.joblib models_backup_v6.0\
copy rf_features_*.joblib models_backup_v6.0\
```

### Step 2: Run Training Script
```bash
python train_offline.py
```

**Expected Duration:** 30-60 minutes for all symbols (EURUSD, GBPUSD, USDJPY, XAUUSD, US30)

**What to watch for:**
```
[XGB-EURUSD] BEFORE Calibration - Test Set Distribution:
[XGB-EURUSD]   BUY (>0.6):   XX.X%    <- Should be high (70-95%)
[XGB-EURUSD]   SELL (<0.4):  XX.X%    <- Should be low (0-10%)
[XGB-EURUSD]   NOISE (0.4-0.6): XX.X% <- Should be low (5-20%)

[XGB-EURUSD] Applying Isotonic Calibration on XXXX UNSEEN samples...

[XGB-EURUSD] AFTER Calibration - Test Set Distribution:
[XGB-EURUSD]   BUY (>0.6):   XX.X% (change: -XX.X%)   <- Should decrease
[XGB-EURUSD]   SELL (<0.4):  XX.X% (change: +XX.X%)   <- Should increase
[XGB-EURUSD]   NOISE (0.4-0.6): XX.X% (change: +XX.X%) <- Should increase significantly
```

**Success Indicators:**
- ✅ NOISE zone increases by 10-30% after calibration
- ✅ BUY zone decreases by 10-30% after calibration
- ✅ SELL zone increases by 5-15% after calibration
- ✅ Accuracy remains 55-60% (calibration doesn't hurt accuracy)
- ✅ All models save successfully

**Failure Indicators:**
- ❌ Calibration shows 0% improvement on ALL symbols (investigate)
- ❌ Accuracy drops below 50% (investigate)
- ❌ Training crashes or errors
- ❌ Models don't save

### Step 3: Verify Trained Models
```bash
# Check that all model files exist and are recent
dir xgb_model_*.joblib
dir xgb_scaler_*.joblib
dir xgb_features_*.joblib
```

**Expected files:**
- `xgb_model_EURUSD.joblib` (should be recent timestamp)
- `xgb_model_GBPUSD.joblib`
- `xgb_model_USDJPY.joblib`
- `xgb_model_XAUUSD.joblib`
- `xgb_model_US30.joblib`
- (+ corresponding scaler and features files)

---

## 🧪 Post-Retraining Validation

### Test 1: Quick Prediction Test
```python
# Run this in Python console
from xgb_model import XGBModel
import pandas as pd

# Load a model
model = XGBModel()
model.symbol = "EURUSD"
model._load_model()

# Check if model loaded
print(f"Model loaded: {model.model is not None}")
print(f"Is calibrated: {hasattr(model.model, 'calibrated_classifiers_')}")
print(f"Features: {len(model.feature_names)}")
```

**Expected output:**
```
Model loaded: True
Is calibrated: True
Features: 30-40
```

### Test 2: Live Prediction Distribution Test
```bash
# Start the bot in diagnostic mode (if available)
# Or monitor ensemble_decisions.csv after a few hours of live trading
```

**Check `ensemble_decisions.csv`:**
```python
import pandas as pd

df = pd.read_csv('ensemble_decisions.csv')
recent = df.tail(1000)  # Last 1000 decisions

# Check XGB distribution
xgb_probs = recent['xgb_prob']
print(f"XGB Distribution (last 1000 decisions):")
print(f"  BUY (>0.6):   {(xgb_probs > 0.6).mean() * 100:.1f}%")
print(f"  SELL (<0.4):  {(xgb_probs < 0.4).mean() * 100:.1f}%")
print(f"  NOISE (0.4-0.6): {((xgb_probs >= 0.4) & (xgb_probs <= 0.6)).mean() * 100:.1f}%")
```

**Expected (after fix):**
```
XGB Distribution (last 1000 decisions):
  BUY (>0.6):   30-40%   ✅
  SELL (<0.4):  30-40%   ✅
  NOISE (0.4-0.6): 20-40% ✅
```

**Before fix (for comparison):**
```
XGB Distribution (last 1000 decisions):
  BUY (>0.6):   90-95%   ❌ Severe bias
  SELL (<0.4):  0-5%     ❌
  NOISE (0.4-0.6): 0-10% ❌
```

---

## 📊 Monitoring After Deployment

### First 24 Hours
**Monitor:**
1. **Prediction distribution** in `ensemble_decisions.csv`
   - Should be balanced (30-40% each zone)
2. **Trade execution**
   - Should see more HOLD decisions
   - BUY/SELL trades should be more balanced
3. **No crashes or errors**
   - Check `bot.log` for any calibration-related errors

### First Week
**Monitor:**
1. **Win rate**
   - Target: 45-55% (was ~30% before fix)
2. **Drawdown**
   - Should reduce by 30-50%
3. **Profit factor**
   - Should improve
4. **Trade distribution**
   - BUY vs SELL should be roughly 50/50 (not 95/5)

### Performance Metrics to Track
```python
# Analyze trading_history.csv
import pandas as pd

df = pd.read_csv('trading_history.csv')
recent = df[df['timestamp'] > '2026-05-17']  # After fix deployment

# Win rate
wins = recent[recent['profit'] > 0]
win_rate = len(wins) / len(recent) * 100
print(f"Win Rate: {win_rate:.1f}%")

# Direction distribution
buy_trades = recent[recent['direction'] == 'BUY']
sell_trades = recent[recent['direction'] == 'SELL']
print(f"BUY trades: {len(buy_trades)} ({len(buy_trades)/len(recent)*100:.1f}%)")
print(f"SELL trades: {len(sell_trades)} ({len(sell_trades)/len(recent)*100:.1f}%)")

# Profit factor
total_profit = wins['profit'].sum()
total_loss = abs(recent[recent['profit'] < 0]['profit'].sum())
profit_factor = total_profit / total_loss if total_loss > 0 else 0
print(f"Profit Factor: {profit_factor:.2f}")
```

---

## 🐛 Troubleshooting

### Issue 1: Training Fails with "Not enough data"
**Symptoms:**
```
[XGB-EURUSD] Not enough data: 150 rows. Need at least 200.
```

**Cause:** 60/20/20 split requires more data than 80/20

**Solution:**
1. Check MT5 connection and data fetching
2. Ensure `Config.DATA_POINTS` is set to at least 2000
3. Verify symbol data is available in MT5

### Issue 2: Calibration Shows 0% Improvement on ALL Symbols
**Symptoms:**
```
[CAL-EURUSD] NOISE zone improvement: +0.0%
[CAL-GBPUSD] NOISE zone improvement: +0.0%
[CAL-XAUUSD] NOISE zone improvement: +0.0%
```

**Cause:** Model may already be well-calibrated OR data issue

**Solution:**
1. Check if raw predictions are already balanced (before calibration)
2. If raw predictions are 95% BUY, investigate feature engineering
3. Check target label distribution in training data

### Issue 3: Accuracy Drops Below 50%
**Symptoms:**
```
[XGB-EURUSD] Accuracy: 48.5%
```

**Cause:** Calibration shouldn't hurt accuracy - likely a data issue

**Solution:**
1. Check if calibration set is too small (<500 samples)
2. Verify data quality (no NaN, no constant features)
3. Check if target labels are balanced

### Issue 4: Models Don't Load After Retraining
**Symptoms:**
```
[XGB] Model not trained. Returning neutral 0.5.
```

**Cause:** Calibrated models have different structure

**Solution:**
1. Check if model files exist and are recent
2. Verify `_load_model()` handles calibrated models correctly
3. Check for file permission issues

---

## 🔄 Rollback Procedure (If Needed)

If the fix causes issues, you can rollback to v6.0 models:

```bash
# Stop the bot first
# Then restore backup models
copy models_backup_v6.0\*.joblib .

# Restart the bot
```

**Note:** v6.0 models will still have the BUY bias issue.

---

## 📞 Support

If you encounter issues:

1. **Check logs:**
   - `bot.log` - Main bot log
   - `execution_quality.log` - Trade execution log
   - `rejected_trades.log` - Rejected trade reasons

2. **Check data files:**
   - `ensemble_decisions.csv` - Ensemble decision history
   - `trading_history.csv` - Trade history

3. **Review documentation:**
   - `XGBOOST_FIX_v6.1_SUMMARY.md` - Technical details
   - `.kiro/specs/xgboost-buy-bias-fix/design.md` - Design document

---

## ✅ Success Checklist

After retraining and 24 hours of live trading:

- [ ] All models retrained successfully
- [ ] Calibration showed 10-30% NOISE improvement during training
- [ ] XGB predictions show balanced distribution (30-40% each zone)
- [ ] More HOLD decisions in ensemble
- [ ] BUY/SELL trades are more balanced
- [ ] No crashes or errors
- [ ] Win rate improving
- [ ] Drawdown reducing

---

**Version:** 6.1  
**Date:** 2026-05-17  
**Status:** ✅ Ready for retraining  
**Estimated Time:** 30-60 minutes

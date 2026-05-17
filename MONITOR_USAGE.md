# 📊 XGBoost Fix Monitor - Usage Guide

## Overview

`monitor_xgb_fix.py` is a comprehensive monitoring tool that tracks the effectiveness of the XGBoost BUY bias fix (v6.1). It analyzes prediction distributions, trading performance, and temporal trends.

---

## Quick Start

### Basic Usage
```bash
# Full analysis (recommended)
python monitor_xgb_fix.py

# Quick summary only
python monitor_xgb_fix.py --quick

# Analyze specific symbol
python monitor_xgb_fix.py --symbol EURUSD

# Analyze last 7 days only
python monitor_xgb_fix.py --days 7

# Combine options
python monitor_xgb_fix.py --symbol GBPUSD --days 3
```

---

## What It Analyzes

### 1. 📊 XGBoost Prediction Distribution
**Shows:**
- Percentage of predictions in BUY zone (>0.6)
- Percentage of predictions in SELL zone (<0.4)
- Percentage of predictions in NOISE zone (0.4-0.6)
- Histogram of probability distribution

**Expected (after fix):**
```
✅ BUY (>0.6):      35.0%  (expected: 30-40%)
✅ SELL (<0.4):     35.0%  (expected: 30-40%)
✅ NOISE (0.4-0.6): 30.0%  (expected: 20-40%)
```

**Before fix (for comparison):**
```
❌ BUY (>0.6):      95.0%  (expected: 30-40%)
❌ SELL (<0.4):      2.0%  (expected: 30-40%)
❌ NOISE (0.4-0.6):  3.0%  (expected: 20-40%)
```

### 2. 🎯 Ensemble Decision Analysis
**Shows:**
- Total decisions made
- Breakdown by direction (BUY/SELL/HOLD)
- Top skip reasons (why trades were not taken)

**Example output:**
```
Total decisions: 1,234

Decision breakdown:
  BUY   :    456 ( 37.0%)
  SELL  :    432 ( 35.0%)
  HOLD  :    346 ( 28.0%)
```

### 3. 💰 Trading Performance Analysis
**Shows:**
- Win rate (percentage of profitable trades)
- Profit factor (total profit / total loss)
- Net profit/loss
- Trade direction balance (BUY vs SELL)

**Example output:**
```
Win Rate: 52.5% (105 wins / 95 losses)
✅ Excellent win rate!

Profit Metrics:
  Total Profit:  $1,250.00
  Total Loss:    $850.00
  Net Profit:    $400.00
  Profit Factor: 1.47

Trade Direction Balance:
  BUY:  48.5% (97 trades)
  SELL: 51.5% (103 trades)
✅ Well balanced (diff: 3.0%)
```

### 4. 📈 Temporal Trend Analysis
**Shows:**
- Daily breakdown of prediction distribution
- Trend over time (improving/worsening)
- Last 14 days of data

**Example output:**
```
Date         Decisions    BUY%   SELL%  NOISE%  Status
──────────────────────────────────────────────────────────
2026-05-10       1,234    65.0%   15.0%   20.0%  ⚠️  Improving
2026-05-11       1,456    55.0%   22.0%   23.0%  ⚠️  Improving
2026-05-12       1,389    42.0%   30.0%   28.0%  ⚠️  Improving
2026-05-13       1,512    38.0%   34.0%   28.0%  ✅ Balanced
2026-05-14       1,445    36.0%   35.0%   29.0%  ✅ Balanced

Trend:
✅ BUY bias is decreasing (improving)
```

### 5. 📋 Summary Report
**Shows:**
- Overall fix status (working/not working/partial)
- Key recommendations
- Action items if needed

---

## Understanding the Output

### Color Coding

- 🟢 **Green (✅)**: Good - within expected range
- 🟡 **Yellow (⚠️)**: Warning - acceptable but could be better
- 🔴 **Red (❌)**: Critical - needs attention

### Status Indicators

#### Prediction Distribution
- **✅ Balanced**: BUY 30-40%, SELL 30-40%, NOISE 20-40%
- **⚠️ Improving**: BUY 40-60%, showing improvement trend
- **❌ BUY bias**: BUY >70%, fix not working

#### Win Rate
- **✅ Excellent**: ≥50%
- **⚠️ Acceptable**: 40-50%
- **❌ Low**: <40%

#### Profit Factor
- **✅ Strong**: ≥1.5
- **⚠️ Profitable**: 1.0-1.5
- **❌ Unprofitable**: <1.0

#### Direction Balance
- **✅ Well balanced**: Difference <20%
- **⚠️ Moderate imbalance**: Difference 20-40%
- **❌ Severe imbalance**: Difference >40%

---

## When to Run

### Recommended Schedule

1. **Immediately after retraining** (Day 0)
   ```bash
   python monitor_xgb_fix.py --quick
   ```
   - Verify models loaded correctly
   - Check initial distribution

2. **After 24 hours** (Day 1)
   ```bash
   python monitor_xgb_fix.py
   ```
   - Full analysis of first day
   - Verify fix is working

3. **Daily for first week** (Days 2-7)
   ```bash
   python monitor_xgb_fix.py --days 7
   ```
   - Track trends
   - Ensure stability

4. **Weekly thereafter**
   ```bash
   python monitor_xgb_fix.py --days 30
   ```
   - Long-term monitoring
   - Detect any regressions

---

## Interpreting Results

### Scenario 1: Fix is Working ✅
```
XGBoost Fix Status:
  ✅ FIX IS WORKING - Distribution is balanced!

Trading Performance:
  ✅ Win rate: 52.5% - Excellent!

Recommendations:
  • Continue monitoring - system is working well
  • Focus on optimizing other parameters
```

**Action:** None needed. System is working as expected.

---

### Scenario 2: Partial Fix ⚠️
```
XGBoost Fix Status:
  ⚠️  PARTIAL FIX - Moderate BUY bias remains

Trading Performance:
  ⚠️  Win rate: 45.0% - Acceptable

Recommendations:
  • Monitor for 2-3 more days to see if it stabilizes
  • Check if recent market conditions are unusual
```

**Action:** 
- Wait 2-3 more days
- Check if market is in strong uptrend (may cause temporary bias)
- If persists, investigate further

---

### Scenario 3: Fix Not Working ❌
```
XGBoost Fix Status:
  ❌ FIX NOT WORKING - Severe BUY bias persists

Trading Performance:
  ❌ Win rate: 32.0% - Needs improvement

Recommendations:
  • Verify models were retrained with v6.1 fix
  • Check training logs for calibration effectiveness
  • Consider re-running train_offline.py
```

**Action:**
1. Check model file timestamps - are they recent?
   ```bash
   dir xgb_model_*.joblib
   ```

2. Check training logs for calibration messages:
   ```
   [XGB-EURUSD] BEFORE Calibration - Test Set Distribution:
   [XGB-EURUSD] AFTER Calibration - Test Set Distribution:
   ```

3. If models are old or calibration didn't run, retrain:
   ```bash
   python train_offline.py
   ```

---

## Troubleshooting

### Error: "ensemble_decisions.csv not found"
**Cause:** Bot hasn't been running or hasn't generated decisions yet

**Solution:**
1. Start the bot: `python main.py`
2. Wait for at least 1 hour of trading
3. Run monitor again

---

### Error: "No data found in the specified time range"
**Cause:** Using `--days` filter that's too restrictive

**Solution:**
- Remove `--days` filter to see all data
- Or use larger value: `--days 30`

---

### Warning: "trading_history.csv not found"
**Cause:** No trades have been closed yet

**Solution:**
- This is normal if bot just started
- Trading performance metrics will be available after first closed trade
- Other analyses (prediction distribution) will still work

---

### Distribution shows 0% everywhere
**Cause:** `xgb_prob` column is missing or empty

**Solution:**
1. Check if bot is using XGBoost (not LSTM)
2. Verify ensemble_engine.py is logging XGBoost probabilities
3. Check bot.log for errors

---

## Advanced Usage

### Compare Before/After Fix
```bash
# Analyze data before fix (if you have old data)
python monitor_xgb_fix.py --days 30

# Look at the temporal trend to see improvement over time
```

### Monitor Specific Symbol
```bash
# If one symbol is problematic
python monitor_xgb_fix.py --symbol XAUUSD

# Compare with another symbol
python monitor_xgb_fix.py --symbol EURUSD
```

### Automated Monitoring
```bash
# Add to cron/task scheduler for daily reports
# Windows Task Scheduler:
# - Run: python monitor_xgb_fix.py --quick > daily_report.txt
# - Schedule: Daily at 23:00
```

---

## Output Files

The monitor reads from these files:
- `ensemble_decisions.csv` - Required (ensemble decision history)
- `trading_history.csv` - Optional (trade execution history)

It does NOT create any files - it only reads and analyzes.

---

## Tips

1. **Run after retraining**: Always run immediately after `train_offline.py` to verify fix

2. **Check trends**: Don't judge by single day - look at 3-7 day trends

3. **Market conditions matter**: Strong trending markets may temporarily skew distribution

4. **Compare symbols**: If one symbol shows bias but others don't, investigate that symbol specifically

5. **Save reports**: Redirect output to file for historical comparison:
   ```bash
   python monitor_xgb_fix.py > report_2026-05-17.txt
   ```

---

## Expected Timeline

### Day 0 (Immediately after retraining)
- Models should load successfully
- Initial predictions may still show some bias (normal)

### Day 1-3
- Distribution should start balancing
- BUY bias should decrease from 95% to 50-60%

### Day 4-7
- Distribution should stabilize at 30-40% each zone
- Win rate should improve to 45-55%

### Week 2+
- System should be fully stable
- Consistent balanced distribution
- Improved trading performance

---

## Support

If monitor shows persistent issues:

1. **Check documentation:**
   - `XGBOOST_FIX_v6.1_SUMMARY.md` - Technical details
   - `RETRAIN_INSTRUCTIONS.md` - Retraining guide

2. **Check logs:**
   - `bot.log` - Main bot log
   - Training output from `train_offline.py`

3. **Verify fix was applied:**
   - Check `xgb_model.py` for v6.1 changes
   - Run `test_xgb_fix.py` to verify code

---

**Version:** 6.1  
**Last Updated:** 2026-05-17  
**Maintainer:** Kiro AI

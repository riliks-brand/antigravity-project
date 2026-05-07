# 🤖 ELITE TRADING BOT v4.0 — COMPLETE HANDOVER DOCUMENT

> **Purpose**: This document gives any AI assistant 100% understanding of this project — every file, every decision, every formula, every data flow.

---

## 📋 TABLE OF CONTENTS
1. [Project Identity](#1-project-identity)
2. [Architecture Overview](#2-architecture-overview)
3. [File Map & Responsibilities](#3-file-map--responsibilities)
4. [Data Flow Pipeline](#4-data-flow-pipeline)
5. [Ensemble Engine Deep Dive](#5-ensemble-engine-deep-dive)
6. [Trade Management Lifecycle](#6-trade-management-lifecycle)
7. [Feature Engineering Details](#7-feature-engineering-details)
8. [Risk Management System](#8-risk-management-system)
9. [Configuration Reference](#9-configuration-reference)
10. [ML Models](#10-ml-models)
11. [Key Formulas & Thresholds](#11-key-formulas--thresholds)
12. [Known Issues & Evolution History](#12-known-issues--evolution-history)
13. [How to Run](#13-how-to-run)
14. [Dependencies](#14-dependencies)

---

## 1. PROJECT IDENTITY

| Field | Value |
|-------|-------|
| **Name** | Elite MT5 Trading Bot v4.0 (XGBoost Ensemble Edition) |
| **Language** | Python 3.8+ |
| **Platform** | Windows (MetaTrader 5 native) |
| **Broker** | MetaQuotes-Demo (Account: 5049001425) |
| **Directory** | `D:\Candlestick_Detection` |
| **Entry Point** | `main.py` |
| **Account Mode** | MICRO ($10 simulated start, 0.01 lot forced) |

**What it does**: Automated multi-symbol forex/commodity/index trading bot that uses an XGBoost + Random Forest ensemble to generate BUY/SELL signals, executes them on MT5, and manages positions with trailing stops, partial closes, and AI-driven smart exits.

**Symbols Traded**: EURUSD, GBPUSD, USDJPY, XAUUSD, US30, BTCUSD

---

## 2. ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────┐
│                      MAIN LOOP (main.py) v4.0                    │
│  Every 10s: tick management + hybrid event/candle evaluation     │
└───────────┬─────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────┐     ┌──────────────────────┐
│  DATA LOADER      │────▶│  FEATURE ENGINEERING  │
│  (data_loader.py) │     │  (features.py)        │
│  M5 + M15 + H1    │     │  + candles.py          │
│  from MT5 native   │     │  + pattern_detector.py │
└───────────────────┘     │  + divergence.py       │
                          └──────────┬───────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                  ▼
           ┌──────────────┐                   ┌──────────────┐
           │  LSTM MODEL   │                   │  RF MODEL     │
           │ (lstm_model.py)│                   │ (rf_model.py) │
           │  Sequential    │                   │  De-correlated│
           │  Time-series   │                   │  Interaction  │
           └──────┬───────┘                   └──────┬───────┘
                  │                                    │
                  ▼                                    ▼
           ┌─────────────────────────────────────────────┐
           │         ENSEMBLE ENGINE (ensemble_engine.py) │
           │  Dynamic Weighted Soft Voting                │
           │  Session-Aware + Market-Adaptive              │
           │  17-Step Decision Pipeline                    │
           └──────────────────┬──────────────────────────┘
                              │
                              ▼
           ┌─────────────────────────────────────────────┐
           │          HYBRID FILTERS (main.py)            │
           │  ADX + Volatility + News + Session + Spread  │
           │  + Memory Similarity + Context Boosts         │
           └──────────────────┬──────────────────────────┘
                              │
                              ▼
           ┌──────────────────────────────────────────────┐
           │        PORTFOLIO RANKER (main.py)             │
           │  Rank all opportunities → Execute top picks   │
           │  Correlation filter + USD diversification     │
           └──────────────────┬───────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      ┌──────────────┐ ┌──────────┐ ┌──────────────┐
      │ MT5 ENGINE    │ │ TRADE    │ │ NOTIFIER     │
      │(mt5_engine.py)│ │ MANAGER  │ │(notifier.py) │
      │ Execute order │ │(trade_   │ │ Telegram     │
      │ Lot sizing    │ │manager.py│ │ Rate-limited │
      └──────────────┘ │ Trailing │ └──────────────┘
                        │ Partial  │
                        │ SmartExit│
                        └──────────┘
```

---

## 3. FILE MAP & RESPONSIBILITIES

### Core Runtime Files
| File | Lines | Role |
|------|-------|------|
| `main.py` | 726 | **Master orchestrator**. Main loop, portfolio evaluation, opportunity ranking, execution dispatch |
| `config.py` | 256 | **All configuration**. Credentials, thresholds, risk params, session times, feature flags |
| `ensemble_engine.py` | 880 | **The Brain**. 17-step decision pipeline: weighting → scoring → filtering → threshold → confidence |
| `trade_manager.py` | 1383 | **Trade lifecycle**. Registration, tick management, partial close, trailing stop, smart exit, guards, persistence |
| `mt5_engine.py` | 448 | **Execution engine**. MT5 connection, lot sizing, order sending, kill switch, heartbeat |
| `data_loader.py` | 273 | **Data source**. Multi-timeframe OHLC from MT5 (M5/M15/H1), tick data, market open check |
| `features.py` | 534 | **Feature pipeline**. 80+ features: technicals, SMC, MTF injection, drift detection, target generation |

### ML Models
| File | Role |
|------|------|
| `lstm_model.py` | LSTM architecture (128→64→32→1 sigmoid), training with loss-weighted samples |
| `rf_model.py` | Random Forest with DE-CORRELATED features (interaction terms, rolling stats, divergences) |
| `train_offline.py` | Offline training script for both LSTM and RF on 50K candles of EURUSD |

### Phase 1: PDF Vision Layer (Pattern Detectors)
| File | Features Added |
|------|---------------|
| `candles.py` | 16 candlestick patterns + composite reversal score (Engulfing, Hammer, Doji, Morning/Evening Star, etc.) |
| `pattern_detector.py` | 14 chart patterns + composite bias score + volatility squeeze (H&S, Double Top/Bottom, Triangles, Flags, Wedges) |
| `divergence.py` | 4 RSI divergence types + composite score (Regular/Hidden Bull/Bear divergence) |

### Phase 2: Smart Exit
| File | Role |
|------|------|
| `smart_exit.py` | AI-driven dynamic exits using counter-signal danger scoring from Phase 1 detectors |

### Phase 3: Macro & Filters
| File | Role |
|------|------|
| `macro_context.py` | DXY Dollar Strength (-1 to +1) via native or synthetic DXY basket |
| `news_filter.py` | ForexFactory scraper, blocks trading ±15min around high-impact news |
| `notifier.py` | Telegram notifications (rate-limited, spam-controlled, daily summaries) |

### Support Files
| File | Role |
|------|------|
| `run_bot.ps1` | PowerShell auto-restart wrapper (max 10 restarts, 30s cooldown) |
| `setup_telegram.py` | Interactive Telegram bot setup wizard |
| `sentiment_analyzer.py` | Sentiment analysis module (not actively used in main loop) |
| `executor.py` | Legacy/extended executor (57KB, historical) |
| `otc_scraper.py` | Legacy OTC binary options scraper (not used in current MT5 mode) |

### Persisted Model Files
| File | Purpose |
|------|---------|
| `lstm_model.h5` | Trained LSTM weights (Keras H5 format) |
| `lstm_scaler.joblib` | RobustScaler fitted on training features |
| `rf_model.joblib` | Trained RandomForest model |
| `rf_scaler.joblib` | RobustScaler for RF features |
| `rf_features.joblib` | Feature name list for RF column alignment |

### Log & State Files
| File | Purpose |
|------|---------|
| `active_trades.json` | Persisted active trade state (crash recovery) |
| `bot.log` | Master log file (all modules) |
| `ensemble_decisions.csv` | Every ensemble decision logged with all fields |
| `smart_exit_log.csv` | Smart exit evaluations |
| `execution_quality.log` | Slippage & latency tracking |
| `rejected_trades.log` | Rejected signals for over-filtering analysis |
| `trading_history.csv` | Closed trade history with market context |

---

## 4. DATA FLOW PIPELINE

### Main Loop Cycle (every 10 seconds)

```
1. HEARTBEAT CHECK → MT5 alive? Reconnect if dead
2. KILL SWITCH CHECK → Daily loss > 5%? Close all, pause until tomorrow
3. DAILY RESET → At 00:00 UTC, reset daily stats
4. TICK MANAGEMENT → For each symbol: update trailing stops, check TP/SL hits
5. EVALUATION (every 10s, forced on candle close or market events):
   a. Fetch MTF data (M5 + M15 + H1) from MT5
   b. Run feature_engineering_pipeline() → 80+ features
   c. Smart Exit check for open trades
   d. LSTM prediction (sequence of last 60 candles)
   e. RF prediction (de-correlated features, latest row)
   f. Ensemble prediction (17-step pipeline)
   g. Dual evaluation (original + diagnostic mode)
   h. Memory similarity check
   i. Context boosts (H1 trend alignment, volatility)
   j. Hybrid filters (ADX, volatility, news, session, spread)
   k. Collect as opportunity with rank score
6. PORTFOLIO RANKING → Sort opportunities by score, execute top 2 per cycle
7. EXECUTION GUARDS → Cooldown, correlation, USD exposure, max trades, dedup
8. MT5 ORDER → Build request, send, register trade in manager
9. NOTIFICATIONS → Telegram alert for trade opened
```

### Hybrid Evaluation Trigger (NOT just candle close)
The bot evaluates on:
- **Candle close** (every 5 minutes, M5 timeframe)
- **Price spike** > 0.5 × ATR from last close
- **Order Block proximity** (price near bullish/bearish OB)
- **FVG fill** (price enters Fair Value Gap)
- **Structure break** (price exceeds past 10-bar high/low)

---

## 5. ENSEMBLE ENGINE DEEP DIVE

### The 17-Step Decision Pipeline

```
Step 1:  Trend Strength = clip((ADX - 20) / 30, 0, 1)
Step 2:  ATR Double Filter: ratio < 0.5 OR absolute < threshold → SKIP
Step 3:  Dynamic Weights: LSTM = 0.5 + (ts × 0.3), RF = 0.5 - (ts × 0.3)
Step 4:  Weighted Average = (LSTM_w × lstm_prob) + (RF_w × rf_prob)
Step 5:  Conflict Detection:
         - |LSTM - RF| >= 0.60 → HARD BLOCK
         - 0.45-0.60 + opposite directions → HARD BLOCK
         - 0.45-0.60 + same direction → proceed with penalty
Step 6:  Disagreement Penalty = |LSTM - RF| × 0.30
         base_score = weighted_avg ± penalty (pushes toward 0.5)
Step 7:  Side Tracking: BUY if base > 0.5, SELL if base < 0.5
Step 8:  Weak Zone Threshold: Asia = 0.02, others = 0.01
Step 9:  Score Floor: distance < 0.002 from 0.5 → REJECT
Step 10: Weak Zone: distance < threshold → NO ENTRY
Step 11: Regime Conflict: London+ranging or Asia+trending → BLOCK (with override for strong signals)
Step 12: Additive Scoring:
         - Session Bonus: [-0.03, +0.03]
         - Volatility Adjustment: [-0.02, +0.02]
         - Event Boost: up to +0.04 (OB, FVG, liquidity sweep)
         - MTF Penalty: -0.03 if against H1 trend
         - DXY Influence: [-0.02, +0.02] for USD pairs
         - Total capped at [-0.04, +0.04] AND ≤ 8% of base_score
Step 13: Dynamic Thresholds:
         - Strong trend (ts > 0.35): BUY > 0.55, SELL < 0.45
         - Moderate (ts > 0.25): BUY > 0.56, SELL < 0.44
         - Weak: BUY > 0.58, SELL < 0.42
Step 15: Confidence: HIGH (dist > 0.15), MEDIUM (> 0.08), LOW
Step 16: CSV Logging (every decision, mandatory)
Step 17: Console Print
```

### Key Design Decisions
- **Near-Miss DISABLED**: Every losing trade came through near-miss activation. Quality > quantity.
- **Smooth weighting, NOT binary**: Trend strength continuously interpolates LSTM/RF weights.
- **Additive scoring, NOT multiplicative**: Adjustments never overpower model signals (≤8% of base).
- **Session-aware strategy**: London=trend-following, NY=balanced, Asia=mean-reversion.

---

## 6. TRADE MANAGEMENT LIFECYCLE

```
SIGNAL → OPEN → PARTIAL_CLOSED → CLOSED
                  ↑ (TP1 hit)
```

### On Every Tick (`on_tick`):
1. Check TP1 hit → partial close 50%, move SL to breakeven, activate trailing
2. Update trailing stop (ATR × 1.0 distance)
3. Check SL hit (including trailing SL)
4. Check TP2 hit (final target)

### Smart Exit (on candle close):
- Scans Phase 1 detectors for counter-trend signals
- Each signal has a weight (e.g., RSI_BearDiv = 3.0, Engulf_Bear = 2.0)
- Danger score ≥ 3.0 → close trade early
- Danger score ≥ 2.0 → tighten trailing stop
- Only triggers if trade is in profit AND open > 3 candles

### Guards (before execution):
1. Global cooldown (30 min after 3 consecutive losses)
2. Symbol dynamic cooldown (proportional to ATR)
3. Session limits (max 10 trades/session, max 5 near-miss)
4. Max concurrent trades (3)
5. Daily loss kill switch (5%)
6. Signal deduplication (min 3 candles between same-direction trades)
7. Correlation filter (EURUSD/GBPUSD same direction blocked, etc.)
8. USD exposure limit (max 2 concurrent USD trades)
9. Portfolio risk cap (max 3% total risk)

### Persistence:
- `active_trades.json` — atomic write (temp file → rename)
- On crash: rebuilds state from MT5 positions by magic number
- `trading_history.csv` — every closed trade with full market context

---

## 7. FEATURE ENGINEERING DETAILS

### Features Generated (80+)

**Technical Indicators**: RSI(14), MACD(12,26,9), Bollinger Bands(20,2), ATR(14), Volatility (ATR/close)

**Trend**: EMA_50, EMA_200, trend_direction, ema_spread, ADX(14), DI+, DI-, is_trending

**Momentum**: ROC_5, ROC_10, ROC_20, momentum_agreement

**Support/Resistance**: Pivot, R1, R2, S1, S2, distances normalized by ATR

**Sessions**: is_london, is_ny, is_asia, session_overlap

**Price Action**: body_size, upper/lower_shadow, shadow_ratios, body_direction, candle_range_atr

**Smart Money Concepts (SMC)**:
- Market Structure: higher_high, lower_low, structure_trend, bos_strength
- Order Blocks: distance_to_ob, inside_ob_zone, ob_strength
- Fair Value Gaps: fvg_size, distance_to_fvg, fvg_filled
- Liquidity: equal_highs_count, liquidity_sweep_flag

**Candlestick Patterns** (16): Engulfing, Hammer, Doji variants, Morning/Evening Star, Piercing/DarkCloud, Three Soldiers/Crows, Spinning Top, Marubozu + composite `candle_reversal_score`

**Chart Patterns** (14): Double/Triple Top/Bottom, H&S, Triangles, Wedges, Flags, Volatility Squeeze + composite `pattern_bias_score`

**Divergences** (4): RSI Bull/Bear + Hidden Bull/Bear + composite `divergence_score`

**Multi-Timeframe**: M15_RSI, M15_trend, H1_EMA_50, H1_EMA_200, H1_trend, H1_ADX

**Time**: hour, day_of_week

**Target Generation**: BUY(1) if future_move > ATR_LOOKAHEAD_MULT×ATR, SELL(0) if < -ATR_LOOKAHEAD_MULT×ATR, HOLD(NaN) filtered out. The `ATR_LOOKAHEAD_MULT` is per-symbol (1.2 for forex, 1.5 for Gold/US30, 1.8 for BTC). Target is generated as the **last step** in the feature pipeline — no future data leaks into features.

### RF De-Correlated Features (separate from LSTM):
- Cross-products: RSI×ATR, ADX×Vol, MACD×RSI, BB_pos×ADX, body_ATR_ratio
- Rolling stats: close_std, close_zscore, RSI_std, ATR_mean (windows 10,20,50)
- Returns: skewness, kurtosis
- Divergence signals: price_rsi_divergence, macd_price_agree
- Ratios: shadow_ratio, ema_ratio, di_ratio
- Lagged deltas: RSI/ADX/ATR/MACD delta_1, delta_5, acceleration
- Vol regime: ATR / ATR_MA50

---

## 8. RISK MANAGEMENT SYSTEM

### Lot Sizing
- **MICRO MODE** (active): Forces broker minimum lot (0.01), simulates $10 balance
- **Standard**: `lot = (balance × risk%) / (SL_points × tick_value)`, clamped to broker limits

### SL/TP Calculation
- Default: SL = 1.5×ATR, TP1 = 1.5×ATR, TP2 = 2.0×ATR
- **MICRO overrides**: SL = 2.0×ATR, TP1 = 3.0×ATR, TP2 = 4.0×ATR (RR = 1.5:1)
- **XAUUSD overrides**: SL = 3.0×ATR, TP1 = 3.0×ATR, TP2 = 4.5×ATR
- **US30 overrides**: SL = 2.5×ATR, TP1 = 2.5×ATR, TP2 = 3.5×ATR

### Adaptive Risk (`get_adaptive_risk`):
- Strong trend + HIGH confidence → 2.0% (micro) / 1.0% (standard)
- Otherwise → 1.0% (micro) / 0.5% (standard)
- Regime just changed → ×0.7
- DXY contradicts → ×0.5
- Near-miss signal (confidence_level == "LOW") → ×0.25 (`NEAR_MISS_RISK_REDUCTION`)
- Drawdown > 2% → ×0.5 (survival mode)
- Equity below MA(20) → ×0.5

### Kill Switches
- Daily loss > 5% → close ALL positions, pause until next day
- Portfolio risk cap: total open risk cannot exceed 3%

---

## 9. CONFIGURATION REFERENCE

### Critical Config Values (v3.3)
```python
MICRO_ACCOUNT_MODE = True          # Simulates $10 account
SYMBOLS = ["EURUSD","GBPUSD","USDJPY","XAUUSD","US30","BTCUSD"]
TIMEFRAME = mt5.TIMEFRAME_M5      # Primary
TIMEFRAME_CONFIRM = mt5.TIMEFRAME_M15
TIMEFRAME_TREND = mt5.TIMEFRAME_H1
SEQUENCE_LENGTH = 60               # LSTM lookback candles
ENSEMBLE_ENABLED = True
ENSEMBLE_CONFLICT_THRESHOLD = 0.50
ENSEMBLE_DISAGREEMENT_PENALTY = 0.30
MIN_GLOBAL_SCORE = 0.35            # v3.3: Raised from 0.05 to block weak signals
ADX_RANGING_THRESHOLD = 20         # v3.3: Tightened from 25 to filter more noise
NEAR_MISS_RISK_REDUCTION = 0.25    # v3.3: Added — risk multiplier for LOW confidence signals
MICRO_TP1_ATR_MULT = 3.0           # v3.3: Was 1.5 — RR now 1.5:1
MICRO_TP2_ATR_MULT = 4.0           # v3.3: Was 2.5 — more profit room
MAX_CONCURRENT_TRADES = 3
MAX_DAILY_LOSS_PCT = 5.0
COOLDOWN_AFTER_LOSSES = 3
COOLDOWN_DURATION_MINUTES = 30
NEWS_FILTER_ENABLED = True
SMART_EXIT_ENABLED = True
SMART_EXIT_DANGER_THRESHOLD = 3.0
SESSION_LONDON = (7, 16)           # UTC hours
SESSION_NY = (13, 22)
SESSION_ASIA = (0, 9)
TRADE_SESSION_ASIA = False         # Asia disabled
MAGIC_NUMBER = 121052              # Bot identifier in MT5
ATR_LOOKAHEAD_MULT = 1.2           # Default target threshold, overridden per-symbol in training
```

---

## 10. ML MODELS

### XGBoost (v4.0 — replaced LSTM)
```
XGBClassifier:
  n_estimators=500, max_depth=4, learning_rate=0.02
  subsample=0.8, colsample_bytree=0.8
  min_child_weight=10 (regularization)
  reg_alpha=0.1 (L1), reg_lambda=1.0 (L2)
  scale_pos_weight=auto (class balance)
  eval_metric=logloss

Input: Tabular features + lagged features (lag 1/3/5 + rolling stats + deltas)
Output: predict_proba → P(BUY) ∈ [0,1]
```

**Why XGBoost replaced LSTM:**
- LSTM accuracy was 45-54% on binary = coin flip (data leakage was hiding this)
- XGBoost handles tabular financial data better than sequence models on small datasets
- Faster training (seconds vs minutes) and inference
- Interpretable feature importance
- No TensorFlow dependency needed

- **Scaler**: RobustScaler — fitted on TRAIN data only
- **Feature Selection**: SelectKBest(f_classif, k=50) from all features, fitted on train only
- **Lagged Features**: `engineer_lagged_features()` adds lag 1/3/5 + rolling mean/std(5) + delta 1/3 for 16 key indicators
- **Data**: Per-symbol training (17,280 M5 candles each) with MTF features
- **Split**: 80/20 chronological on raw data BEFORE scaling (leak-free)

### Random Forest
- **Purpose**: De-correlated from XGBoost (sees interaction features, NOT lagged context)
- **Config**: 200 trees, max_depth=10, balanced class weights
- **Scaler**: RobustScaler (fitted on train only)
- **Retraining**: Every 24 hours or 288 candles (currently uses saved model)

### Per-Symbol Model Registry (v4.0)
- Each symbol has its own XGB + RF models: `xgb_model_{SYMBOL}.joblib`, `rf_model_{SYMBOL}.joblib`
- `model_registry.py` auto-loads all models and routes predictions by symbol
- Universal RF fallback trained on all symbols combined (for missing models)
- No TensorFlow dependency — all models are joblib-serialized

### Previous LSTM Performance (for reference, before removal)
| Symbol | LSTM Test | RF Test | Assessment |
|--------|-----------|---------|------------|
| EURUSD | 49.6% | 55.2% | LSTM = random |
| GBPUSD | 54.0% | 57.2% | Best LSTM |
| USDJPY | 45.5% | 53.9% | LSTM below random |
| XAUUSD | 45.5% | 52.1% | LSTM below random |
| US30 | 49.9% | 53.9% | LSTM = random |

**Conclusion**: LSTM was replaced with XGBoost in v4.0. XGBoost expected to deliver 55-60% accuracy with proper lagged features.

---

## 11. KEY FORMULAS & THRESHOLDS

```
trend_strength = clip((ADX - 20) / 30, 0, 1)

# v5.0 XGB-RF Dynamic Weights:
XGB_weight = 0.55 + (trend_strength × 0.10)    # 55-65% (PRIMARY)
RF_weight  = 0.45 - (trend_strength × 0.10)    # 35-45% (COMPLEMENT)

# RF Confidence Gate (pre-filter):
# rf_prob in [0.43, 0.57] → HOLD (noise zone, ~65% of predictions)

weighted_avg = XGB_w × xgb_prob + RF_w × rf_prob
penalty = |xgb_prob - rf_prob| × 0.30
base_score = weighted_avg ± penalty   (pushes toward 0.5)

distance_from_neutral = |base_score - 0.5|
# Reject if distance < 0.002
# Weak zone if distance < 0.01 (0.02 for Asia)

session_bonus ∈ [-0.03, +0.03]
volatility_adj = clip(ATR_zscore × 0.01, -0.02, 0.02)
total_adj = clip(all_adjustments, -0.04, +0.04)
total_adj = min(total_adj, base_score × 0.08)  # ≤8% of base

final_prob = clip(base_score + total_adj, 0, 1)

# v5.0 Thresholds (tighter than v4.2):
# ts > 0.35: BUY > 0.60, SELL < 0.40
# ts > 0.25: BUY > 0.61, SELL < 0.39
# else:      BUY > 0.62, SELL < 0.38

memory_bias = -scale × 0.50  (where scale = normalized similarity to past losses)
# Hard block if similarity > 90%

rank_score = (base_prob + memory_bias + sym_perf_mod + context_boost)
           × max(0.1, trend_strength) × max(1.0, event_strength)

DXY_strength = tanh(slope20 + slope50×0.5 + RSI_norm×1.5)  ∈ [-1, 1]
```

---

## 12. KNOWN ISSUES & EVOLUTION HISTORY

### Evolution Timeline
1. **v1.0**: Single-symbol binary options bot with OTC scraper
2. **v2.0**: MT5 forex, single LSTM, basic SL/TP
3. **v3.0**: Multi-symbol portfolio, Random Forest ensemble, trailing stops, partial closes
4. **v3.1**: Dynamic weighted soft voting, conflict detection, Telegram notifications
5. **v3.2**: BiLSTM(48) architecture, feature selection (106→25), per-symbol training
6. **v3.3**: **Data leakage fix** — scaler fit on train only, session filtering hardened, near-miss logic fixed
7. **v4.0**: **XGBoost replaces LSTM** — LSTM was 45-54% (random). XGBClassifier + lagged features + RF noise gate. ensemble_engine v5.0

### Key Lessons Learned (from conversation history)
- **Near-miss signals were unprofitable**: Every loss came through near-miss activation → risk reduced to 0.25×
- **XAUUSD needs wider SL/TP**: Gold moves $5-15 per candle, default ATR multipliers too tight → per-symbol overrides added
- **Binary regime switching caused flip-flop**: Now requires 2 confirmation candles before regime change
- **Multiplicative scoring was dangerous**: Small adjustments could amplify errors → switched to additive with 8% cap
- **FOK vs IOC filling**: Some brokers reject IOC → fallback to FOK on error code 10013
- **Data leakage inflated accuracy**: Scaler was fitted on ALL data (train+test) before split. After fix, LSTM accuracy dropped from ~52% to ~49% average — revealing the model was never truly learning
- **Session filtering used stale tick time**: `tick.time` from MT5 can be hours old during low liquidity. Fixed to use `datetime.utcnow()` as sole time source
- **LSTM is effectively dead weight**: Post leak-fix, predictions cluster in 0.45-0.55 noise zone. RF carries the ensemble alone

### v3.3 Fixes Applied (May 2026 Session)

**1. Data Leakage Fix (`lstm_model.py`)**
- **Bug**: `scaler.fit(df[ALL].values)` fitted on entire dataset before train/test split
- **Fix**: Split raw data first → `scaler.fit(features[:split_raw])` on train only
- **Impact**: LSTM test accuracy dropped (expected — was artificially inflated)

**2. Session Filtering Fix (`trade_manager.py`)**
- **Bug**: `is_in_trading_session()` and `get_active_session()` used `tick.time` which can be stale
- **Fix**: Both functions now use `datetime.datetime.utcnow()` as sole time source
- **Impact**: Asia session leakage eliminated

**3. Near-Miss Risk Logic (`main.py`)**
- **Bug**: Checked `decision_reason == "NEAR_MISS_ACTIVATION"` — a deprecated string never set by ensemble v4.2
- **Fix**: Now checks `confidence_level == "LOW"`, applies `Config.NEAR_MISS_RISK_REDUCTION` (0.25)

**4. LSTM Feature Selection Mismatch (`model_registry.py`)**
- **Bug**: Passed 106 features to model expecting 25 → `ValueError`
- **Fix**: Applied `scaler.selected_indices_` during inference to select the correct 25 features

### Active Concerns
- **LSTM is non-functional**: 45-54% accuracy on binary = coin flip. Consider replacing with gradient boosting or removing entirely
- **Ensemble thresholds too high**: BUY>0.65 almost never triggers with LSTM~0.50 and RF~0.55. Bot mostly HOLDs
- **M5 data window too short**: 17,280 candles = ~12 trading days. Non-stationary patterns don't generalize
- MICRO_ACCOUNT_MODE should be set to False when balance grows past $100
- Asia session trading is DISABLED (low liquidity)
- Telegram is DISABLED (set TELEGRAM_ENABLED=True after setup)
- BTCUSD training fails — symbol not available in MT5 broker

---

## 13. HOW TO RUN

### Prerequisites
1. MetaTrader 5 installed and open with "Allow Algorithmic Trading" checked
2. Python 3.8+ with venv
3. Per-symbol trained models exist (generated by `train_offline.py`):
   - `lstm_model_{SYMBOL}.h5` + `lstm_scaler_{SYMBOL}.joblib` per symbol
   - `rf_model_{SYMBOL}.joblib` + `rf_scaler_{SYMBOL}.joblib` + `rf_features_{SYMBOL}.joblib` per symbol
   - `rf_model_universal.joblib` (fallback)

### First-Time Setup
```powershell
cd D:\Candlestick_Detection
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Train models (requires MT5 open):
python train_offline.py

# Optional: Setup Telegram
python setup_telegram.py
```

### Daily Run
```powershell
# Option 1: Direct
python main.py

# Option 2: Auto-restart wrapper (recommended for production)
powershell -ExecutionPolicy Bypass -File .\run_bot.ps1
```

### Graceful Stop
Press `Ctrl+C` — saves state to `active_trades.json`, MT5 continues managing open positions.

---

## 14. DEPENDENCIES

```
MetaTrader5>=5.0.45      # MT5 Python API
xgboost>=1.7.0            # XGBoost model (replaced TensorFlow LSTM)
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0       # Random Forest + scalers + SelectKBest
ta>=0.10.2                # Technical indicators library
matplotlib>=3.7.0         # Training curve plots
requests>=2.31.0          # News scraper + Telegram
beautifulsoup4>=4.12.0    # ForexFactory HTML parsing
joblib>=1.3.0             # Model serialization
python-dotenv>=1.0.0      # .env file loading
scipy                     # argrelextrema for pattern detection
# tensorflow              # REMOVED in v4.0 — no longer needed
```

---

## APPENDIX: INTER-MODULE IMPORT MAP

```
main.py imports:
  ├── config.Config
  ├── mt5_engine.{connect_to_exness, heartbeat, execute_forex_trade, check_kill_switch, ...}
  ├── trade_manager.TradeManager
  ├── notifier.get_notifier
  ├── data_loader.{fetch_mtf_data, fetch_tick_data, is_market_open}
  ├── features.feature_engineering_pipeline
  ├── model_registry.ModelRegistry  ← v4.0 (loads per-symbol XGB + RF)
  ├── rf_model.RFModel              (kept for feature importance summary)
  ├── ensemble_engine.ensemble_predict  (v5.0 — xgb_prob first param)
  ├── news_filter.is_news_window
  └── macro_context.get_dxy_strength

model_registry.py imports:
  ├── xgb_model.engineer_lagged_features  ← v4.0
  └── rf_model.engineer_rf_features

xgb_model.py imports:                    ← v4.0 NEW
  ├── xgboost.XGBClassifier
  ├── sklearn.preprocessing.RobustScaler
  └── sklearn.feature_selection.SelectKBest

features.py imports:
  ├── candles.add_candlestick_patterns
  ├── pattern_detector.add_chart_patterns
  └── divergence.add_divergence_features

trade_manager.py imports:
  └── smart_exit.{evaluate_smart_exit, should_tighten_sl, get_tighten_atr_mult}

ensemble_engine.py imports:             (v5.0 — XGB Edition)
  └── config.Config (only)

macro_context.py imports:
  └── MetaTrader5 (direct MT5 data fetch for DXY)
```

---

## 15. MULTI-SYMBOL TRAINING (✅ COMPLETE — XGBoost v4.0)

### Status: Fully Operational
All per-symbol models use XGBoost + RF. `main.py` uses `ModelRegistry` for routing.

### Per-Symbol Config (v5.3 — Final SHAP + RFE Results)
| Symbol | XGB WFV | RF WFV | Features | Stability | SHAP #1 Feature |
|--------|---------|--------|----------|-----------|-----------------|
| EURUSD | **61.4%** | 57.3% | 40 | 97.6% | M15_RSI |
| GBPUSD | **62.6%** | 57.5% | 40 | 98.3% | M15_RSI |
| USDJPY | **60.4%** | 57.8% | 40 | 97.3% | M15_RSI |
| XAUUSD | **59.8%** | 56.4% | 40 | 98.3% | M15_RSI |
| US30   | **60.3%** | 58.3% | 40 | 95.9% | M15_RSI |
| BTCUSD | — | — | — | — | Not in broker |

**Average XGB WFV: 60.9% | RF WFV: 57.5% | Features: 80 → 40** — Training time: 11.5 min

**Key insights:**
- `M15_RSI` is the #1 SHAP feature for ALL symbols — M15 timeframe context dominates
- `divergence_score` is consistently top-5 across all symbols
- `EMA_50_delta1` (momentum) is critical for GBPUSD and US30
- Balanced SHAP sampling (1000 BUY + 1000 SELL) prevents directional bias
- XGB shows BUY-heavy distribution (market was in uptrend during training period) — this is correct behavior, RF balances it in ensemble via conflict detection

### Model Files (per symbol) — v4.0
```
xgb_model_{SYMBOL}.joblib    + xgb_scaler_{SYMBOL}.joblib + xgb_features_{SYMBOL}.joblib
rf_model_{SYMBOL}.joblib      + rf_scaler_{SYMBOL}.joblib + rf_features_{SYMBOL}.joblib
rf_model_universal.joblib     (fallback for missing symbols)
```

### Integration (v4.0)
```python
from model_registry import ModelRegistry
registry = ModelRegistry()  # auto-loads all per-symbol models
xgb_prob = registry.predict_xgb("XAUUSD", df_processed)  # with lagged features + selection
rf_prob  = registry.predict_rf("XAUUSD", df_processed)    # with column alignment
```

### Data Pipeline (v4.0 — Leak-Free + Tabular)
```
1. Feature engineering pipeline (100+ features)
2. engineer_lagged_features() adds lag 1/3/5 + rolling stats + deltas
3. Drop rows without Target / NaN cleanup
4. Split raw data 80/20 chronologically
5. SelectKBest(k=50) on TRAIN portion ONLY
6. Fit RobustScaler on TRAIN selected features
7. Transform all data using train statistics
8. Train XGBClassifier(500 trees, depth=4, lr=0.02) with early stopping
```

---

## 16. ACCURACY IMPROVEMENT ROADMAP

| Step | Task | Status | Expected Gain |
|------|------|--------|---------------|
| **1** | Increase training data: 17K → 99K candles per symbol | ✅ Done | +3-5% accuracy |
| **2** | Walk-forward validation: rolling window retraining | ✅ Done | +2-3% stability |
| **3** | SHAP feature importance + RFE cleanup noisy features | ✅ Done | +1-2% accuracy |
| **4** | Regime-specific thresholds (Trending/Ranging/Volatile) | ⏳ Next | +2-3% win rate |
| **5** | Add LightGBM to ensemble (XGB + RF + LGB) | ⏳ Pending | +1-2% accuracy |

---

> **END OF HANDOVER (Updated: May 6, 2026)**
>
> **v5.3 Step 3 — SHAP + RFE + Balanced Target (May 7, 2026)**:
>
> **Root cause of BUY-heavy distribution:**
> The old ATR-based target (`BUY if future_move > ATR*1.2`) generates all-BUY labels during uptrend markets.
>
> **Fix: Percentile-based target generation (`features.py generate_target_column`)**:
> - Old: `BUY if move > ATR*1.2` → all BUY in uptrend
> - New: `BUY if move in top 33% of rolling 2000-candle window` → always balanced
> - Result: BUY:SELL ratio = ~1:1 regardless of market regime
> - Note: Initial version used 500-candle window which caused accuracy drop (thresholds shifted too fast). Fixed to 2000-candle window for stable thresholds.
>
> **Accuracy after fix (latest training May 7 14:31):**
> | Symbol | XGB WFV | Labels Balance |
> |--------|---------|----------------|
> | EURUSD | 54.8% | BUY=30K SELL=30K ✅ |
> | GBPUSD | 55.2% | BUY=31K SELL=31K ✅ |
> | USDJPY | 60.4% | BUY=32K SELL=32K ✅ |
> | XAUUSD | 59.6% | BUY=31K SELL=30K ✅ |
> | US30   | 58.4% | BUY=29K SELL=29K ✅ |
>
> EURUSD/GBPUSD accuracy dropped vs old biased target — this is expected and correct. The old 61-63% was inflated by the model learning "always BUY". The new 54-55% is honest accuracy on a balanced task. USDJPY/XAUUSD/US30 maintained accuracy because they have more complex patterns beyond simple trend-following.
>
> **Other v5.3 changes:**
> - `xgb_model.py`: SHAP balanced sampling (1000 BUY + 1000 SELL), RFE 80→40 features
> - `calibrate_model()` function added (available but not used — percentile target is the real fix)
>
> **v5.2 Step 2 — Walk-Forward Validation (May 6, 2026)**:
> - xgb_model.py: 5-fold rolling WFV, stability score, weekly auto-retrain in main.py
>
> **Walk-Forward vs Static Split:**
> ```
> Static:      [====Train 80%====][=Test 20%=]  ← one snapshot
> Walk-Forward: fold1: [Train60%][Test10%]
>               fold2:   [Train60%][Test10%]     ← rolling
>               fold3:     [Train60%][Test10%]
>               fold4:       [Train60%][Test10%]
>               fold5:         [Train60%][Test10%]
>               Final:   [====Train 80%====]     ← most recent data
> ```
> **Stability score** = 1 - (std/mean) × 100. Score > 90% = consistent across market regimes.
>
> **v5.1 Step 1 — Large Dataset Training (May 6, 2026)**:
> - train_offline.py: 17K → 99K candles, XGB 1000 trees + early stopping, RF 500 trees
> - **features.py**: Fixed `active_bullish_ob`, `active_bearish_ob` being dropped before main.py reads them for state tracking. Fixed `last_fvg_price` being dropped before main.py reads it.
> - **model_registry.py**: Added `has_model()` method to check if XGB model exists for a symbol
> - **main.py**: Added `registry.has_model(symbol)` check to skip symbols without XGB models (e.g., BTCUSD)
> - **config.py**: Updated PROB_THRESHOLD_BUY (0.70→0.56), renamed LSTM ensemble weights to XGB, added legacy aliases
> - **ensemble_decisions.csv**: Rotated (old file had `lstm_prob` column header, new file uses `xgb_prob`)
>
> **Root cause of no-trade issue**: RF_NOISE_GATE was too narrow (0.45-0.55) blocking ~65% of signals. Combined with threshold too high (0.60-0.62) vs actual XGB output range (0.55-0.62), the bot was HOLDing almost everything. After calibration from 2394 historical decisions, thresholds now match actual model output distribution.
>
> **v5.1 second pass (May 5, 2026 — post first live run)**:
> - **ensemble_engine.py**: trend_strength formula changed from `(ADX-20)/30` to `(ADX-15)/35` for smoother gradient at low ADX values. ADX=19.7 now gives ts=0.134 instead of 0.000.
> - **main.py apply_hybrid_filters**: ADX ranging filter now uses `H1_ADX` (same as ensemble) instead of noisy M5 ADX. This fixed EURUSD SELL being rejected by M5 ADX=14.2 while H1_ADX=15.9 was valid.
> - **config.py ADX_RANGING_THRESHOLD**: Lowered from 20 to 15. H1 ADX 15-20 is weak trend, not pure ranging.
>
> **v5.1 third pass (May 5, 2026 — bug fixes from Claude analysis)**:
> - **data_loader.py**: Fixed `Config.FOREX_SYMBOL` AttributeError — all 5 functions now use `getattr(Config, 'FOREX_SYMBOL', None) or Config.SYMBOLS[0]` safe resolution. This was a latent crash bug if any function was called without explicit symbol.
> - **features.py generate_target_column**: Added `symbol` parameter. Now uses `Config.ATR_LOOKAHEAD_MULT_PER_SYMBOL` dict to match training targets exactly (XAUUSD=1.5, US30=1.5, BTCUSD=1.8, forex=1.2). Previously live bot used 1.2 for all symbols while models were trained with different multipliers — causing label mismatch.
> - **features.py feature_engineering_pipeline**: Added `symbol` parameter, passes it to `generate_target_column`.
> - **main.py**: Updated `feature_engineering_pipeline` call to pass `symbol=symbol`.
> - **train_offline.py**: Updated `feature_engineering_pipeline` call to pass `symbol=symbol` (was already using Config override but now consistent).
> - **config.py ATR_LOOKAHEAD_MULT_PER_SYMBOL**: New dict with per-symbol values matching train_offline.py exactly.
> - **config.py SMART_EXIT**: Raised `SMART_EXIT_DANGER_THRESHOLD` from 3.0 to 4.0, `SMART_EXIT_TIGHTEN_THRESHOLD` from 2.0 to 2.5, `SMART_EXIT_MIN_CANDLES_OPEN` from 3 to 5. Previous values caused trades to close early at small profit (0.02) before reaching TP.
>
> **Status of all reported issues**:
> | Issue | Status | Fix |
> |-------|--------|-----|
> | Penalty always subtracts even when models agree | ✅ Already correct | Step 6 in ensemble_engine subtracts if weighted_avg>0.5, adds if <0.5 |
> | Config.FOREX_SYMBOL AttributeError | ✅ Fixed | data_loader.py uses safe getattr resolution |
> | ATR_LOOKAHEAD_MULT mismatch live vs training | ✅ Fixed | Per-symbol dict in config, passed through pipeline |
> | Universal model trained without MTF features | ✅ Not a real issue | Universal is RF-only fallback, BTCUSD now skipped via has_model() |
> | Smart Exit closing trades too early | ✅ Fixed | Thresholds raised: 3.0→4.0 danger, 3→5 min candles |

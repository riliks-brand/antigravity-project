# 🤖 ELITE TRADING BOT v3.1 — COMPLETE HANDOVER DOCUMENT

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
| **Name** | Elite MT5 Trading Bot v3.1 (Ensemble Edition) |
| **Language** | Python 3.10+ |
| **Platform** | Windows (MetaTrader 5 native) |
| **Broker** | MetaQuotes-Demo (Account: 5049001425) |
| **Directory** | `D:\Candlestick_Detection` |
| **Entry Point** | `main.py` |
| **Account Mode** | MICRO ($10 simulated start, 0.01 lot forced) |

**What it does**: Automated multi-symbol forex/commodity/index trading bot that uses an LSTM + Random Forest ensemble to generate BUY/SELL signals, executes them on MT5, and manages positions with trailing stops, partial closes, and AI-driven smart exits.

**Symbols Traded**: EURUSD, GBPUSD, USDJPY, XAUUSD, US30, BTCUSD

---

## 2. ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────┐
│                        MAIN LOOP (main.py)                       │
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

**Target Generation**: BUY(1) if future_move > 1.2×ATR, SELL(0) if < -1.2×ATR, HOLD(NaN) filtered out

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
- **MICRO overrides**: SL = 2.0×ATR, TP1 = 1.5×ATR, TP2 = 2.5×ATR
- **XAUUSD overrides**: SL = 3.0×ATR, TP1 = 3.0×ATR, TP2 = 4.5×ATR
- **US30 overrides**: SL = 2.5×ATR, TP1 = 2.5×ATR, TP2 = 3.5×ATR

### Adaptive Risk (`get_adaptive_risk`):
- Strong trend + HIGH confidence → 2.0% (micro) / 1.0% (standard)
- Otherwise → 1.0% (micro) / 0.5% (standard)
- Regime just changed → ×0.7
- DXY contradicts → ×0.5
- Near-miss signal → ×0.5
- Drawdown > 2% → ×0.5 (survival mode)
- Equity below MA(20) → ×0.5

### Kill Switches
- Daily loss > 5% → close ALL positions, pause until next day
- Portfolio risk cap: total open risk cannot exceed 3%

---

## 9. CONFIGURATION REFERENCE

### Critical Config Values
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
MIN_GLOBAL_SCORE = 0.05           # Relaxed to allow trades
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
```

---

## 10. ML MODELS

### LSTM Architecture
```
Input: (batch, 60, num_features)  — 60 M5 candles of raw features
  → LSTM(128, return_sequences=True)
  → Dropout(0.3) → BatchNorm
  → LSTM(64)
  → Dropout(0.2) → BatchNorm
  → Dense(32, relu)
  → Dropout(0.1)
  → Dense(1, sigmoid)    — Output: probability ∈ [0,1]
Output: P(bullish) — values > 0.5 indicate BUY tendency
```

- **Scaler**: RobustScaler (outlier-immune)
- **Training**: Binary crossentropy, Adam(lr=0.001), EarlyStopping(patience=5)
- **Sample weighting**: 1.5× penalty for samples resembling past losses
- **Data**: Trained on EURUSD M5 (50K candles) with MTF features

### Random Forest
- **Purpose**: De-correlated from LSTM (sees interaction features, NOT raw sequences)
- **Config**: 200 trees, max_depth=10, balanced class weights
- **Scaler**: RobustScaler
- **Retraining**: Every 24 hours or 288 candles (currently uses saved model)

---

## 11. KEY FORMULAS & THRESHOLDS

```
trend_strength = clip((ADX - 20) / 30, 0, 1)

LSTM_weight = 0.5 + (trend_strength × 0.3)    # 50-80%
RF_weight   = 0.5 - (trend_strength × 0.3)    # 20-50%

weighted_avg = LSTM_w × lstm_prob + RF_w × rf_prob
penalty = |lstm_prob - rf_prob| × 0.30
base_score = weighted_avg ± penalty   (pushes toward 0.5)

distance_from_neutral = |base_score - 0.5|
# Reject if distance < 0.002
# Weak zone if distance < 0.01 (0.02 for Asia)

session_bonus ∈ [-0.03, +0.03]
volatility_adj = clip(ATR_zscore × 0.01, -0.02, 0.02)
total_adj = clip(all_adjustments, -0.04, +0.04)
total_adj = min(total_adj, base_score × 0.08)  # ≤8% of base

final_prob = clip(base_score + total_adj, 0, 1)

# Thresholds (trend-adaptive):
# ts > 0.35: BUY > 0.55, SELL < 0.45
# ts > 0.25: BUY > 0.56, SELL < 0.44
# else:      BUY > 0.58, SELL < 0.42

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
5. **v4.0**: Session-aware strategy, regime persistence, adaptive risk, event-driven evaluation
6. **v4.1**: Full diagnostics (side, stage, distance), symmetric scoring, additive model

### Key Lessons Learned (from conversation history)
- **Near-miss signals were unprofitable**: Every loss came through near-miss activation → DISABLED
- **XAUUSD needs wider SL/TP**: Gold moves $5-15 per candle, default ATR multipliers too tight → per-symbol overrides added
- **Binary regime switching caused flip-flop**: Now requires 2 confirmation candles before regime change
- **Multiplicative scoring was dangerous**: Small adjustments could amplify errors → switched to additive with 8% cap
- **FOK vs IOC filling**: Some brokers reject IOC → fallback to FOK on error code 10013

### Active Concerns
- MICRO_ACCOUNT_MODE should be set to False when balance grows past $100
- Asia session trading is DISABLED (low liquidity)
- Telegram is DISABLED (set TELEGRAM_ENABLED=True after setup)
- `DIAGNOSTIC_MODE = False` — when True, uses relaxed thresholds (for testing only)

---

## 13. HOW TO RUN

### Prerequisites
1. MetaTrader 5 installed and open with "Allow Algorithmic Trading" checked
2. Python 3.10+ with venv
3. Trained models exist: `lstm_model.h5`, `lstm_scaler.joblib`, `rf_model.joblib`

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
tensorflow>=2.13.0        # LSTM model
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0       # Random Forest + scalers
ta>=0.10.2                # Technical indicators library
matplotlib>=3.7.0         # Training curve plots
requests>=2.31.0          # News scraper + Telegram
beautifulsoup4>=4.12.0    # ForexFactory HTML parsing
joblib>=1.3.0             # Model serialization
python-dotenv>=1.0.0      # .env file loading
scipy                     # argrelextrema for pattern detection (imported but not in requirements.txt!)
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
  ├── rf_model.RFModel
  ├── ensemble_engine.ensemble_predict
  ├── news_filter.is_news_window
  └── macro_context.get_dxy_strength

features.py imports:
  ├── candles.add_candlestick_patterns
  ├── pattern_detector.add_chart_patterns
  └── divergence.add_divergence_features

trade_manager.py imports:
  └── smart_exit.{evaluate_smart_exit, should_tighten_sl, get_tighten_atr_mult}

ensemble_engine.py imports:
  └── config.Config (only)

macro_context.py imports:
  └── MetaTrader5 (direct MT5 data fetch for DXY)
```

---

## 15. 🚨 IN-PROGRESS MIGRATION: Multi-Symbol Training

### The Problem (Critical)
The LSTM and RF models are trained on **EURUSD only** (50K candles), but the bot trades **6 symbols** including XAUUSD, BTCUSD, and US30 — assets with completely different behavior. This means the models are effectively "blind" to 5 of 6 symbols.

### What Was Done (by Claude in a previous session)
`train_offline.py` was **completely rewritten** (127 lines → 670 lines) to support per-symbol training:

**1. `SYMBOL_CONFIGS` dictionary** — per-symbol training parameters:
| Symbol | M5 Candles | ATR Lookahead Mult | Reason |
|--------|-----------|-------------------|--------|
| EURUSD | 50,000 | 1.2 | Baseline, stable |
| GBPUSD | 50,000 | 1.2 | Slightly more volatile |
| USDJPY | 50,000 | 1.2 | Range-bound tendency |
| XAUUSD | 40,000 | 1.5 | High volatility ($5-15/candle) |
| US30 | 40,000 | 1.5 | News-driven index |
| BTCUSD | 40,000 | 1.8 | Extreme crypto noise |

**2. Per-symbol model files** (naming convention):
```
lstm_model_{SYMBOL}.h5      + lstm_scaler_{SYMBOL}.joblib
rf_model_{SYMBOL}.joblib     + rf_scaler_{SYMBOL}.joblib + rf_features_{SYMBOL}.joblib
```

**3. Universal fallback model** — trained on ALL symbols combined (with price normalized to % change + `symbol_id` feature). Used when a specific symbol's model is missing.

**4. `model_registry.py`** — auto-generated by `train_offline.py` when training completes. Provides:
```python
from model_registry import ModelRegistry
registry = ModelRegistry()
lstm_prob = registry.predict_lstm("XAUUSD", df_processed)  # auto-fallback
rf_prob   = registry.predict_rf("XAUUSD", df_processed)    # auto-fallback
```

### ⚠️ What Is NOT Done Yet (Action Required)

> [!CAUTION]
> The following 3 integration steps have NOT been completed. The bot currently still runs on the old single-model system.

**Step 1: `config.py` — Missing `ATR_LOOKAHEAD_MULT`**
`train_offline.py` references `Config.ATR_LOOKAHEAD_MULT` but it doesn't exist in `config.py`. Must add:
```python
ATR_LOOKAHEAD_MULT = 1.2  # Default, overridden per-symbol during training
```

**Step 2: Run `python train_offline.py`**
This generates:
- 6 × LSTM models + scalers
- 6 × RF models + scalers + feature lists
- 1 × universal RF fallback
- `model_registry.py` file

**Step 3: Update `main.py` to use `ModelRegistry`**
Currently `main.py` line 224 still loads: `lstm_model_path = "lstm_model.h5"` (single model).
Must replace with:
```python
from model_registry import ModelRegistry
registry = ModelRegistry()

# In evaluation loop, change:
# OLD: lstm_prob = lstm_model.predict(seq)
# NEW: lstm_prob = registry.predict_lstm(symbol, df_processed)
# OLD: rf_prob = rf_engine.predict_proba(df)
# NEW: rf_prob = registry.predict_rf(symbol, df_processed)
```

### Current State Summary
| Component | Status |
|-----------|--------|
| `train_offline.py` rewrite | ✅ Complete (670 lines) |
| `SYMBOL_CONFIGS` | ✅ Defined in train_offline.py |
| `model_registry.py` code | ✅ Written (embedded in train_offline.py, auto-saved on run) |
| `config.py` ATR_LOOKAHEAD_MULT | ❌ **Not added** |
| Training execution | ❌ **Not run yet** (no per-symbol model files exist) |
| `main.py` integration | ❌ **Not updated** (still uses single EURUSD model) |

---

> **END OF HANDOVER** — This document covers 100% of the codebase. Any AI reading this should be able to modify, debug, or extend any part of the system.

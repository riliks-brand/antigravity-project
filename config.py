# config.py
# Professional MT5 Trading Bot — Elite Configuration v3.1
# Ensemble Edition (LSTM + Random Forest + Dynamic Voting)

import MetaTrader5 as mt5
import os

class Config:
    # =========================================
    # MetaTrader 5 Credentials & Connection
    # =========================================
    LOGIN = 5049001425
    PASSWORD = "_sTcEx2i"
    SERVER = "MetaQuotes-Demo"
    MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

    # =========================================
    # Trading Symbol & Timeframe
    # =========================================
    FOREX_SYMBOL = "EURUSD"
    TIMEFRAME = mt5.TIMEFRAME_M5        # Primary execution timeframe
    TIMEFRAME_CONFIRM = mt5.TIMEFRAME_M15  # Confirmation timeframe
    TIMEFRAME_TREND = mt5.TIMEFRAME_H1     # Trend timeframe
    DATA_POINTS = 2000                   # Candles to fetch per timeframe

    # =========================================
    # LSTM Model Settings
    # =========================================
    SEQUENCE_LENGTH = 120
    PREDICT_LOOKAHEAD = 5

    # Decision Thresholds (Adaptive base values — adjusted by volatility at runtime)
    PROB_THRESHOLD_BUY = 0.70
    PROB_THRESHOLD_SELL = 0.30
    ADAPTIVE_THRESHOLD_ENABLED = True    # If True, thresholds shift with volatility

    # =========================================
    # Ensemble Engine (LSTM + Random Forest)
    # =========================================
    ENSEMBLE_ENABLED = True

    # Dynamic Weighting (shifts based on ADX / market state)
    ENSEMBLE_LSTM_WEIGHT_TRENDING = 0.70   # LSTM weight when ADX > threshold (trending)
    ENSEMBLE_RF_WEIGHT_TRENDING = 0.30     # RF weight when trending
    ENSEMBLE_LSTM_WEIGHT_RANGING = 0.50    # LSTM weight when ADX < threshold (ranging)
    ENSEMBLE_RF_WEIGHT_RANGING = 0.50      # RF weight when ranging

    # Conflict Handling
    ENSEMBLE_CONFLICT_THRESHOLD = 0.50     # If |LSTM - RF| > this → SKIP trade
    ENSEMBLE_DISAGREEMENT_PENALTY = 0.30   # Penalty factor: final -= |LSTM-RF| * penalty

    # RF Retraining Schedule
    RF_RETRAIN_EVERY_HOURS = 24            # Retrain RF every N hours
    RF_RETRAIN_EVERY_CANDLES = 288         # OR every N candles evaluated (288 = 24h of M5)
    RF_N_ESTIMATORS = 200                  # Number of trees
    RF_MAX_DEPTH = 10                      # Max tree depth (prevent overfitting)

    # Ensemble Logging
    ENSEMBLE_LOG_FILE = "ensemble_decisions.csv"

    # =========================================
    # Risk Management
    # =========================================
    RISK_PERCENT_PER_TRADE = 1.0         # Default % of account balance risked per trade
    MAX_DAILY_LOSS_PCT = 5.0             # Kill switch: stop trading if daily loss > X%
    MAX_CONCURRENT_TRADES = 3            # Max open positions at any time

    # Confidence Weighting (Dynamic Risk based on Signal Strength)
    CONFIDENCE_WEIGHTING_ENABLED = True
    CONFIDENCE_STRONG_BUY = 0.85         # Above this = strong signal
    CONFIDENCE_STRONG_SELL = 0.15        # Below this = strong signal
    CONFIDENCE_STRONG_MULTIPLIER = 2.0   # Multiply RISK_PERCENT_PER_TRADE by this

    # Cooldown: pause after N consecutive losses
    COOLDOWN_AFTER_LOSSES = 3
    COOLDOWN_DURATION_MINUTES = 30       # How long to pause

    # Equity Curve Protection
    EQUITY_MA_PERIOD = 20                # Moving average window for equity curve
    EQUITY_RISK_REDUCTION = 0.5          # Reduce risk to 50% if equity < MA

    # =========================================
    # Execution Safety
    # =========================================
    MAX_SPREAD_POINTS = 30               # Reject trade if spread > this
    SLIPPAGE_TOLERANCE = 10              # MT5 deviation parameter (points)
    MAX_RETRIES = 2                      # Retry order_send on failure
    MAGIC_NUMBER = 121052                # Unique bot identifier

    # =========================================
    # Trade Management (Trailing / Partial)
    # =========================================
    # ATR Multipliers for SL/TP
    SL_ATR_MULT = 1.5                    # Stop Loss = ATR * 1.5
    TP1_ATR_MULT = 2.0                   # Take Profit 1 = ATR * 2.0
    TP2_ATR_MULT = 3.0                   # Take Profit 2 (trailing target) = ATR * 3.0

    # Partial Close
    PARTIAL_CLOSE_PCT = 0.5              # Close 50% at TP1
    MOVE_SL_TO_BE_AFTER_TP1 = True       # Move SL to breakeven after TP1 hit

    # Trailing Stop
    TRAILING_STOP_ATR_MULT = 1.0         # Trail distance = ATR * 1.0
    TRAILING_ACTIVATE_ATR_MULT = 1.5     # Activate trailing after price moves 1.5 * ATR in profit

    # =========================================
    # Signal Deduplication
    # =========================================
    MIN_CANDLES_BETWEEN_TRADES = 3       # No same-direction trade within N candles

    # =========================================
    # Session Times (UTC hours)
    # =========================================
    # NOTE: We use mt5.symbol_info_tick().time for server time, not local clock
    SESSION_LONDON = (7, 16)
    SESSION_NY = (13, 22)
    SESSION_ASIA = (0, 9)
    TRADE_ONLY_IN_SESSIONS = True        # If True, only trade during active sessions

    # =========================================
    # News Filter
    # =========================================
    NEWS_FILTER_ENABLED = True
    NEWS_BLOCK_MINUTES_BEFORE = 15       # Block trading X min before high-impact news
    NEWS_BLOCK_MINUTES_AFTER = 15        # Block trading X min after high-impact news

    # =========================================
    # Feature Engineering
    # =========================================
    ATR_THRESHOLD = 0.0002               # Filter for low liquidity periods
    ADX_RANGING_THRESHOLD = 25           # ADX < 25 = ranging market → skip
    DXY_TICKER = "DX-Y.NYB"

    # Feature Drift Detection
    DRIFT_DETECTION_ENABLED = True
    DRIFT_WINDOW = 100                   # Rolling window for mean/std monitoring
    DRIFT_THRESHOLD_ZSCORE = 3.0         # Alert if feature Z-score > 3

    # =========================================
    # Memory Similarity (Probability Modifier)
    # =========================================
    # Memory now MODIFIES probability instead of hard-blocking
    MEMORY_BIAS_SCALE = 0.10             # Max probability adjustment (±10%)
    MEMORY_SIMILARITY_THRESHOLD = 60     # Only apply bias if similarity > 60%

    # =========================================
    # Telegram Notifications
    # =========================================
    TELEGRAM_ENABLED = True              # Set to True after running setup_telegram.py
    TELEGRAM_ENV_FILE = ".env"           # Credentials file (generated by setup script)

    # Spam Control
    TELEGRAM_MAX_MESSAGES_PER_MINUTE = 5
    TELEGRAM_DAILY_SUMMARY_HOUR = 23    # Hour (UTC) to send daily summary
    TELEGRAM_ALERT_LEVELS = ["TRADE", "CLOSE", "EMERGENCY", "DAILY"]

    # =========================================
    # Logging & Persistence
    # =========================================
    ACTIVE_TRADES_FILE = "active_trades.json"
    TRADING_HISTORY_FILE = "trading_history.csv"
    EXECUTION_QUALITY_LOG = "execution_quality.log"
    REJECTED_TRADES_LOG = "rejected_trades.log"
    LOG_FILE = "bot.log"

    # =========================================
    # Deployment
    # =========================================
    HEARTBEAT_INTERVAL_SECONDS = 30      # Check MT5 connection every N seconds

    # Legacy (kept for backward compatibility)
    TRADING_MODE = "FOREX"
    SYMBOL = "BTCUSD"
    FOREX_RISK_PER_TRADE = 10.0
    OTC_CANDLE_INTERVAL = 60
    OTC_CDP_PORT = 9225

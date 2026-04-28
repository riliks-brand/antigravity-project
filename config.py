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
    # Portfolio Symbols & Timeframes
    # =========================================
    SYMBOLS = [
        "EURUSD", 
        "GBPUSD", 
        "USDJPY", 
        "XAUUSD", 
        "US30", 
        "BTCUSD"
    ]
    TRADE_CRYPTO_WEEKENDS = False          # Skip crypto analysis on weekends
    
    TIMEFRAME = mt5.TIMEFRAME_M5        # Primary execution timeframe
    TIMEFRAME_CONFIRM = mt5.TIMEFRAME_M15  # Confirmation timeframe
    TIMEFRAME_TREND = mt5.TIMEFRAME_H1     # Trend timeframe
    DATA_POINTS = 2000                   # Candles to fetch per timeframe

    # =========================================
    # LSTM Model Settings
    # =========================================
    SEQUENCE_LENGTH = 60  # Reduced for diagnostic phase
    PREDICT_LOOKAHEAD = 5
    
    # Phase 1 Diagnostic Mode
    DIAGNOSTIC_MODE = False

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
    # Portfolio Smart Execution & Context Ranking
    # =========================================
    MIN_GLOBAL_SCORE = 0.05              # Relaxed from 0.60 to allow trades
    
    # Priority Context Boosts ( added to raw prediction )
    BOOST_STRONG_TREND = 0.02
    BOOST_HIGH_VOLATILITY = 0.01

    # Diversification limits
    MAX_USD_EXPOSURE = 2                 # Max concurrent open trades involving 'USD'
    
    # Anti-Spam / Correlation 
    CORRELATION_RULES = [
        {"pairs": ["EURUSD", "GBPUSD"], "type": "DIRECT"},  # Prevent same direction
        {"pairs": ["USDJPY", "US30"], "type": "DIRECT"},    # Prevent same direction
        {"pairs": ["XAUUSD", "USDJPY"], "type": "INVERSE"}  # Prevent buy XAU/buy USDJPY
    ]

    # =========================================
    # Risk Management
    # =========================================
    MAX_GLOBAL_RISK_PCT = 3.0            # Max active risk across entire portfolio
    DRAWDOWN_SURVIVAL_THRESHOLD = 2.0    # If overall Drawdown > this %, trigger survival mode
    SURVIVAL_RISK_MODIFIER = 0.5         # Reduce new trade risk by 50% in survival mode
    
    # Smart execution risk tiers
    RISK_TIER_STRONG = 1.0               # Final score >= 0.70
    RISK_TIER_WEAK = 0.5                 # Final score >= 0.60 and < 0.70
    
    MAX_DAILY_LOSS_PCT = 5.0             # Kill switch: stop ALL trading if daily loss > X%
    MAX_CONCURRENT_TRADES = 3            # Overall portfolio max open positions

    # Cooldown: pause after N consecutive losses
    COOLDOWN_AFTER_LOSSES = 3
    COOLDOWN_DURATION_MINUTES = 30       # How long to pause

    # Equity Curve Protection
    EQUITY_MA_PERIOD = 20                # Moving average window for equity curve
    EQUITY_RISK_REDUCTION = 0.5          # Reduce risk to 50% if equity < MA

    # =========================================
    # Execution Safety
    # =========================================
    MAX_SPREAD_POINTS = 500              # Relaxed from 30 to allow Gold/Indices
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
    
    # Session-Specific Trading Toggles
    TRADE_ONLY_IN_SESSIONS = True        # If True, enforces session filtering
    TRADE_SESSION_LONDON = True          # Enable trading during London session
    TRADE_SESSION_NY = True              # Enable trading during New York session
    TRADE_SESSION_ASIA = False           # Enable trading during Asia session (often lower liquidity)

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
    ADX_RANGING_THRESHOLD = 15           # Relaxed from 25
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
    TELEGRAM_ENABLED = False             # Set to True after running setup_telegram.py
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

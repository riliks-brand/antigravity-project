# config.py
# Professional MT5 Trading Bot — Elite Configuration v3.0

import MetaTrader5 as mt5

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
    # Risk Management
    # =========================================
    RISK_PERCENT_PER_TRADE = 1.0         # % of account balance risked per trade
    MAX_DAILY_LOSS_PCT = 5.0             # Kill switch: stop trading if daily loss > X%
    MAX_CONCURRENT_TRADES = 3            # Max open positions at any time

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

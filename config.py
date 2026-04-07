# config.py
# MT5 Configuration and Project Settings

import MetaTrader5 as mt5

class Config:
    # -------------------------------------
    # MetaTrader 5 Credentials & Connection
    # -------------------------------------
    # REPLACE WITH YOUR ACTUAL DEMO CREDENTIALS
    LOGIN = 260633091
    PASSWORD = "P@$$w0rd21"
    SERVER = "Exness-MT5Trial15"
    
    # -------------------------------------
    # Trading Data Settings
    # -------------------------------------
    ATR_THRESHOLD = 0.0002 # Filter for low liquidity periods
    # Using BTCUSD because Forex is closed on weekends!
    SYMBOL = "BTCUSD"
    TIMEFRAME = mt5.TIMEFRAME_M1
    DATA_POINTS = 50000  # Number of candles to fetch
    
    # -------------------------------------
    # Feature Engineering Settings
    # -------------------------------------
    # The predictive horizon: predict if price goes up/down 5 minutes ahead
    PREDICT_LOOKAHEAD = 5  
    
    # LSTM Sequence parameters
    SEQUENCE_LENGTH = 120 # Look back 120 candles
    
    # DXY Feature
    DXY_TICKER = "DX-Y.NYB"
    
    # -------------------------------------
    # Memory Similarity Thresholds
    # -------------------------------------
    SIMILARITY_HARD_BLOCK = 80   # >= 80% similarity = trade BLOCKED
    SIMILARITY_WARNING    = 60   # >= 60% similarity = Co-Pilot confirmation required
    
    # -------------------------------------
    # OTC Scraper Settings
    # -------------------------------------
    # To switch to OTC mode, change SYMBOL to an OTC asset name
    # e.g. "EURUSD-OTC", "BTCUSD-OTC", etc.
    # The presence of "OTC" in the name triggers DOM scraping mode.
    OTC_CANDLE_INTERVAL = 60     # Aggregate DOM ticks into 60-second candles
    OTC_CDP_PORT = 9225          # Chrome DevTools Protocol port for browser connection
    
    # -------------------------------------
    # Trade Mode Selection
    # -------------------------------------
    # "FIXED_TIME" = Up/Down with expiry (Olymp Trade)
    # "FOREX"      = Live MT5 execution (Exness)
    TRADING_MODE = "FOREX"
    
    # -------------------------------------
    # Forex Mode Settings (Exness MT5)
    # -------------------------------------
    FOREX_RISK_PER_TRADE = 10.0   # Actionable risk amount in standard account currency (e.g. $)
    FOREX_SYMBOL = "BTCUSD"       # Target trading symbol in MT5 (e.g., BTCUSD, EURUSDm)
    MAGIC_NUMBER = 121052         # Unique trade ID for the bot

    
    # -------------------------------------
    # Shark Exit Settings (Dynamic Close)
    # -------------------------------------
    SHARK_POLL_INTERVAL = 10          # Seconds between live re-evaluations
    SHARK_MIN_PROFIT_TO_PROTECT = 0.0 # Min PnL ($) before exit logic activates (0 = any profit)
    SHARK_REVERSAL_BB_THRESHOLD = 0.95  # BB position >= 0.95 (near top) for BUY = reversal risk
    SHARK_OPPOSITE_SIGNAL_CONFIDENCE = 0.55  # If LSTM flips to opposite with > 55% confidence, close

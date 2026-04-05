# config.py
# MT5 Configuration and Project Settings

import MetaTrader5 as mt5

class Config:
    # -------------------------------------
    # MetaTrader 5 Credentials & Connection
    # -------------------------------------
    # REPLACE WITH YOUR ACTUAL DEMO CREDENTIALS
    LOGIN = 5048877064
    PASSWORD = "@nRtKeT3"
    SERVER = "MetaQuotes-Demo"
    
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
    # "fixed_time" = Up/Down with expiry (original mode)
    # "forex"      = Buy/Sell with Multiplier + TP/SL (new Forex mode)
    TRADE_MODE = "forex"
    
    # -------------------------------------
    # Forex Mode Settings
    # -------------------------------------
    FOREX_MULTIPLIER = "10"       # Leverage multiplier: "10" or "100"
    FOREX_DEFAULT_AMOUNT = "10"   # Default investment amount in $
    FOREX_TP_ATR_MULT = 2.0      # Take Profit = ATR × this multiplier
    FOREX_SL_ATR_MULT = 1.0      # Stop Loss   = ATR × this multiplier
    FOREX_MAX_HOLD_SECONDS = 600  # Max hold time before force-close check (10 min)
    FOREX_POLL_INTERVAL = 5       # Seconds between trade status polls

# config.py
# MT5 Configuration and Project Settings

import MetaTrader5 as mt5

class Config:
    # -------------------------------------
    # MetaTrader 5 Credentials & Connection
    # -------------------------------------
    # REPLACE WITH YOUR ACTUAL DEMO CREDENTIALS
    LOGIN = 1234567 
    PASSWORD = "YourPasswordHere"
    SERVER = "YourBroker-Demo"
    
    # -------------------------------------
    # Trading Data Settings
    # -------------------------------------
    SYMBOL = "EURUSD"
    TIMEFRAME = mt5.TIMEFRAME_M1
    DATA_POINTS = 50000  # Number of candles to fetch
    
    # -------------------------------------
    # Feature Engineering Settings
    # -------------------------------------
    # The predictive horizon: predict if price goes up/down 5 minutes ahead
    PREDICT_LOOKAHEAD = 5  
    
    # LSTM Sequence parameters
    SEQUENCE_LENGTH = 60 # Look back 60 candles

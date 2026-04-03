import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from config import Config

def init_mt5():
    """
    Initializes MT5, logs in, and returns True if successful.
    """
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
        
    # Attempt to log in to the demo account
    # Note: If no params passed to login(), it uses the last logged in account
    # For automated scripts, it's safer to pass credentials if provided in config
    if hasattr(Config, 'LOGIN') and Config.LOGIN:
        authorized = mt5.login(login=Config.LOGIN, password=Config.PASSWORD, server=Config.SERVER)
        if not authorized:
            print(f"Failed to connect at account #{Config.LOGIN}, error code:", mt5.last_error())
            return False
        print(f"Successfully connected to account #{Config.LOGIN}")
    else:
        print("Connected to MT5 using default active account.")
        
    return True

def fetch_data():
    """
    Fetches the last N candles for the configured symbol and timeframe.
    Returns a Pandas DataFrame.
    """
    print(f"Fetching {Config.DATA_POINTS} candles for {Config.SYMBOL}...")
    
    # Check if symbol is available
    if not mt5.symbol_select(Config.SYMBOL, True):
        print(f"Failed to select {Config.SYMBOL}")
        return None

    # Request historical data from the current time backwards
    rates = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME, 0, Config.DATA_POINTS)
    
    if rates is None or len(rates) == 0:
        print("Failed to fetch rates, error code =", mt5.last_error())
        return None
        
    # Convert to pandas DataFrame
    df = pd.DataFrame(rates)
    
    # MT5 returns time in seconds, convert to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Set time as index
    df.set_index('time', inplace=True)
    
    print(f"Successfully fetched {len(df)} rows.")
    return df

if __name__ == "__main__":
    if init_mt5():
        df = fetch_data()
        if df is not None:
            print(df.head())
        mt5.shutdown()

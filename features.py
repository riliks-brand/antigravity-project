import pandas as pd
import numpy as np
from config import Config
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds RSI, MACD, Bollinger Bands, and ATR to the dataframe.
    """
    # RSI
    df['RSI'] = RSIIndicator(close=df["close"], window=14).rsi()
    
    # MACD
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df['BB_high'] = bb.bollinger_hband()
    df['BB_low'] = bb.bollinger_lband()
    df['BB_mid'] = bb.bollinger_mavg()
    
    # ATR
    df['ATR'] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
    
    return df

def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates numerical representations of candlestick patterns.
    """
    # Calculate real body size and shadows
    df['body_size'] = abs(df['close'] - df['open'])
    
    # Upper shadow length
    df['upper_shadow'] = df.apply(lambda row: row['high'] - max(row['open'], row['close']), axis=1)
    
    # Lower shadow length
    df['lower_shadow'] = df.apply(lambda row: min(row['open'], row['close']) - row['low'], axis=1)
    
    # Feature scaling: shadows relative to body size 
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    df['upper_shadow_ratio'] = df['upper_shadow'] / (df['body_size'] + epsilon)
    df['lower_shadow_ratio'] = df['lower_shadow'] / (df['body_size'] + epsilon)
    
    # Body direction (1 for bullish, -1 for bearish, 0 for doji)
    df['body_direction'] = np.where(df['close'] > df['open'], 1, np.where(df['close'] < df['open'], -1, 0))
    
    return df

def generate_target_column(df: pd.DataFrame, lookahead: int = Config.PREDICT_LOOKAHEAD) -> pd.DataFrame:
    """
    Creates the Target column.
    1 if Close price N periods from now is strictly greater than current Close.
    0 otherwise.
    """
    # Create the future close column by shifting backwards
    df['future_close'] = df['close'].shift(-lookahead)
    
    # Generate Target
    df['Target'] = np.where(df['future_close'] > df['close'], 1, 0)
    
    # We drop the 'future_close' so the model doesn't cheat
    df.drop(['future_close'], axis=1, inplace=True)
    
    return df

def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the entire feature engineering pipeline on raw OHLC data.
    """
    print("Starting feature engineering...")
    
    df = df.copy()
    
    df = add_technical_indicators(df)
    df = add_price_action_features(df)
    df = generate_target_column(df)
    
    # After generating features and the target (which shifts by lookahead),
    # there will be NaN values at the Beginning (from indicators) and at the End (from target shift).
    # We must drop these NaN rows to have clean data for the model.
    print(f"Data shape before dropping NaNs: {df.shape}")
    df.dropna(inplace=True)
    print(f"Data shape after dropping NaNs: {df.shape}")
    
    return df

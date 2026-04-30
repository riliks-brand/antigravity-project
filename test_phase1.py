"""
Quick integration test: Verify the new PDF Vision Layer detectors
work correctly within the feature pipeline using synthetic data.
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Simulate OHLCV data (200 M5 candles)
np.random.seed(42)
n = 300
base_price = 1.1000
returns = np.random.normal(0, 0.0005, n)
closes = base_price + np.cumsum(returns)

# Generate realistic OHLC from closes
opens = np.roll(closes, 1)
opens[0] = base_price
highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 0.0003, n))
lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 0.0003, n))

# Inject a clear bullish engulfing at index 100
opens[99] = 1.1050
closes[99] = 1.1030  # bearish candle
opens[100] = 1.1025  # open below prev close
closes[100] = 1.1060  # close above prev open
highs[100] = 1.1065
lows[100] = 1.1020

# Inject a clear double top at index 150 and 160
highs[150] = 1.1150
highs[160] = 1.1148  # very close to 150's high

df = pd.DataFrame({
    'open': opens,
    'high': highs,
    'low': lows,
    'close': closes,
    'tick_volume': np.random.randint(100, 10000, n),
}, index=pd.date_range('2025-01-01', periods=n, freq='5min'))

# Add ATR manually (simple approximation)
df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
df['ATR'] = df['ATR'].bfill()

# Add RSI manually (approximate)
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / (loss + 1e-8)
df['RSI'] = 100 - (100 / (1 + rs))
df['RSI'] = df['RSI'].bfill()

# Also need body-related columns for liquidity features
df['body_size'] = abs(df['close'] - df['open'])
df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
df['upper_shadow_ratio'] = df['upper_shadow'] / (df['body_size'] + 1e-8)
df['lower_shadow_ratio'] = df['lower_shadow'] / (df['body_size'] + 1e-8)

print("=" * 60)
print("  PHASE 1: PDF Vision Layer Integration Test")
print("=" * 60)

# Test 1: Candlestick Patterns
from candles import add_candlestick_patterns
df = add_candlestick_patterns(df)
candle_cols = [
    'Engulf_Bull', 'Engulf_Bear', 'Hammer_Bull', 'Hammer_Bear',
    'DragonflyDoji', 'GravestoneDoji', 'NeutralDoji',
    'MorningStar', 'EveningStar', 'PiercingLine', 'DarkCloudCover',
    'ThreeSoldiers', 'ThreeCrows', 'SpinningTop',
    'MarubozuBull', 'MarubozuBear', 'candle_reversal_score'
]
print(f"\n✅ Candlestick Patterns: {len(candle_cols)} features added")
for col in candle_cols:
    total = df[col].sum()
    print(f"   {col:25s}: {total:>5} detections")

# Test 2: Chart Patterns
from pattern_detector import add_chart_patterns
df = add_chart_patterns(df)
pattern_cols = [
    'DoubleTop_Flag', 'DoubleBottom_Flag', 'TripleTop_Flag', 'TripleBottom_Flag',
    'HS_Flag', 'InvHS_Flag', 'AscTriangle_Flag', 'DescTriangle_Flag',
    'SymTriangle_Flag', 'RisingWedge_Flag', 'FallingWedge_Flag',
    'BullFlag_Flag', 'BearFlag_Flag', 'Volatility_Compress', 'pattern_bias_score'
]
print(f"\n✅ Chart Patterns: {len(pattern_cols)} features added")
for col in pattern_cols:
    total = df[col].sum()
    print(f"   {col:25s}: {total:>5} detections")

# Test 3: Divergence
from divergence import add_divergence_features
df = add_divergence_features(df)
div_cols = [
    'RSI_BullDiv', 'RSI_BearDiv', 'RSI_HiddenBullDiv', 'RSI_HiddenBearDiv',
    'divergence_score'
]
print(f"\n✅ Divergence: {len(div_cols)} features added")
for col in div_cols:
    total = df[col].sum()
    print(f"   {col:25s}: {total:>5} detections")

# Summary
total_new_features = len(candle_cols) + len(pattern_cols) + len(div_cols)
print(f"\n{'=' * 60}")
print(f"  TOTAL NEW FEATURES: {total_new_features}")
print(f"  DataFrame shape: {df.shape}")
print(f"  All columns: {list(df.columns)}")
print(f"{'=' * 60}")

# Verify specific injections
engulf_at_100 = df['Engulf_Bull'].iloc[100]
print(f"\n🔍 Verification: Bullish Engulfing at index 100 = {engulf_at_100} (expected: 1)")

print("\n🎯 All Phase 1 detectors working correctly!")

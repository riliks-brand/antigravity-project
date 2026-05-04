"""
Phase 2: Smart Exit Integration Test
Tests the smart_exit evaluator with synthetic data to ensure
it correctly detects danger signals and makes exit/tighten decisions.
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

os.environ["PYTHONIOENCODING"] = "utf-8"

from smart_exit import evaluate_smart_exit, should_tighten_sl

# ===== Create base DataFrame with Phase 1 columns =====
def make_base_df(n=100):
    """Create a minimal DataFrame with ATR, RSI, and all Phase 1 columns set to 0."""
    np.random.seed(42)
    base = 1.1000
    closes = base + np.cumsum(np.random.normal(0, 0.0005, n))
    opens = np.roll(closes, 1); opens[0] = base
    highs = np.maximum(opens, closes) + 0.0003
    lows = np.minimum(opens, closes) - 0.0003

    df = pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows, 'close': closes,
        'ATR': 0.001, 'RSI': 50.0,
        # Phase 1: Candles (all zero)
        'Engulf_Bull': 0, 'Engulf_Bear': 0, 'Hammer_Bull': 0, 'Hammer_Bear': 0,
        'DragonflyDoji': 0, 'GravestoneDoji': 0, 'NeutralDoji': 0,
        'MorningStar': 0, 'EveningStar': 0, 'PiercingLine': 0, 'DarkCloudCover': 0,
        'ThreeSoldiers': 0, 'ThreeCrows': 0, 'SpinningTop': 0,
        'MarubozuBull': 0, 'MarubozuBear': 0, 'candle_reversal_score': 0,
        # Phase 1: Patterns (all zero)
        'DoubleTop_Flag': 0, 'DoubleBottom_Flag': 0,
        'TripleTop_Flag': 0, 'TripleBottom_Flag': 0,
        'HS_Flag': 0, 'InvHS_Flag': 0,
        'AscTriangle_Flag': 0, 'DescTriangle_Flag': 0, 'SymTriangle_Flag': 0,
        'RisingWedge_Flag': 0, 'FallingWedge_Flag': 0,
        'BullFlag_Flag': 0, 'BearFlag_Flag': 0,
        'Volatility_Compress': 0, 'pattern_bias_score': 0,
        # Phase 1: Divergence (all zero)
        'RSI_BullDiv': 0, 'RSI_BearDiv': 0,
        'RSI_HiddenBullDiv': 0, 'RSI_HiddenBearDiv': 0,
        'divergence_score': 0,
    })
    return df


print("=" * 60)
print("  PHASE 2: Smart Exit Integration Test")
print("=" * 60)

# ===== TEST 1: No signals = No exit =====
df = make_base_df()
trade_info = {"ticket": 1001, "symbol": "EURUSD", "entry_price": 1.1000,
              "candles_open": 10, "current_pnl": 0.0010}
should_exit, reason, danger = evaluate_smart_exit("BUY", df, trade_info)
assert not should_exit, f"Test 1 failed: should_exit={should_exit}"
assert danger == 0.0, f"Test 1 failed: danger={danger}"
print("[PASS] Test 1: No signals -> No exit (danger=0.0)")

# ===== TEST 2: Bearish Engulfing on BUY = danger +2.0 =====
df2 = make_base_df()
df2.loc[df2.index[-1], 'Engulf_Bear'] = 1
should_exit, reason, danger = evaluate_smart_exit("BUY", df2, trade_info)
assert not should_exit, f"Test 2 failed: should_exit={should_exit} (danger should be 2.0 < 3.0)"
assert danger == 2.0, f"Test 2 failed: danger={danger}"
assert should_tighten_sl(danger), f"Test 2 failed: should tighten but didn't"
print(f"[PASS] Test 2: Bearish Engulfing on BUY -> danger=2.0 (TIGHTEN)")

# ===== TEST 3: Bearish Engulfing + RSI Bearish Div = danger 5.0 -> EXIT =====
df3 = make_base_df()
df3.loc[df3.index[-1], 'Engulf_Bear'] = 1
df3.loc[df3.index[-1], 'RSI_BearDiv'] = 1
should_exit, reason, danger = evaluate_smart_exit("BUY", df3, trade_info)
assert should_exit, f"Test 3 failed: should_exit={should_exit}"
assert danger == 5.0, f"Test 3 failed: danger={danger}"
print(f"[PASS] Test 3: Engulf_Bear + RSI_BearDiv on BUY -> danger=5.0 (EXIT)")

# ===== TEST 4: Trade too young (2 candles) = No exit =====
young_info = {"ticket": 1002, "symbol": "EURUSD", "entry_price": 1.1000,
              "candles_open": 2, "current_pnl": 0.0010}
df4 = make_base_df()
df4.loc[df4.index[-1], 'Engulf_Bear'] = 1
df4.loc[df4.index[-1], 'RSI_BearDiv'] = 1
should_exit, reason, danger = evaluate_smart_exit("BUY", df4, young_info)
assert not should_exit, f"Test 4 failed: should_exit={should_exit}"
assert "too young" in reason.lower(), f"Test 4 failed: reason={reason}"
print(f"[PASS] Test 4: Trade only 2 candles old -> No exit (safety guard)")

# ===== TEST 5: Trade in loss = No exit (profit-only mode) =====
losing_info = {"ticket": 1003, "symbol": "EURUSD", "entry_price": 1.1000,
               "candles_open": 10, "current_pnl": -0.0020}
df5 = make_base_df()
df5.loc[df5.index[-1], 'Engulf_Bear'] = 1
df5.loc[df5.index[-1], 'RSI_BearDiv'] = 1
should_exit, reason, danger = evaluate_smart_exit("BUY", df5, losing_info)
assert not should_exit, f"Test 5 failed: should_exit={should_exit}"
assert "not in profit" in reason.lower(), f"Test 5 failed: reason={reason}"
print(f"[PASS] Test 5: Trade in loss -> No exit (letting SL handle it)")

# ===== TEST 6: Bullish signals on SELL = danger for SELL =====
df6 = make_base_df()
df6.loc[df6.index[-1], 'Engulf_Bull'] = 1
df6.loc[df6.index[-1], 'MorningStar'] = 1
sell_info = {"ticket": 1004, "symbol": "EURUSD", "entry_price": 1.1000,
             "candles_open": 10, "current_pnl": 0.0015}
should_exit, reason, danger = evaluate_smart_exit("SELL", df6, sell_info)
assert danger == 4.5, f"Test 6 failed: danger={danger}"
assert should_exit, f"Test 6 failed: should_exit={should_exit}"
print(f"[PASS] Test 6: Engulf_Bull + MorningStar on SELL -> danger=4.5 (EXIT)")

# ===== TEST 7: Bearish signals DON'T affect SELL trades =====
df7 = make_base_df()
df7.loc[df7.index[-1], 'Engulf_Bear'] = 1
df7.loc[df7.index[-1], 'EveningStar'] = 1
should_exit, reason, danger = evaluate_smart_exit("SELL", df7, sell_info)
assert not should_exit, f"Test 7 failed: should_exit={should_exit}"
assert danger == 0.0, f"Test 7 failed: danger={danger}"
print(f"[PASS] Test 7: Bearish signals on SELL -> danger=0.0 (no conflict)")

# ===== TEST 8: Volatility Squeeze amplifies danger =====
df8 = make_base_df()
df8.loc[df8.index[-1], 'Engulf_Bear'] = 1
df8.loc[df8.index[-1], 'Volatility_Compress'] = 1
should_exit, reason, danger = evaluate_smart_exit("BUY", df8, trade_info)
expected = 2.0 * 1.2  # 20% amplification
assert abs(danger - expected) < 0.01, f"Test 8 failed: danger={danger} (expected {expected})"
print(f"[PASS] Test 8: Engulf_Bear + Vol Squeeze -> danger={danger:.1f} (amplified)")

# ===== TEST 9: Massive signal stack = HIGH danger =====
df9 = make_base_df()
df9.loc[df9.index[-1], 'Engulf_Bear'] = 1      # +2.0
df9.loc[df9.index[-1], 'EveningStar'] = 1       # +2.5
df9.loc[df9.index[-1], 'RSI_BearDiv'] = 1       # +3.0
df9.loc[df9.index[-1], 'HS_Flag'] = 1           # +2.5
df9.loc[df9.index[-1], 'DoubleTop_Flag'] = 1    # +2.0
should_exit, reason, danger = evaluate_smart_exit("BUY", df9, trade_info)
assert danger >= 10.0, f"Test 9 failed: danger={danger} (expected >= 10.0)"
assert should_exit, f"Test 9 failed: should_exit={should_exit}"
print(f"[PASS] Test 9: 5 bearish signals stacked -> danger={danger:.1f} (CRITICAL EXIT)")

print(f"\n{'=' * 60}")
print(f"  ALL 9 TESTS PASSED!")
print(f"{'=' * 60}")

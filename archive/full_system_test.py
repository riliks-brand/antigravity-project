"""
Full System Comprehensive Test
==============================
Tests all major components implemented in Phase 1, 2, 3, and 4.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.environ["PYTHONIOENCODING"] = "utf-8"

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def test_phase_1_features():
    print_header("TESTING PHASE 1: Vision Layer (Features)")
    try:
        from data_loader import fetch_mt5_ohlc
        from features import feature_engineering_pipeline
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            print("[FAIL] MT5 Init failed")
            return False
            
        df = fetch_mt5_ohlc("EURUSD", mt5.TIMEFRAME_M5, 100)
        if df is None:
            print("[FAIL] Fetch failed")
            return False
            
        proc_df = feature_engineering_pipeline(df)
        required_cols = ['Engulf_Bull', 'RSI_BearDiv', 'DoubleTop_Flag', 'ATR']
        missing = [c for c in required_cols if c not in proc_df.columns]
        
        if missing:
            print(f"[FAIL] Missing features: {missing}")
            return False
        else:
            print(f"[PASS] Feature Engineering generated {len(proc_df.columns)} features.")
            print(f"[PASS] Phase 1 Vision Layer OK")
            return True
    except Exception as e:
        print(f"[FAIL] Phase 1 Error: {e}")
        return False

def test_phase_2_smart_exit():
    print_header("TESTING PHASE 2: Smart Exit")
    try:
        import subprocess
        result = subprocess.run(["python", "test_phase2.py"], capture_output=True, text=True, encoding='utf-8')
        if "ALL 9 TESTS PASSED" in result.stdout:
            print("[PASS] Smart Exit Logic OK (9/9 Unit Tests Passed)")
            return True
        else:
            print("[FAIL] Smart Exit Tests failed or output unexpected.")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"[FAIL] Phase 2 Error: {e}")
        return False

def test_phase_3_macro():
    print_header("TESTING PHASE 3: Macro Context (DXY)")
    try:
        from macro_context import get_dxy_strength
        strength = get_dxy_strength()
        print(f"[PASS] DXY Strength Calculated: {strength:+.3f}")
        if -1.0 <= strength <= 1.0:
            print("[PASS] Macro Context Logic OK")
            return True
        else:
            print("[FAIL] DXY Strength out of bounds")
            return False
    except Exception as e:
        print(f"[FAIL] Phase 3 Error: {e}")
        return False

def test_phase_4_sentiment():
    print_header("TESTING PHASE 4: Sentiment Analyzer (NLP)")
    try:
        from sentiment_analyzer import analyze_sentiment_usd
        score = analyze_sentiment_usd()
        print(f"[PASS] Live USD Sentiment Score: {score:+.3f}")
        if -1.0 <= score <= 1.0:
            print("[PASS] Sentiment Analyzer OK")
            return True
        else:
            print("[FAIL] Sentiment Score out of bounds")
            return False
    except Exception as e:
        print(f"[FAIL] Phase 4 NLP Error: {e}")
        return False

def test_ensemble_integration():
    print_header("TESTING INTEGRATION: Ensemble Engine with Macro")
    try:
        from ensemble_engine import ensemble_predict
        # Simulate a BUY setup on EURUSD while DXY is very strong (+0.8) -> Should penalize
        atr_series = pd.Series(np.random.normal(0.001, 0.0001, 50))
        
        # Test 1: Strong USD penalizes EURUSD Buy
        dec_penalized = ensemble_predict(
            lstm_prob=0.8, rf_prob=0.8, current_adx=40.0,
            current_atr=0.001, atr_series=atr_series, session="London",
            dxy_strength=0.8, symbol="EURUSD"
        )
        
        # Test 2: Strong USD boosts USDJPY Buy
        dec_boosted = ensemble_predict(
            lstm_prob=0.8, rf_prob=0.8, current_adx=40.0,
            current_atr=0.001, atr_series=atr_series, session="London",
            dxy_strength=0.8, symbol="USDJPY"
        )
        
        print(f"EURUSD DXY Influence (Buy vs Strong USD): {dec_penalized.dxy_influence:+.3f}")
        print(f"USDJPY DXY Influence (Buy vs Strong USD): {dec_boosted.dxy_influence:+.3f}")
        
        if dec_penalized.dxy_influence < 0 and dec_boosted.dxy_influence > 0:
            print("[PASS] Ensemble Macro Integration OK")
            return True
        else:
            print("[FAIL] Ensemble Macro Integration Failed")
            return False
    except Exception as e:
        print(f"[FAIL] Ensemble Test Error: {e}")
        return False

def run_all_tests():
    p1 = test_phase_1_features()
    p2 = test_phase_2_smart_exit()
    p3 = test_phase_3_macro()
    p4 = test_phase_4_sentiment()
    pe = test_ensemble_integration()
    
    print_header("SYSTEM TEST RESULTS")
    results = [p1, p2, p3, p4, pe]
    if all(results):
        print("[PASS] ALL SYSTEMS GO. Hedge Fund Bot architecture verified.")
    else:
        print("[FAIL] ERRORS DETECTED. Check logs above.")

if __name__ == "__main__":
    run_all_tests()

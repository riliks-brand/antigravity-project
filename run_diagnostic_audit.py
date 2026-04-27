import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import joblib
import os
import sys

# Optional GPU suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model

from config import Config
from data_loader import fetch_mtf_data
from mt5_engine import connect_to_exness
from features import feature_engineering_pipeline
from rf_model import RFModel
from ensemble_engine import ensemble_predict

def run_diagnostic():
    print("=" * 60)
    print("   🚀 RUNNING DIAGNOSTIC AUDIT (OFFLINE SIMULATION)")
    print("=" * 60)
    
    if not connect_to_exness():
        print("[Fatal] MT5 connection failed.")
        return
        
    # Load Models
    lstm_model_path = "lstm_model.h5"
    lstm_scaler_path = "lstm_scaler.joblib"
    
    if not (os.path.exists(lstm_model_path) and os.path.exists(lstm_scaler_path)):
        print("[Fatal] Models not found.")
        return
        
    lstm_model = load_model(lstm_model_path)
    lstm_scaler = joblib.load(lstm_scaler_path)
    rf_model = RFModel()
    
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US30"]
    
    stats = {
        "EVALS": 0,
        "MODEL_LIMITATION": 0,
        "THRESHOLD_BLOCK": 0,
        "NATIVE_EXECUTION": 0,
        "base_scores": [],
        "trend_strengths": []
    }
    
    print("\nFetching data and evaluating recent candles...")
    for symbol in symbols:
        mtf_data = fetch_mtf_data(symbol)
        if mtf_data is None:
            continue
            
        df_m5 = mtf_data.get("M5")
        df_m15 = mtf_data.get("M15", pd.DataFrame())
        df_h1 = mtf_data.get("H1", pd.DataFrame())
        
        if df_m5 is None or df_m5.empty:
            continue
            
        processed_df = feature_engineering_pipeline(
            df_m5, df_confirm=df_m15 if not df_m15.empty else None, df_trend=df_h1 if not df_h1.empty else None
        )
        if processed_df is None or processed_df.empty:
            continue
            
        # We will iterate through the last 50 candles to simulate live loop
        # (Assuming SEQUENCE_LENGTH is 60, we need at least 60 history)
        if len(processed_df) < Config.SEQUENCE_LENGTH + 50:
            continue
            
        for i in range(len(processed_df) - 50, len(processed_df)):
            slice_df = processed_df.iloc[:i+1].copy()
            
            latest_features = slice_df.drop(['Target'], axis=1, errors='ignore').values
            latest_features_scaled = lstm_scaler.transform(latest_features)
            
            if len(latest_features_scaled) < Config.SEQUENCE_LENGTH:
                continue
                
            X_live = np.array([latest_features_scaled[-Config.SEQUENCE_LENGTH:]])
            lstm_prob = float(lstm_model.predict(X_live, verbose=0)[0][0])
            
            rf_prob = rf_model.predict_proba(slice_df)
            
            current_atr = slice_df['ATR'].iloc[-1] if not pd.isna(slice_df['ATR'].iloc[-1]) else 0.001
            h1_adx_val = slice_df['H1_ADX'].iloc[-1] if 'H1_ADX' in slice_df.columns else 25.0
            current_adx = h1_adx_val if not pd.isna(h1_adx_val) else 25.0
            atr_series = slice_df['ATR'].dropna()
            
            # Use 'UNKNOWN' session for simulation to prevent hard block on time if outside hours
            session = "London"  # Force a valid session to avoid out-of-hours skipping
            
            decision_orig = ensemble_predict(
                lstm_prob=lstm_prob, rf_prob=rf_prob, current_adx=current_adx,
                current_atr=current_atr, atr_series=atr_series, session=session,
                diagnostic=False
            )
            
            decision_diag = ensemble_predict(
                lstm_prob=lstm_prob, rf_prob=rf_prob, current_adx=current_adx,
                current_atr=current_atr, atr_series=atr_series, session=session,
                diagnostic=True
            )
            
            stats["EVALS"] += 1
            stats["base_scores"].append(decision_orig.weighted_avg)
            stats["trend_strengths"].append(decision_orig.trend_strength)
            
            if decision_orig.direction is None and decision_diag.direction is not None:
                stats["THRESHOLD_BLOCK"] += 1
            elif decision_orig.direction is None and decision_diag.direction is None:
                stats["MODEL_LIMITATION"] += 1
            elif decision_orig.direction is not None:
                stats["NATIVE_EXECUTION"] += 1

    print("\n========================================")
    print("      🧪 DIAGNOSTIC SUMMARY REPORT      ")
    print("========================================")
    print(f" Total Evaluations: {stats['EVALS']}")
    if stats['EVALS'] > 0:
        print(f" % MODEL_LIMITATION: {stats['MODEL_LIMITATION'] / stats['EVALS'] * 100:.1f}% ({stats['MODEL_LIMITATION']})")
        print(f" % THRESHOLD_BLOCK:  {stats['THRESHOLD_BLOCK'] / stats['EVALS'] * 100:.1f}% ({stats['THRESHOLD_BLOCK']})")
        print(f" % NATIVE_EXECUTION: {stats['NATIVE_EXECUTION'] / stats['EVALS'] * 100:.1f}% ({stats['NATIVE_EXECUTION']})")
        print(f" Avg base_score:     {np.mean(stats['base_scores']):.4f}")
        print(f" Avg trend_strength: {np.mean(stats['trend_strengths']):.4f}")
    print("========================================\n")

if __name__ == "__main__":
    run_diagnostic()

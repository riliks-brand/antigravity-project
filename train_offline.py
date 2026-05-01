import os
import sys
import numpy as np
import pandas as pd
import datetime
import logging
import MetaTrader5 as mt5

from config import Config
from data_loader import fetch_mt5_ohlc
from features import feature_engineering_pipeline
from lstm_model import prepare_sequential_data, train_and_evaluate
from rf_model import RFModel

# Ensure consistent paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainOffline")

def compute_class_weights(y):
    """Computes sample weights to handle class imbalance (0=SELL, 1=BUY)."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    weight_dict = dict(zip(classes, weights))
    logger.info("Computed class weights: %s", weight_dict)
    
    # Map to sample_weights array
    sample_weights = np.zeros_like(y, dtype=float)
    for c, w in weight_dict.items():
        sample_weights[y == c] = w
    return sample_weights

def visualize_predictions(y_pred, name="Model"):
    """Prints a histogram of predictions to verify they aren't clustered at 0.5."""
    bins = [0.0, 0.2, 0.4, 0.45, 0.55, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(y_pred, bins=bins)
    
    print(f"\n{'='*40}")
    print(f" {name} Prediction Distribution")
    print(f"{'='*40}")
    labels = ["Strong SELL (<0.2)", "Weak SELL (0.2-0.4)", "Neutral SELL (0.4-0.45)",
              "NOISE (0.45-0.55)", "Neutral BUY (0.55-0.6)", "Weak BUY (0.6-0.8)", "Strong BUY (>0.8)"]
    
    for count, label in zip(hist, labels):
        bar = "#" * int(count / max(hist) * 40) if max(hist) > 0 else ""
        print(f"  {label:23s}: {count:5d} {bar}")
    print(f"{'='*40}\n")

def main():
    logger.info("Initializing MT5...")
    if not mt5.initialize():
        logger.error("MT5 Initialization failed. Make sure terminal is open.")
        return

    # 1. Fetch massive historical dataset
    logger.info("Fetching historical data (100,000 candles max)...")
    # Fetch M5 (Primary), M15 (Confirm), H1 (Trend)
    df_m5 = fetch_mt5_ohlc(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, count=50000)
    df_m15 = fetch_mt5_ohlc(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M15, count=15000)
    df_h1 = fetch_mt5_ohlc(symbol="EURUSD", timeframe=mt5.TIMEFRAME_H1, count=5000)
    
    if df_m5 is None or df_m5.empty:
        logger.error("Failed to fetch M5 data.")
        return

    # 2. Feature Engineering & Strict Labeling
    logger.info("Running Feature Engineering Pipeline...")
    # Features pipeline drops NaNs (which now includes HOLD/Noise labels)
    df_processed = feature_engineering_pipeline(df_m5, df_confirm=df_m15, df_trend=df_h1)
    
    logger.info("Data Shape after noise filtering: %s", df_processed.shape)
    
    # Check class balance
    target_counts = df_processed['Target'].value_counts()
    logger.info("Label Distribution (0=SELL, 1=BUY):\n%s", target_counts.to_string())

    # 3. Train LSTM
    logger.info("Preparing LSTM sequential data...")
    X_train, X_test, y_train, y_test, scaler, base_weights = prepare_sequential_data(df_processed)
    
    # Overwrite weights to use strict class balancing instead of just penalty weighting
    train_sample_weights = compute_class_weights(y_train)
    
    logger.info("Training LSTM Model...")
    lstm_model, history, lstm_acc = train_and_evaluate(
        X_train, X_test, y_train, y_test, sample_weights=train_sample_weights
    )
    
    # Save LSTM and scaler
    lstm_model.save("lstm_model.h5")
    from joblib import dump
    dump(scaler, "lstm_scaler.joblib")
    logger.info("LSTM Model saved to lstm_model.h5")

    # Evaluate LSTM prediction distribution
    lstm_preds = lstm_model.predict(X_test, verbose=0).flatten()
    visualize_predictions(lstm_preds, "LSTM")

    # 4. Train RF Model
    logger.info("Training Random Forest...")
    rf_engine = RFModel()
    
    # Extract features matching the ones RF expects
    feature_cols = [c for c in df_processed.columns if c != 'Target']
    
    # We train RF on the same processed dataset. rf_engine.train() splits internally
    rf_engine.train(df_processed)
    
    # Predict on the last 20% to check RF distribution
    split = int(len(df_processed) * 0.8)
    df_test = df_processed.iloc[split:]
    
    if not df_test.empty:
        # scale
        X_rf_test = rf_engine.scaler.transform(df_test[feature_cols].values)
        rf_preds = rf_engine.model.predict_proba(X_rf_test)[:, 1]
        visualize_predictions(rf_preds, "Random Forest")

    mt5.shutdown()
    logger.info("Offline training complete!")

if __name__ == "__main__":
    main()

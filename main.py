import numpy as np
from config import Config
from data_loader import init_mt5, fetch_data
from features import feature_engineering_pipeline
from executor import TradeExecutor

def main():
    if not init_mt5():
        print("Failed initialization.")
        return

    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Start Loop
    import time
    import datetime
    
    print("\n" + "="*50)
    print("INITIATING LIVE CONTINUOUS TRADING LOOP (5m Candles)")
    print("="*50)
    
    while True:
        # Wait for the next 5-minute candle close (modulo 5 minutes)
        # e.g., 10:00, 10:05, 10:10
        # We will check the current minute every 10 seconds to avoid high CPU
        now = datetime.datetime.now()
        
        # When remainder is 0, we are precisely at the close of a 5-minute candle
        # To avoid re-triggering multiple times in the same minute, we track last_trade_minute
        if 'last_trade_minute' not in locals():
            last_trade_minute = -1
            
        if now.minute % 5 == 0 and now.minute != last_trade_minute:
            print("\n" + "="*50)
            print(f"[SIGNAL EVALUATION] Trigger Time: {now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            print("="*50)
            
            last_trade_minute = now.minute
            
            # 1. Fetch raw data
            raw_df = fetch_data()
            if raw_df is None or raw_df.empty:
                print("Data fetch failed. Will retry next 5m cycle.")
                time.sleep(60)
                continue
                
            # 2. Apply Feature Engineering
            processed_df = feature_engineering_pipeline(raw_df)
            
            # 3. Train LSTM quickly
            # To avoid saving/loading overhead in this rapid loop, we train and predict each time
            from lstm_model import prepare_sequential_data, train_and_evaluate
            X_train, X_test, y_train, y_test, scaler = prepare_sequential_data(processed_df)
            model, history, acc = train_and_evaluate(X_train, X_test, y_train, y_test)
            
            # 4. Extract latest sample
            latest_features = processed_df.drop(['Target'], axis=1).values
            latest_features_scaled = scaler.transform(latest_features)
            
            if len(latest_features_scaled) >= Config.SEQUENCE_LENGTH:
                X_live = latest_features_scaled[-Config.SEQUENCE_LENGTH:]
                X_live = np.array([X_live])
                
                prob = model.predict(X_live)[0][0]
                action = "buy" if prob > 0.5 else "sell"
                
                sig_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                print(f"[Prediction Generated] {sig_time} -> Prob: {prob:.4f} -> {action.upper()}")
                
                # 5. Strike execution
                executor = TradeExecutor()
                print("Launching Web Executer for Olymp Trade...")
                success, output_msg = executor.execute_web(action=action)
                
                if success:
                    print(f"\n[LIVE TRADE VERIFIED: {action.upper()} ON OLYMP TRADE]")
                else:
                    print(f"Failed to open trade on Olymp Trade. Reason: {output_msg}")
                    print(f"!!! CRITICAL ALERT: Check 'error_screenshot.png' for exact DOM failure !!!")
            else:
                print("Not enough data to form a sequence.")
        else:
            time.sleep(5) # Sleep 5 seconds before checking time again

if __name__ == "__main__":
    main()

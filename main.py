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

    import time
    import datetime
    
    print("\n" + "="*50)
    print("INITIATING LIVE CONTINUOUS TRADING LOOP (5m Candles)")
    print("="*50)
    
    executor = TradeExecutor()
    last_trade_minute = -1
    warm_session = None  # (proc, playwright, browser, page) — filled during warm-up
    
    while True:
        now = datetime.datetime.now()
        minutes_to_next = 5 - (now.minute % 5)
        seconds_to_next = minutes_to_next * 60 - now.second
        
        # ===== WARM-UP PHASE: Open browser 60s before candle close =====
        if seconds_to_next <= 60 and seconds_to_next > 5 and warm_session is None:
            print(f"\n[WARM-UP] Candle closes in ~{seconds_to_next}s. Pre-opening browser...")
            try:
                warm_session = executor.warm_up_browser()
            except Exception as e:
                print(f"[WARM-UP] Failed: {e}. Will cold-start at execution time.")
                warm_session = None
        
        # ===== EXECUTION PHASE: At the candle close =====
        if now.minute % 5 == 0 and now.minute != last_trade_minute:
            print("\n" + "="*50)
            print(f"[SIGNAL EVALUATION] Trigger Time: {now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            print("="*50)
            
            last_trade_minute = now.minute
            
            # 1. Fetch raw data (DXY in parallel thread)
            raw_df = fetch_data()
            if raw_df is None or raw_df.empty:
                print("Data fetch failed. Will retry next 5m cycle.")
                # Clean up warm session if it exists
                if warm_session:
                    try:
                        warm_session[1].stop()  # playwright
                        warm_session[0].terminate()  # proc
                    except:
                        pass
                    warm_session = None
                time.sleep(60)
                continue
                
            # 2. Apply Feature Engineering
            processed_df = feature_engineering_pipeline(raw_df)
            
            # 3. Train LSTM
            from lstm_model import prepare_sequential_data, train_and_evaluate
            X_train, X_test, y_train, y_test, scaler = prepare_sequential_data(processed_df)
            model, history, acc = train_and_evaluate(X_train, X_test, y_train, y_test)
            
            # 4. Predict
            latest_features = processed_df.drop(['Target'], axis=1).values
            latest_features_scaled = scaler.transform(latest_features)
            
            if len(latest_features_scaled) >= Config.SEQUENCE_LENGTH:
                X_live = latest_features_scaled[-Config.SEQUENCE_LENGTH:]
                X_live = np.array([X_live])
                
                # === DXY MONITOR: Confirm latest DXY value fed to model ===
                if 'DXY_Close' in processed_df.columns:
                    last_dxy = processed_df['DXY_Close'].iloc[-1]
                    print(f"[DXY Monitor] Latest DXY value fed to model: {last_dxy:.4f}")
                else:
                    print("[DXY Monitor] WARNING: DXY_Close column not found. Model running without DXY.")
                
                prob = model.predict(X_live)[0][0]
                raw_action = "buy" if prob > 0.5 else "sell"
                
                sig_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                print(f"[Prediction Generated] {sig_time} -> Prob: {prob:.4f} -> {raw_action.upper()}")
                
                # === BOLLINGER TREND FILTER ===
                action = raw_action
                if 'BB_mid' in processed_df.columns:
                    last_close = processed_df['close'].iloc[-1]
                    bb_mid = processed_df['BB_mid'].iloc[-1]
                    bb_high = processed_df['BB_high'].iloc[-1]
                    bb_low = processed_df['BB_low'].iloc[-1]
                    print(f"[Bollinger] Close: {last_close:.4f} | BB_mid: {bb_mid:.4f} | BB_high: {bb_high:.4f} | BB_low: {bb_low:.4f}")
                    
                    if raw_action == "buy" and last_close < bb_mid:
                        print(f"[Bollinger FILTER] BUY BLOCKED: Price ({last_close:.4f}) is BELOW BB Middle ({bb_mid:.4f}). Trend is bearish.")
                        print("[Bollinger FILTER] SKIP: Better to wait for trade-aligned signal.")
                        action = None
                    elif raw_action == "sell" and last_close > bb_mid:
                        print(f"[Bollinger FILTER] SELL BLOCKED: Price ({last_close:.4f}) is ABOVE BB Middle ({bb_mid:.4f}). Trend is bullish.")
                        print("[Bollinger FILTER] SKIP: Better to wait for trade-aligned signal.")
                        action = None
                    else:
                        print(f"[Bollinger FILTER] PASSED: {action.upper()} confirmed by trend direction.")
                else:
                    print("[Bollinger] WARNING: BB_mid not found. Skipping trend filter.")
                
                # 5. Execute only if passed filter
                if action:
                    print(f"[FINAL DECISION] {action.upper()} (raw: {raw_action.upper()}, prob: {prob:.4f})")
                    print("Launching Web Executer for Olymp Trade...")
                    success, output_msg = executor.execute_web(
                        action=action, 
                        warm_session=warm_session
                    )
                else:
                    print("[FINAL DECISION] NO TRADE (Filter Blocked)")
                    # Clean up session
                    if warm_session:
                        try:
                            warm_session[1].stop()
                            warm_session[0].terminate()
                        except:
                            pass
                warm_session = None  # Session consumed/cleaned, will re-warm next cycle
                
                if success:
                    print(f"\n[LIVE TRADE VERIFIED: {action.upper()} ON OLYMP TRADE]")
                else:
                    print(f"Failed to open trade on Olymp Trade. Reason: {output_msg}")
                    print(f"!!! CRITICAL ALERT: Check 'error_screenshot.png' for exact DOM failure !!!")
            else:
                print("Not enough data to form a sequence.")
                # Clean up warm session
                if warm_session:
                    try:
                        warm_session[1].stop()
                        warm_session[0].terminate()
                    except:
                        pass
                    warm_session = None
        else:
            time.sleep(3)  # Check every 3 seconds

if __name__ == "__main__":
    main()

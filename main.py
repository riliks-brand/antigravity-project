import numpy as np
from config import Config
from data_loader import init_mt5, fetch_data
from features import feature_engineering_pipeline
from executor import TradeExecutor
import pandas as pd
import time
import datetime
import os
import MetaTrader5 as mt5
import csv

def log_loss(processed_df):
    try:
        last_row = processed_df.iloc[-1]
        loss_dict = {
            'DXY': last_row.get('DXY_Close', 0),
            'BB_Pos': last_row.get('close', 0) - last_row.get('BB_mid', 0) if 'BB_mid' in last_row else 0,
            'RSI': last_row.get('RSI', 0),
            'ATR': last_row.get('ATR', 0),
            'Volatility': last_row.get('Volatility', 0)
        }
        
        file_exists = os.path.isfile('losses_log.csv')
        with open('losses_log.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=loss_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(loss_dict)
        import shutil
        if not os.path.exists('archive'):
            os.makedirs('archive')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy('losses_log.csv', f'archive/losses_log_backup_{timestamp}.csv')
        
        print(f"\033[92m[Loss Logger] Appended state to losses_log.csv and backed up to archive.\033[0m")
    except Exception as e:
        print(f"\033[91mFailed to log loss: {e}\033[0m")

def main():
    if not init_mt5():
        print("Failed initialization.")
        return

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n" + "="*50)
    print("INITIATING LIVE CONTINUOUS TRADING LOOP (5m Candles)")
    print("="*50)
    
    executor = TradeExecutor()
    last_trade_minute = -1
    warm_session = None
    
    while True:
        success = None
        try:
            now = datetime.datetime.now()
            minutes_to_next = 5 - (now.minute % 5)
            seconds_to_next = minutes_to_next * 60 - now.second
            
            if seconds_to_next <= 60 and seconds_to_next > 5 and warm_session is None:
                print(f"\n[WARM-UP] Candle closes in ~{seconds_to_next}s. Pre-opening browser...")
                try:
                    warm_session = executor.warm_up_browser()
                except Exception as e:
                    print(f"[WARM-UP] Failed: {e}. Will cold-start at execution time.")
                    warm_session = None
            
            if now.minute % 5 == 0 and now.minute != last_trade_minute:
                print("\n" + "="*50)
                print(f"[SIGNAL EVALUATION] Trigger Time: {now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
                print("="*50)
                
                last_trade_minute = now.minute
                
                raw_df = fetch_data()
                if raw_df is None or raw_df.empty:
                    print("Data fetch failed. Initializing MT5 and retrying next cycle...")
                    init_mt5()
                    if warm_session:
                        try:
                            warm_session[1].stop()
                            warm_session[0].terminate()
                        except: pass
                        warm_session = None
                    time.sleep(60)
                    continue
                    
                processed_df = feature_engineering_pipeline(raw_df)
                
                from lstm_model import prepare_sequential_data, train_and_evaluate
                X_train, X_test, y_train, y_test, scaler, train_weights = prepare_sequential_data(processed_df)
                model, history, acc = train_and_evaluate(X_train, X_test, y_train, y_test, sample_weights=train_weights)
                
                latest_features = processed_df.drop(['Target'], axis=1).values
                latest_features_scaled = scaler.transform(latest_features)
                
                if len(latest_features_scaled) >= Config.SEQUENCE_LENGTH:
                    X_live = latest_features_scaled[-Config.SEQUENCE_LENGTH:]
                    X_live = np.array([X_live])
                    
                    if 'DXY_Close' in processed_df.columns:
                        last_dxy = processed_df['DXY_Close'].iloc[-1]
                        print(f"[DXY Monitor] Latest DXY: {last_dxy:.4f}")
                    
                    prob = model.predict(X_live)[0][0]
                    raw_action = "buy" if prob > 0.5 else "sell"
                    
                    sig_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    
                    action = raw_action
                    filter_status = "Skipped"
                    if 'BB_mid' in processed_df.columns:
                        last_close = processed_df['close'].iloc[-1]
                        bb_mid = processed_df['BB_mid'].iloc[-1]
                        dist_to_mid = last_close - bb_mid
                        
                        print(f"[Signal] Prob: {prob*100:.2f}% -> {raw_action.upper()} | Distance to BB_mid: {dist_to_mid:.4f}")
                        
                        if raw_action == "buy" and last_close < bb_mid:
                            filter_status = "BLOCKED (Bearish Trend)"
                            action = None
                        elif raw_action == "sell" and last_close > bb_mid:
                            filter_status = "BLOCKED (Bullish Trend)"
                            action = None
                        else:
                            filter_status = "PASSED"
                    else:
                        print(f"[Signal] Prob: {prob*100:.2f}% -> {raw_action.upper()}")
                        
                    print(f"Filter: {filter_status}")
                    
                    forced_tier = None
                    if not action:
                        print("\n[!] Filter Blocked.")
                        force = input("Force Trade? (y/n): ")
                        if force.strip().lower() == 'y':
                            action = raw_action
                            print("Choose Tier:")
                            print("[1] 1$ (1m)")
                            print("[2] 10$ (2m)")
                            print("[3] Custom")
                            tier_choice = input("Choice: ")
                            if tier_choice == '1': 
                                forced_tier = ("1 min", "1", 1)
                            elif tier_choice == '2': 
                                forced_tier = ("2 min", "10", 2)
                            elif tier_choice == '3':
                                cust_amt = input("Amount: ")
                                cust_dur = input("Duration (e.g. 1m): ")
                                dur_mins = int(''.join(filter(str.isdigit, cust_dur)) or '1')
                                forced_tier = (cust_dur, cust_amt, dur_mins)
                    
                    if action:
                        if forced_tier:
                            duration, amount, dur_mins = forced_tier
                        else:
                            if prob > 0.55 or prob < 0.45:
                                print("[Tier A Execution]")
                                duration, amount, dur_mins = ("2 min", "10", 2)
                            else:
                                print("[Tier B Execution]")
                                duration, amount, dur_mins = ("1 min", "1", 1)
                                
                        print(f"[FINAL DECISION] {action.upper()} | Amount: {amount}$ | Duration: {duration}")
                        
                        # Validate MT5 sync is ready
                        init_mt5()
                        entry_tick = mt5.symbol_info_tick(Config.SYMBOL)
                        if entry_tick is None:
                            print("[MT5 Sync] Warning: Could not fetch entry tick. Continuing web trade, but sync may fail.")
                            entry_price = 0
                        else:
                            entry_price = entry_tick.ask if action == "buy" else entry_tick.bid
                            print(f"[MT5 Sync] Entry Price: {entry_price}")
                        
                        # Check warm session
                        try:
                            if warm_session and warm_session[0].poll() is not None:
                                # Process died
                                warm_session = None
                        except:
                            warm_session = None
                            
                        # Execute
                        success, output_msg = executor.execute_web(
                            action=action, 
                            duration=duration,
                            amount=amount,
                            warm_session=warm_session
                        )
                        
                        warm_session = None
                        
                        if success:
                            print(f"\n[LIVE TRADE VERIFIED: {action.upper()} ON OLYMP TRADE]")
                            
                            if entry_price > 0:
                                print(f"[MT5 Sync] Waiting {dur_mins} minutes for trade completion...")
                                time.sleep(dur_mins * 60)
                                
                                init_mt5() # Ensure connected
                                exit_tick = mt5.symbol_info_tick(Config.SYMBOL)
                                if exit_tick is not None:
                                    exit_price = exit_tick.ask if action == "buy" else exit_tick.bid
                                    print(f"[MT5 Sync] Expiry Price: {exit_price} (Entry: {entry_price})")
                                    
                                    loss = False
                                    if action == "buy" and exit_price <= entry_price:
                                        loss = True
                                    elif action == "sell" and exit_price >= entry_price:
                                        loss = True
                                        
                                    if loss:
                                        print("[Outcome] LOSS detected via MT5 sync. Triggering Loss Log.")
                                        log_loss(processed_df)
                                    else:
                                        print("[Outcome] WIN detected via MT5 sync.")
                                else:
                                    print("[MT5 Sync] Failed to get exit tick.")
                        else:
                            print(f"Failed to open trade on Olymp Trade. Reason: {output_msg}")
                            print(f"!!! CRITICAL ALERT: Check 'error_screenshot.png' for exact DOM failure !!!")
                    else:
                        print("[FINAL DECISION] NO TRADE")
                        
                    # Cleanup
                    if warm_session:
                        try:
                            warm_session[1].stop()
                            warm_session[0].terminate()
                        except: pass
                        warm_session = None 
                else:
                    print("Not enough data to form a sequence.")
                    if warm_session:
                        try:
                            warm_session[1].stop()
                            warm_session[0].terminate()
                        except: pass
                        warm_session = None
            else:
                time.sleep(3)
        except Exception as e:
            print(f"[MAIN LOOP EXCEPTION] {e}")
            import traceback
            traceback.print_exc()
            init_mt5()
            if warm_session:
                try:
                    warm_session[1].stop()
                    warm_session[0].terminate()
                except: pass
                warm_session = None
            time.sleep(5)

if __name__ == "__main__":
    main()

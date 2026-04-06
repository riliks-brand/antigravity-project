import numpy as np
from config import Config
from data_loader import fetch_data
from features import feature_engineering_pipeline
from executor import TradeExecutor
import pandas as pd
import time
import datetime
import os
import csv

# ===== GLOBAL AVOIDANCE COUNTER =====
trades_avoided_by_memory = 0

def log_loss(processed_df, weekend_mode):
    try:
        log_file = f'losses_log_{Config.TRADE_MODE}.csv'
        last_row = processed_df.iloc[-1]
        loss_dict = {
            'DXY': last_row.get('DXY_Close', 0),
            'BB_Pos': last_row.get('close', 0) - last_row.get('BB_mid', 0) if 'BB_mid' in last_row else 0,
            'RSI': last_row.get('RSI', 0),
            'ATR': last_row.get('ATR', 0),
            'Volatility': last_row.get('Volatility', 0)
        }
        
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=loss_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(loss_dict)
            
        import shutil
        if not os.path.exists('archive'):
            os.makedirs('archive')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy(log_file, f'archive/losses_log_{Config.TRADE_MODE}_backup_{timestamp}.csv')
        
        print(f"\033[92m[Loss Logger] Appended state to {log_file} and backed up to archive.\033[0m")
    except Exception as e:
        print(f"\033[91mFailed to log loss: {e}\033[0m")


def compute_memory_similarity(processed_df):
    """
    Computes the maximum percentage similarity between the current market state
    and all recorded loss patterns in the trade mode's respective log.
    Returns (max_similarity_pct, matching_loss_index) or (0.0, -1) if no losses exist.
    """
    log_file = f'losses_log_{Config.TRADE_MODE}.csv'
    if not os.path.exists(log_file):
        return 0.0, -1
    
    try:
        losses_df = pd.read_csv(log_file)
        if losses_df.empty:
            return 0.0, -1
    except Exception:
        return 0.0, -1
    
    last_row = processed_df.iloc[-1]
    
    # Current state vector (normalized)
    current_state = np.array([
        last_row.get('DXY_Close', 0),
        last_row.get('close', 0) - last_row.get('BB_mid', 0) if 'BB_mid' in processed_df.columns else 0,
        last_row.get('RSI', 0),
        last_row.get('ATR', 0),
        last_row.get('Volatility', 0) if 'Volatility' in processed_df.columns else 0
    ], dtype=float)
    
    max_sim = 0.0
    match_idx = -1
    
    for idx, loss_row in losses_df.iterrows():
        loss_state = np.array([
            loss_row.get('DXY', 0),
            loss_row.get('BB_Pos', 0),
            loss_row.get('RSI', 0),
            loss_row.get('ATR', 0),
            loss_row.get('Volatility', 0)
        ], dtype=float)
        
        # Euclidean distance -> percentage similarity
        # Normalize by magnitude to get a meaningful percentage
        norm_current = np.linalg.norm(current_state)
        norm_loss = np.linalg.norm(loss_state)
        
        if norm_current == 0 and norm_loss == 0:
            similarity = 100.0
        elif norm_current == 0 or norm_loss == 0:
            similarity = 0.0
        else:
            # Cosine similarity mapped to 0-100%
            cos_sim = np.dot(current_state, loss_state) / (norm_current * norm_loss)
            # Clamp to [0, 1] (cosine can be negative for opposite directions)
            cos_sim = max(0.0, min(1.0, cos_sim))
            similarity = cos_sim * 100.0
        
        if similarity > max_sim:
            max_sim = similarity
            match_idx = idx
    
    return max_sim, match_idx


def print_memory_report(similarity_pct, match_idx):
    """
    Prints the Memory Similarity Report with color-coded tiers.
    Returns: 'BLOCK', 'WARN', or 'PASS'
    """
    global trades_avoided_by_memory
    
    print(f"\n\033[96m{'='*55}\033[0m")
    print(f"\033[96m       🧠 MEMORY SIMILARITY CHECK\033[0m")
    print(f"\033[96m{'='*55}\033[0m")
    
    if similarity_pct >= Config.SIMILARITY_HARD_BLOCK:
        # ===== HARD BLOCK =====
        trades_avoided_by_memory += 1
        print(f"\033[91m[DANGER] {similarity_pct:.1f}% Similarity with Previous Loss (#{match_idx}). Trade BLOCKED.\033[0m")
        print(f"\033[91m[Memory] \"لقد تعلمت من هذه الخسارة السابقة، ولن أدخل هذه الصفقة لأنها تشبهها بنسبة {similarity_pct:.0f}%\"\033[0m")
        print(f"\033[93m[Stats] Total Trades Avoided by Memory: {trades_avoided_by_memory}\033[0m")
        print(f"\033[96m{'='*55}\033[0m\n")
        return 'BLOCK'
    
    elif similarity_pct >= Config.SIMILARITY_WARNING:
        # ===== CO-PILOT WARNING =====
        print(f"\033[93m[WARNING] {similarity_pct:.1f}% Similarity with Previous Loss (#{match_idx}). Requesting confirmation...\033[0m")
        print(f"\033[93m[Memory] Current market state is {similarity_pct:.0f}% similar to a previous loss. Proceeding with caution...\033[0m")
        print(f"\033[93m[Stats] Total Trades Avoided by Memory: {trades_avoided_by_memory}\033[0m")
        print(f"\033[96m{'='*55}\033[0m\n")
        return 'WARN'
    
    else:
        # ===== SAFE =====
        print(f"\033[92m[SAFE] {similarity_pct:.1f}% Similarity — Below danger threshold.\033[0m")
        print(f"\033[92m[Memory] Current market state is {similarity_pct:.0f}% similar to closest loss. Proceeding normally.\033[0m")
        print(f"\033[93m[Stats] Total Trades Avoided by Memory: {trades_avoided_by_memory}\033[0m")
        print(f"\033[96m{'='*55}\033[0m\n")
        return 'PASS'

def main():
    global trades_avoided_by_memory
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    executor = TradeExecutor()
    last_trade_minute = -1
    last_fetch_minute = -1
    warm_session = None
    
    stale_count = 0
    last_cycle_close = None
    force_evaluation = False
    
    print("\n" + "="*55)
    print("  ⏱️ DIRECT PLATFORM INTELLIGENCE v2.3 — EXPERT FIXED TIME")
    print("  Data Source: Universal Browser DOM/WebSocket")
    print("  Trade Mode: FIXED TIME | Chain Attacks: ENABLED")
    print("="*55)
    
    while True:
        try:
            now = datetime.datetime.now()
            
            minutes_to_next = 5 - (now.minute % 5)
            seconds_to_next_candle = minutes_to_next * 60 - now.second if minutes_to_next < 5 else 60 - now.second

            # Warm-up phase triggers strictly at roughly T-60s
            if seconds_to_next_candle <= 60 and seconds_to_next_candle > 5 and warm_session is None:
                print(f"\n[WARM-UP] Candle closes in ~{seconds_to_next_candle}s. Pre-opening browser...")
                try:
                    warm_session = executor.warm_up_browser()
                except Exception as e:
                    print(f"[WARM-UP] Failed: {e}. Will cold-start at execution time.")
                    warm_session = None
                    
            # 60s Sampling Rate
            if now.minute != last_fetch_minute:
                last_fetch_minute = now.minute
                
                raw_df = fetch_data()
                if raw_df is None or raw_df.empty:
                    print("\033[91mData fetch failed. Retrying next cycle...\033[0m")
                    time.sleep(5)
                    continue
                    
                processed_df = feature_engineering_pipeline(raw_df)
                
                # Anti-freeze Check
                current_close = processed_df['close'].iloc[-1]
                if last_cycle_close == current_close:
                    stale_count += 1
                else:
                    stale_count = 0
                    
                last_cycle_close = current_close
                
                if stale_count >= 3:
                    print(f"\033[91m[Stale Data] Market is flat or frozen for 3 consecutive cycles. Waiting...\033[0m")
                    time.sleep(5)
                    continue
                    
                print(f"[Loop] Active cache maintained. Current Close: {current_close:.2f} (Stale Count: {stale_count})")
                
                # Trade Evaluation ONLY on 5m boundary OR if a trade just finished (Expert Chaining)
                if (now.minute % 5 == 0 and now.minute != last_trade_minute) or force_evaluation:
                    if force_evaluation:
                        print("\n" + "="*50)
                        print(f"⚡ [EXPERT CHAINING] Immediate Follow-up Evaluation Triggered!")
                        print("="*50)
                    else:
                        print("\n" + "="*50)
                        print(f"[SIGNAL EVALUATION] Trigger Time: {now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
                        print("="*50)
                        
                    force_evaluation = False  # Reset flag so it only fires once per trigger
                    last_trade_minute = now.minute
                    
                    from lstm_model import prepare_sequential_data, train_and_evaluate
                    X_train, X_test, y_train, y_test, scaler, train_weights = prepare_sequential_data(processed_df)
                    model, history, acc = train_and_evaluate(X_train, X_test, y_train, y_test, sample_weights=train_weights)
                    
                    latest_features = processed_df.drop(['Target'], axis=1).values
                    latest_features_scaled = scaler.transform(latest_features)
                    
                    if len(latest_features_scaled) >= Config.SEQUENCE_LENGTH:
                        X_live = latest_features_scaled[-Config.SEQUENCE_LENGTH:]
                        X_live = np.array([X_live])
                        
                        prob = model.predict(X_live)[0][0]
                        
                        # Trust the Brain: Expert entry thresholds
                        action = None
                        raw_action = None
                        if prob > 0.52:
                            raw_action = "buy"
                            action = raw_action
                        elif prob < 0.48:
                            raw_action = "sell"
                            action = raw_action
                        else:
                            print(f"\033[93m[Neutral Zone] Prob {prob*100:.2f}% is between 48% and 52%. Skipping trade.\033[0m")
                        
                        # ===== MEMORY SIMILARITY CHECK =====
                        similarity_pct, match_idx = compute_memory_similarity(processed_df)
                        memory_verdict = print_memory_report(similarity_pct, match_idx)
                        
                        if memory_verdict == 'BLOCK':
                            print("\033[91m[FINAL DECISION] TRADE BLOCKED BY MEMORY. Skipping cycle.\033[0m")
                            if warm_session:
                                try:
                                    warm_session[1].stop()  # Disconnect Playwright only
                                except: pass
                                warm_session = None
                            continue
                            
                        # ===== TREND FILTER (WARNING ONLY) =====
                        if 'BB_mid' in processed_df.columns and action:
                            last_close = processed_df['close'].iloc[-1]
                            bb_mid = processed_df['BB_mid'].iloc[-1]
                            dist_to_mid = last_close - bb_mid
                            
                            print(f"[Signal] Prob: {prob*100:.2f}% -> {action.upper()} | Distance to BB_mid: {dist_to_mid:.4f}")
                            
                            if (action == "buy" and last_close < bb_mid) or (action == "sell" and last_close > bb_mid):
                                print("\033[93m[Warning] Trading AGAINST the Bollinger Band trend.\033[0m")
                                print("\033[96m[Expert Mode] LSTM Confidence is strong. Bypassing Trend Filter for Entry. Relying on Shark Exit.\033[0m")
                        
                        # ===== MEMORY WARNING TIER (60-80%): Auto-proceed with caution =====
                        if memory_verdict == 'WARN' and action:
                            print(f"\033[93m[Memory] {similarity_pct:.0f}% Similarity — Proceeding with caution (auto-mode).\033[0m")
                        
                        
                        if action:
                            # Apply Config values
                            fx_amount = Config.FOREX_DEFAULT_AMOUNT
                            
                            duration = "2 min"  # Standard fixed time duration
                            
                            print(f"\n\033[96m{'='*55}\033[0m")
                            print(f"\033[96m       ⏱️ EXPERT FIXED TIME TRADE\033[0m")
                            print(f"\033[96m{'='*55}\033[0m")
                            print(f"\033[96m  Action      : {action.upper()}\033[0m")
                            print(f"\033[96m  Amount      : {fx_amount}$\033[0m")
                            print(f"\033[96m  Duration    : {duration}\033[0m")
                            print(f"\033[96m{'='*55}\033[0m")
                            
                            # Execute Fixed Time trade
                            success, output_msg = executor.execute_web(
                                action=action,
                                duration=duration,
                                amount=fx_amount,
                                warm_session=warm_session
                            )
                            
                            warm_session = None
                            
                            if success:
                                print(f"\n\033[92m[LIVE EXPERT TRADE VERIFIED: {action.upper()} FIXED TIME ON OLYMP TRADE]\033[0m")
                                print("\033[93m[Wait] Waiting for fixed time expiration (2 mins)...\033[0m")
                                time.sleep(125)  # Wait for 2 minutes + brief buffer
                                print("\033[96m[Completed] Trade expired. Please verify UI for result.\033[0m")
                            else:
                                print(f"\033[91mFailed to open Fixed Time trade. Reason: {output_msg}\033[0m")
                            
                            # Trigger a follow-up chain evaluation
                            print("\n\033[93m[Chain Attack] Activating continuous market scan (bypassing 5m wait)...\033[0m")
                            force_evaluation = True
                            
                        else:
                            print("[FINAL DECISION] NO TRADE")
                            
                        if warm_session:
                            try:
                                warm_session[1].stop()  # Disconnect Playwright only
                            except: pass
                            warm_session = None 
                    else:
                        print("Not enough data to form a sequence.")
                        
            time.sleep(2)
        except Exception as e:
            print(f"\033[91m[MAIN LOOP EXCEPTION] {e}\033[0m")
            import traceback
            traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    main()

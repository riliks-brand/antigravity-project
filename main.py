import numpy as np
import yfinance as yf
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
    
    print("\n" + "="*50)
    print("INITIATING LIVE CONTINUOUS TRADING LOOP")
    print("="*50)
    
    while True:
        try:
            now = datetime.datetime.now()
            
            # Detect OTC mode from Config
            otc_mode = "OTC" in Config.SYMBOL.upper() if hasattr(Config, 'SYMBOL') else False
            weekend_mode = now.weekday() >= 5
            
            if not weekend_mode and not otc_mode:
                if mt5.terminal_info() is None:
                    init_mt5()
                    
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
                
                raw_df = fetch_data(weekend_mode=weekend_mode)
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
                
                # Trade Evaluation ONLY on 5m boundary
                if now.minute % 5 == 0 and now.minute != last_trade_minute:
                    print("\n" + "="*50)
                    print(f"[SIGNAL EVALUATION] Trigger Time: {now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
                    print("="*50)
                    
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
                        raw_action = "buy" if prob > 0.5 else "sell"
                        
                        # ===== MEMORY SIMILARITY CHECK (BEFORE BB Filter) =====
                        similarity_pct, match_idx = compute_memory_similarity(processed_df)
                        memory_verdict = print_memory_report(similarity_pct, match_idx)
                        
                        if memory_verdict == 'BLOCK':
                            # Hard block — skip this entire trade cycle
                            print("\033[91m[FINAL DECISION] TRADE BLOCKED BY MEMORY. Skipping cycle.\033[0m")
                            if warm_session:
                                try:
                                    warm_session[1].stop()
                                    warm_session[0].terminate()
                                except: pass
                                warm_session = None
                            continue
                        
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
                                
                        print(f"Filter: {filter_status}")
                        
                        # ===== MEMORY WARNING TIER (60-80%): Co-Pilot Confirmation =====
                        if memory_verdict == 'WARN' and action:
                            print(f"\n\033[93m[WARNING] {similarity_pct:.0f}% Similarity. Do you still want to force entry? (y/n):\033[0m")
                            force_mem = input("Decision: ")
                            if force_mem.strip().lower() != 'y':
                                trades_avoided_by_memory += 1
                                print(f"\033[91m[Memory] Trade cancelled by operator. Total Avoided: {trades_avoided_by_memory}\033[0m")
                                action = None
                        
                        forced_tier = None
                        if not action:
                            print("\n[!] Filter Blocked.")
                            force = input("Force Trade? (y/n): ")
                            if force.strip().lower() == 'y':
                                action = raw_action
                                if Config.TRADE_MODE == "forex":
                                    print("Choose Amount:")
                                    print("[1] 1$ (x10) | [2] 10$ (x10) | [3] Custom")
                                    tier_choice = input("Choice: ")
                                    if tier_choice == '1': 
                                        forced_tier = {"amount": "1", "multiplier": Config.FOREX_MULTIPLIER}
                                    elif tier_choice == '2': 
                                        forced_tier = {"amount": "10", "multiplier": Config.FOREX_MULTIPLIER}
                                    elif tier_choice == '3':
                                        cust_amt = input("Amount ($): ")
                                        cust_mult = input(f"Multiplier (default {Config.FOREX_MULTIPLIER}): ") or Config.FOREX_MULTIPLIER
                                        forced_tier = {"amount": cust_amt, "multiplier": cust_mult}
                                else:
                                    print("Choose Tier:")
                                    print("[1] 1$ (1m) | [2] 10$ (2m) | [3] Custom")
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
                            # ===== FOREX MODE =====
                            if Config.TRADE_MODE == "forex":
                                # Determine amount and multiplier
                                if forced_tier and isinstance(forced_tier, dict):
                                    fx_amount = forced_tier["amount"]
                                    fx_multiplier = forced_tier["multiplier"]
                                else:
                                    fx_amount = Config.FOREX_DEFAULT_AMOUNT
                                    fx_multiplier = Config.FOREX_MULTIPLIER
                                
                                # Calculate ATR-based TP and SL
                                current_atr = processed_df['ATR'].iloc[-1]
                                entry_price = processed_df['close'].iloc[-1]
                                
                                if action == "buy":
                                    tp_price = entry_price + (current_atr * Config.FOREX_TP_ATR_MULT)
                                    sl_price = entry_price - (current_atr * Config.FOREX_SL_ATR_MULT)
                                else:
                                    tp_price = entry_price - (current_atr * Config.FOREX_TP_ATR_MULT)
                                    sl_price = entry_price + (current_atr * Config.FOREX_SL_ATR_MULT)
                                
                                print(f"\n\033[96m{'='*55}\033[0m")
                                print(f"\033[96m       💹 FOREX TRADE PARAMETERS\033[0m")
                                print(f"\033[96m{'='*55}\033[0m")
                                print(f"\033[96m  Action      : {action.upper()}\033[0m")
                                print(f"\033[96m  Amount      : {fx_amount}$\033[0m")
                                print(f"\033[96m  Multiplier  : x{fx_multiplier}\033[0m")
                                print(f"\033[96m  Entry Price : {entry_price:.5f}\033[0m")
                                print(f"\033[96m  ATR         : {current_atr:.5f}\033[0m")
                                print(f"\033[92m  Take Profit : {tp_price:.5f} (ATR × {Config.FOREX_TP_ATR_MULT})\033[0m")
                                print(f"\033[91m  Stop Loss   : {sl_price:.5f} (ATR × {Config.FOREX_SL_ATR_MULT})\033[0m")
                                print(f"\033[96m{'='*55}\033[0m")
                                
                                # Execute Forex trade
                                success, output_msg, outcome = executor.execute_forex(
                                    action=action,
                                    amount=fx_amount,
                                    multiplier=fx_multiplier,
                                    tp_price=tp_price,
                                    sl_price=sl_price,
                                    warm_session=warm_session
                                )
                                
                                warm_session = None
                                
                                if success:
                                    print(f"\n[LIVE TRADE VERIFIED: {action.upper()} FOREX ON OLYMP TRADE]")
                                    
                                    # Forex loss = SL hit
                                    if outcome.get('hit') == 'sl':
                                        print("\033[91m[Outcome] STOP LOSS HIT — LOSS. Logging market state for future avoidance.\033[0m")
                                        log_loss(processed_df, weekend_mode)
                                    elif outcome.get('hit') == 'tp':
                                        print("\033[92m[Outcome] TAKE PROFIT HIT — WIN! 💰\033[0m")
                                    elif outcome.get('hit') == 'timeout':
                                        print("\033[93m[Outcome] TIMEOUT — Trade may still be active. Check platform.\033[0m")
                                    else:
                                        print(f"\033[93m[Outcome] Unknown result: {outcome}\033[0m")
                                else:
                                    print(f"Failed to open Forex trade on Olymp Trade. Reason: {output_msg}")
                            
                            # ===== FIXED TIME MODE (Original) =====
                            else:
                                if forced_tier and isinstance(forced_tier, tuple):
                                    duration, amount, dur_mins = forced_tier
                                else:
                                    if prob > 0.55 or prob < 0.45:
                                        duration, amount, dur_mins = ("2 min", "10", 2)
                                    else:
                                        duration, amount, dur_mins = ("1 min", "1", 1)
                                        
                                print(f"[FINAL DECISION] {action.upper()} | Amount: {amount}$ | Duration: {duration}")
                                
                                entry_price = 0
                                if weekend_mode:
                                    tic = yf.Ticker("BTC-USD")
                                    try:
                                        entry_price = tic.fast_info.last_price
                                    except:
                                        entry_price = current_close
                                    print(f"[Yfinance Sync] Entry Price: {entry_price:.2f}")
                                elif otc_mode:
                                    from otc_scraper import OTCScraper
                                    entry_price = OTCScraper.get_last_price()
                                    print(f"[OTC Scraper] Entry Price: {entry_price:.5f}")
                                else:
                                    init_mt5()
                                    entry_tick = mt5.symbol_info_tick(Config.SYMBOL)
                                    if entry_tick:
                                        entry_price = entry_tick.ask if action == "buy" else entry_tick.bid
                                        print(f"[MT5 Sync] Entry Price: {entry_price:.5f}")
                                
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
                                        print(f"[Verify] Waiting {dur_mins} minutes for trade completion...")
                                        time.sleep(dur_mins * 60)
                                        
                                        exit_price = entry_price
                                        if weekend_mode:
                                            try:
                                                tic = yf.Ticker("BTC-USD")
                                                exit_price = tic.fast_info.last_price
                                                print(f"[Yfinance Sync] Expiry Price: {exit_price:.2f} (Entry: {entry_price:.2f})")
                                            except: pass
                                        elif otc_mode:
                                            from otc_scraper import OTCScraper
                                            exit_price = OTCScraper.get_last_price()
                                            print(f"[OTC Scraper] Expiry Price: {exit_price:.5f} (Entry: {entry_price:.5f})")
                                        else:
                                            init_mt5()
                                            exit_tick = mt5.symbol_info_tick(Config.SYMBOL)
                                            if exit_tick:
                                                exit_price = exit_tick.ask if action == "buy" else exit_tick.bid
                                                print(f"[MT5 Sync] Expiry Price: {exit_price:.5f} (Entry: {entry_price:.5f})")
                                                
                                        loss = False
                                        if action == "buy" and exit_price <= entry_price:
                                            loss = True
                                        elif action == "sell" and exit_price >= entry_price:
                                            loss = True
                                            
                                        if loss:
                                            print("\033[91m[Outcome] LOSS detected. Triggering Loss Log.\033[0m")
                                            log_loss(processed_df, weekend_mode)
                                        else:
                                            print("\033[92m[Outcome] WIN detected.\033[0m")
                                else:
                                    print(f"Failed to open trade on Olymp Trade. Reason: {output_msg}")
                        else:
                            print("[FINAL DECISION] NO TRADE")
                            
                        if warm_session:
                            try:
                                warm_session[1].stop()
                                warm_session[0].terminate()
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

"""
Main Loop — Elite v3.1 (Ensemble Edition)
============================================
The master orchestrator that ties ALL systems together.

Architecture:
  Data (MT5) → Features → [LSTM + RF] → Ensemble Voting → Hybrid Filters
  → Trade Manager → Execute → Notify (Telegram)

New in v3.1:
- Ensemble Engine (Dynamic Weighted Soft Voting)
- Conflict Detection & Disagreement Penalty
- RF De-correlated Features
- Telegram Notifications (Spam-Controlled)
- Comprehensive Ensemble Decision Logging
"""

import sys
import os

# Fix Windows terminal Unicode encoding (cp1252 → utf-8)
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        # Python 3.6 fallback
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import time
import datetime
import logging
from config import Config

# ===== Setup Master Logger =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("Main")


def compute_memory_similarity(processed_df):
    """
    Computes similarity as a PROBABILITY MODIFIER (not a hard block).
    Returns (bias_adjustment, similarity_pct, match_idx).
    """
    log_file = Config.TRADING_HISTORY_FILE
    if not os.path.exists(log_file):
        return 0.0, 0.0, -1

    try:
        losses_df = pd.read_csv(log_file)
        losses_df = losses_df[losses_df.get('pnl', pd.Series(dtype=float)) < 0]
        if losses_df.empty:
            return 0.0, 0.0, -1
    except Exception:
        return 0.0, 0.0, -1

    last_row = processed_df.iloc[-1]

    current_state = np.array([
        last_row.get('RSI', 50),
        last_row.get('ATR', 0),
        last_row.get('Volatility', 0),
        last_row.get('ADX', 25),
        last_row.get('BB_position', 0.5),
    ], dtype=float)

    max_sim = 0.0
    match_idx = -1

    for idx, loss_row in losses_df.iterrows():
        loss_state = np.array([
            loss_row.get('RSI', 50) if 'RSI' in losses_df.columns else 50,
            loss_row.get('ATR', 0) if 'ATR' in losses_df.columns else 0,
            loss_row.get('Volatility', 0) if 'Volatility' in losses_df.columns else 0,
            loss_row.get('ADX', 25) if 'ADX' in losses_df.columns else 25,
            loss_row.get('BB_position', 0.5) if 'BB_position' in losses_df.columns else 0.5,
        ], dtype=float)

        norm_c = np.linalg.norm(current_state)
        norm_l = np.linalg.norm(loss_state)

        if norm_c == 0 and norm_l == 0:
            similarity = 100.0
        elif norm_c == 0 or norm_l == 0:
            similarity = 0.0
        else:
            cos_sim = np.dot(current_state, loss_state) / (norm_c * norm_l)
            cos_sim = max(0.0, min(1.0, cos_sim))
            similarity = cos_sim * 100.0

        if similarity > max_sim:
            max_sim = similarity
            match_idx = idx

    if max_sim >= Config.MEMORY_SIMILARITY_THRESHOLD:
        scale = (max_sim - Config.MEMORY_SIMILARITY_THRESHOLD) / (100 - Config.MEMORY_SIMILARITY_THRESHOLD)
        bias = -scale * Config.MEMORY_BIAS_SCALE
        return bias, max_sim, match_idx

    return 0.0, max_sim, match_idx


def apply_hybrid_filters(processed_df, direction, symbol, server_time=None):
    """
    The Hybrid Filter Layer — rejects bad signals before they reach execution.
    Returns (passed: bool, reason: str)
    """
    last = processed_df.iloc[-1]
    reasons = []

    # 1. ADX Range Filter
    adx_val = last.get('ADX', 30)
    if adx_val < Config.ADX_RANGING_THRESHOLD:
        reasons.append(f"RANGING: ADX={adx_val:.1f} < {Config.ADX_RANGING_THRESHOLD}")

    # 2. Trend Alignment (H1)
    h1_trend = last.get('H1_trend', 0)
    #if h1_trend != 0:
    #    if direction == "BUY" and h1_trend == -1:
    #        reasons.append("COUNTER-TREND: BUY against H1 downtrend")
    #    elif direction == "SELL" and h1_trend == 1:
    #        reasons.append("COUNTER-TREND: SELL against H1 uptrend")

    # 3. Low Volatility Filter
    volatility = last.get('Volatility', 0)
    if volatility < Config.ATR_THRESHOLD:
        reasons.append(f"LOW VOLATILITY: {volatility:.6f} < {Config.ATR_THRESHOLD}")

    # 4. News Filter
    from news_filter import is_news_window
    news_blocked, news_reason = is_news_window(symbol, server_time)
    if news_blocked:
        reasons.append(news_reason)

    # 5. Session Filter
    from trade_manager import TradeManager
    in_session, session_reason = TradeManager.is_in_trading_session(symbol)
    if not in_session:
        reasons.append(f"SESSION: {session_reason}")

    # 6. Spread Filter
    spread_ok, spread_pts = TradeManager.check_spread(symbol)
    if not spread_ok and spread_pts > 0:
        reasons.append(f"SPREAD: {spread_pts:.1f} > {Config.MAX_SPREAD_POINTS}")

    if reasons:
        return False, " | ".join(reasons)
    return True, "All filters passed"


def main():
    print("\n" + "=" * 65)
    print("  🚀 ELITE TRADING BOT v3.1 — MULTI-SYMBOL PORTFOLIO")
    print("=" * 65)
    print("  📊 Data Source   : MetaTrader 5 (Native)")
    print("  🧠 Intelligence : LSTM + Random Forest (Voting)")
    print("  ⚙️  Execution     : Portfolio Manager & Ranker")
    print("  🛡️  Risk Engine   : Equity Curve + Kill Switch + Drawdown Survival")
    print("  📰 News Filter   : ForexFactory (High Impact)")
    print("  📲 Alerts        : Telegram (Spam-Controlled)")
    print(f"  💹 Symbols       : {', '.join(Config.SYMBOLS)}")
    print(f"  🔴 Daily Max Loss: {Config.MAX_DAILY_LOSS_PCT}%")
    print(f"  🧬 Global Min Thr: {Config.MIN_GLOBAL_SCORE}")
    if getattr(Config, 'MICRO_ACCOUNT_MODE', False):
        print(f"  💰 MICRO MODE    : ON (Balance < ${Config.MICRO_BALANCE_THRESHOLD})")
        print(f"     └─ Lot: MIN | SL: {Config.MICRO_SL_ATR_MULT}x ATR | Max Trades: {Config.MICRO_MAX_CONCURRENT_TRADES}")
    print("=" * 65)

    # ===== PHASE 1: Connect to MT5 =====
    from mt5_engine import (
        connect_to_exness, heartbeat, execute_forex_trade,
        check_kill_switch, get_account_balance, get_account_equity,
        get_server_time, close_all_positions,
    )

    if not connect_to_exness():
        logger.critical("[Fatal] Could not establish MT5 connection. Exiting.")
        return

    # ===== PHASE 2: Initialize Trade Manager =====
    from trade_manager import TradeManager
    manager = TradeManager()
    manager.reset_daily_stats(get_account_balance())

    # ===== PHASE 3: Initialize Notifier =====
    from notifier import get_notifier
    notifier = get_notifier()

    # ===== PHASE 4: Import modules =====
    from data_loader import fetch_mtf_data, fetch_tick_data, is_market_open
    from features import feature_engineering_pipeline
    import joblib
    import os
    
    # Optional GPU suppression if needed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow.keras.models import load_model

    # ===== PHASE 5: Initialize RF & LSTM Models =====
    from rf_model import RFModel
    from ensemble_engine import ensemble_predict

    rf_model = RFModel()
    
    # Load Offline LSTM Model
    lstm_model_path = "lstm_model.h5"
    lstm_scaler_path = "lstm_scaler.joblib"
    
    if os.path.exists(lstm_model_path) and os.path.exists(lstm_scaler_path):
        logger.info("Loading offline LSTM model and scaler...")
        lstm_model = load_model(lstm_model_path)
        lstm_scaler = joblib.load(lstm_scaler_path)
    else:
        logger.critical("[Fatal] Offline LSTM models not found. Please run train_offline.py first.")
        return

    # State variables
    last_eval_candle = -1
    last_heartbeat = time.time()
    candle_index = 0
    last_daily_summary_date = None
    last_eval_time = 0
    symbol_states = {}

    print("\n\033[92m[STARTUP] ✅ All systems online. Entering main loop.\033[0m\n")

    while True:
        try:
            now = datetime.datetime.utcnow()

            # ===== HEARTBEAT =====
            if time.time() - last_heartbeat > Config.HEARTBEAT_INTERVAL_SECONDS:
                if not heartbeat():
                    logger.error("[Heartbeat] MT5 reconnection failed. Waiting 30s...")
                    notifier.connection_lost()
                    time.sleep(30)
                    continue
                last_heartbeat = time.time()

            # ===== KILL SWITCH =====
            if check_kill_switch():
                logger.critical("[KILL SWITCH] Trading halted. Closing all positions...")
                close_all_positions()

                # Notify via Telegram
                account_balance = get_account_balance()
                account_equity = get_account_equity()
                daily_loss = ((account_equity - account_balance) / account_balance * 100) if account_balance > 0 else 0
                notifier.kill_switch_activated(abs(daily_loss), account_balance)

                print("\n\033[91m⛔ KILL SWITCH ACTIVATED — Daily loss limit exceeded.\033[0m")
                print("\033[91m⛔ All positions closed. Bot paused until next day.\033[0m\n")

                tomorrow = (now + datetime.timedelta(days=1)).replace(hour=0, minute=5, second=0)
                wait_seconds = (tomorrow - now).total_seconds()
                time.sleep(max(wait_seconds, 60))
                manager.reset_daily_stats(get_account_balance())
                continue

            # ===== DAILY RESET & SUMMARY =====
            today = now.date()
            if now.hour == 0 and now.minute < 2:
                manager.reset_daily_stats(get_account_balance())

            # Daily Summary (send at configured hour)
            if now.hour == Config.TELEGRAM_DAILY_SUMMARY_HOUR and last_daily_summary_date != today:
                stats = manager.get_stats()
                top_features = rf_model.get_top_features(5) if rf_model.model else None
                notifier.daily_summary(stats, top_features)
                last_daily_summary_date = today

            # ===== TICK MANAGEMENT (every cycle) =====
            for symbol in Config.SYMBOLS:
                if not is_market_open(symbol):
                    continue
                if symbol == "BTCUSD" and not Config.TRADE_CRYPTO_WEEKENDS and now.weekday() >= 5:
                    continue

                tick_data = fetch_tick_data(symbol)
                if tick_data and len(manager.active_trades) > 0:
                    current_atr = 0.001
                    try:
                        from data_loader import fetch_mt5_ohlc
                        quick_df = fetch_mt5_ohlc(symbol, Config.TIMEFRAME, 20)
                        if quick_df is not None and 'high' in quick_df.columns:
                            from ta.volatility import AverageTrueRange
                            atr_series = AverageTrueRange(
                                high=quick_df['high'], low=quick_df['low'],
                                close=quick_df['close'], window=14
                            ).average_true_range()
                            current_atr = atr_series.iloc[-1] if len(atr_series) > 0 else 0.001
                    except Exception:
                        pass

                    manager.on_tick(symbol, tick_data['bid'], tick_data['ask'], current_atr)

            # ===== 10-SECOND HYBRID LOOP =====
            if time.time() - last_eval_time < 10:
                notifier.flush_queue()
                time.sleep(1)
                continue
            
            last_eval_time = time.time()
            server_minute = now.minute

            is_candle_close = (server_minute % 5 == 0) and (server_minute != last_eval_candle)

            if is_candle_close:
                last_eval_candle = server_minute
                candle_index += 1
                logger.info("\n" + "=" * 65)
                logger.info("[PORTFOLIO EVALUATION] Candle #%d | Loop Trigger Time: %s",
                            candle_index, now.strftime('%Y-%m-%d %H:%M:%S'))
                logger.info("=" * 65)

            opportunities = []

            for symbol in Config.SYMBOLS:
                if not is_market_open(symbol):
                    continue
                if symbol == "BTCUSD" and not Config.TRADE_CRYPTO_WEEKENDS and now.weekday() >= 5:
                    continue

                tick_data_eval = fetch_tick_data(symbol)
                current_price = tick_data_eval['bid'] if tick_data_eval else None
                if not current_price: continue
                
                state = symbol_states.get(symbol, {})
                force_eval = is_candle_close
                
                if not force_eval and state.get("last_close"):
                    if abs(current_price - state["last_close"]) > 0.5 * state.get("atr", 0.001):
                        force_eval = True
                    elif state.get("ob_bull") and abs(current_price - state["ob_bull"]) < 0.5 * state.get("atr", 0.001):
                        force_eval = True
                    elif state.get("ob_bear") and abs(current_price - state["ob_bear"]) < 0.5 * state.get("atr", 0.001):
                        force_eval = True
                    elif state.get("fvg") and abs(current_price - state["fvg"]) < 0.2 * state.get("atr", 0.001):
                        force_eval = True
                    elif state.get("past_max") and current_price > state["past_max"]:
                        force_eval = True
                    elif state.get("past_min") and current_price < state["past_min"]:
                        force_eval = True
                        
                if not force_eval:
                    continue

                # ===== FETCH MULTI-TIMEFRAME DATA =====
                mtf_data = fetch_mtf_data(symbol)
                if mtf_data is None: continue
                
                df_m5 = mtf_data.get("M5")
                df_m15 = mtf_data.get("M15", pd.DataFrame())
                df_h1 = mtf_data.get("H1", pd.DataFrame())
                
                if df_m5 is None or df_m5.empty: continue

                # ===== FEATURE ENGINEERING =====
                processed_df = feature_engineering_pipeline(
                    df_m5, df_confirm=df_m15 if not df_m15.empty else None, df_trend=df_h1 if not df_h1.empty else None
                )
                if processed_df is None or processed_df.empty: continue

                # Update State
                symbol_states[symbol] = {
                    "last_close": current_price,
                    "atr": processed_df['ATR'].iloc[-1] if not pd.isna(processed_df['ATR'].iloc[-1]) else 0.001,
                    "ob_bull": processed_df['active_bullish_ob'].iloc[-1] if 'active_bullish_ob' in processed_df.columns and not pd.isna(processed_df['active_bullish_ob'].iloc[-1]) else None,
                    "ob_bear": processed_df['active_bearish_ob'].iloc[-1] if 'active_bearish_ob' in processed_df.columns and not pd.isna(processed_df['active_bearish_ob'].iloc[-1]) else None,
                    "fvg": processed_df['last_fvg_price'].iloc[-1] if 'last_fvg_price' in processed_df.columns and not pd.isna(processed_df['last_fvg_price'].iloc[-1]) else None,
                    "past_max": processed_df['high'].iloc[-10:].max() if len(processed_df) >= 10 else None,
                    "past_min": processed_df['low'].iloc[-10:].min() if len(processed_df) >= 10 else None,
                }

                # ===== PHASE 2: SMART EXIT CHECK (before entry evaluation) =====
                if len(manager.active_trades) > 0:
                    try:
                        manager.evaluate_smart_exits(symbol, processed_df)
                    except Exception as e:
                        logger.error("[SmartExit] Evaluation error for %s: %s", symbol, e)

                # ===== LSTM PREDICTION =====
                latest_features = processed_df.drop(['Target'], axis=1, errors='ignore').values
                latest_features_scaled = lstm_scaler.transform(latest_features)

                if len(latest_features_scaled) < Config.SEQUENCE_LENGTH: 
                    continue
                
                X_live = np.array([latest_features_scaled[-Config.SEQUENCE_LENGTH:]])
                lstm_prob = float(lstm_model.predict(X_live, verbose=0)[0][0])

                # ===== RANDOM FOREST PREDICTION =====
                rf_prob = 0.5
                if Config.ENSEMBLE_ENABLED:
                    rf_prob = rf_model.predict_proba(processed_df)

                current_atr = processed_df['ATR'].iloc[-1] if not pd.isna(processed_df['ATR'].iloc[-1]) else 0.001
                # Use H1_ADX for macro trend strength instead of noisy M5 ADX
                h1_adx_val = processed_df['H1_ADX'].iloc[-1] if 'H1_ADX' in processed_df.columns else 25.0
                current_adx = h1_adx_val if not pd.isna(h1_adx_val) else 25.0
                atr_series = processed_df['ATR'].dropna()

                # ===== DYNAMIC CONFIDENCE BOOST & MTF LOGIC (Phase 3) =====
                event_boost = 0.0
                event_strength = 1.0
                
                inside_ob = processed_df['inside_ob_zone'].iloc[-1] if 'inside_ob_zone' in processed_df.columns else 0
                ob_strength = processed_df['ob_strength'].iloc[-1] if 'ob_strength' in processed_df.columns else 0.0
                fvg_filled = processed_df['fvg_filled'].iloc[-1] if 'fvg_filled' in processed_df.columns else 0
                fvg_size = processed_df['fvg_size'].iloc[-1] if 'fvg_size' in processed_df.columns else 0.0
                liq_sweep = processed_df['liquidity_sweep_flag'].iloc[-1] if 'liquidity_sweep_flag' in processed_df.columns else 0
                
                if inside_ob == 1:
                    event_strength += ob_strength
                    event_boost += 0.02 * min(1.0, ob_strength)
                if fvg_filled == 1:
                    event_strength += fvg_size
                    event_boost += 0.01
                if liq_sweep != 0:
                    event_strength += 0.5
                    event_boost += 0.02
                    
                h1_trend = int(processed_df['H1_trend'].iloc[-1]) if 'H1_trend' in processed_df.columns and not pd.isna(processed_df['H1_trend'].iloc[-1]) else 0

                # ===== SESSION DETECTION (v4.0) =====
                session = TradeManager.get_active_session(symbol)

                # ===== ENSEMBLE PREDICTION (Session-Aware) =====
                decision_original = ensemble_predict(
                    lstm_prob=lstm_prob, rf_prob=rf_prob, current_adx=current_adx,
                    current_atr=current_atr, atr_series=atr_series, session=session,
                    diagnostic=False, event_boost=event_boost, h1_trend=h1_trend
                )
                
                decision_diagnostic = ensemble_predict(
                    lstm_prob=lstm_prob, rf_prob=rf_prob, current_adx=current_adx,
                    current_atr=current_atr, atr_series=atr_series, session=session,
                    diagnostic=True, event_boost=event_boost, h1_trend=h1_trend
                )
                
                # --- Dual Evaluation Logging ---
                if decision_original.direction is None and decision_diagnostic.direction is not None:
                    logger.info("[%s] 📊 [DUAL] THRESHOLD_BLOCK: Original=HOLD, Diagnostic=%s", symbol, decision_diagnostic.direction)
                elif decision_original.direction is None and decision_diagnostic.direction is None:
                    logger.info("[%s] 📊 [DUAL] MODEL_LIMITATION: Original=HOLD, Diagnostic=HOLD", symbol)
                elif decision_original.direction is not None:
                    logger.info("[%s] 📊 [DUAL] NATIVE_EXECUTION: Original=%s", symbol, decision_original.direction)

                # Use diagnostic decision to unblock execution if DIAGNOSTIC_MODE is enabled
                decision = decision_diagnostic if getattr(Config, "DIAGNOSTIC_MODE", False) else decision_original

                direction = decision.direction
                base_prob = decision.final_prob
                trend_strength = decision.trend_strength

                # ===== REGIME PERSISTENCE (v4.0) =====
                current_regime, regime_changed = manager.update_regime(trend_strength)

                # ===== SAFETY RULE: if direction is None → DO NOT EXECUTE =====
                if direction is None:
                    logger.info(
                        "[%s] SKIP: direction=None | reason=%s | confidence=%s | session=%s",
                        symbol, decision.decision_reason, decision.confidence_level, session
                    )
                    continue
                    
                # ===== PORTFOLIO MACROS: CONTEXT BOOSTS & MEMORY =====
                # 1. Symbol Memory Bias
                memory_bias_local, sim_pct, sim_idx = compute_memory_similarity(processed_df)
                sym_perf_mod = manager.get_symbol_performance_modifier(symbol)
                
                # 2. Context Boosts
                context_boost = 0.0
                h1_trend = processed_df['H1_trend'].iloc[-1] if 'H1_trend' in processed_df.columns else 0
                if (direction == "BUY" and h1_trend == 1) or (direction == "SELL" and h1_trend == -1):
                    context_boost += Config.BOOST_STRONG_TREND
                    
                volatility = processed_df['Volatility'].iloc[-1] if 'Volatility' in processed_df.columns else 0
                if volatility > (Config.ATR_THRESHOLD * 1.5):
                    context_boost += Config.BOOST_HIGH_VOLATILITY
                    
                final_rank_score = base_prob + memory_bias_local + sym_perf_mod + context_boost
                final_rank_score = np.clip(final_rank_score, 0.0, 1.0)
                
                logger.info(
                    "[%s] Base: %.3f | Context: %+.3f | MemBias: %+.3f | SymPerf: %+.3f -> RANK: %.3f | Session: %s | Confidence: %s", 
                    symbol, base_prob, context_boost, memory_bias_local, sym_perf_mod,
                    final_rank_score, session, decision.confidence_level
                )
                
                # Multiply score with trend and event strength for opportunity ranking
                final_rank_score = final_rank_score * max(0.1, trend_strength) * max(1.0, event_strength)

                # Minimum score threshold check
                if final_rank_score < Config.MIN_GLOBAL_SCORE:
                    logger.info("[%s] Dropped: Rank %.3f < %.2f", symbol, final_rank_score, Config.MIN_GLOBAL_SCORE)
                    continue

                # Hybrid local filters
                # We use now as the server_time for news filters since UTC minute matches market minute
                filter_passed, filter_reason = apply_hybrid_filters(processed_df, direction, symbol, now)
                if not filter_passed:
                    logger.info("[%s] Rejected Locally: %s", symbol, filter_reason)
                    continue

                opportunities.append({
                    "symbol": symbol,
                    "direction": direction,
                    "rank_score": final_rank_score,
                    "lstm_prob": lstm_prob,
                    "rf_prob": rf_prob,
                    "penalty": decision.penalty,
                    "current_atr": current_atr,
                    "trend_strength": trend_strength,
                    "confidence_level": decision.confidence_level,
                    "session": session,
                    "regime_changed": regime_changed,
                    "decision_reason": decision.decision_reason,
                })

            # === RANK & EXECUTE PORTFOLIO ===
            opportunities.sort(key=lambda x: x['rank_score'], reverse=True)
            
            if not opportunities:
                logger.info("No viable opportunities this cycle across portfolio.")
                continue
                
            logger.info("--- PORTFOLIO RANKING: Top %d Opportunities ---", len(opportunities))
            for i, opp in enumerate(opportunities):
                logger.info(" #%d %s %s | Score: %.3f", i+1, opp['direction'], opp['symbol'], opp['rank_score'])
                
            executed_this_cycle = 0
            
            for opp in opportunities:
                # Max new trades per cycle limit (prevent spam)
                if executed_this_cycle >= 2:
                    break
                    
                sym = opp["symbol"]
                dir_ = opp["direction"]
                score = opp["rank_score"]
                atr = opp["current_atr"]
                is_near_miss = opp.get("decision_reason") == "NEAR_MISS_ACTIVATION"
                
                # 1. Global Trade Guard
                can_trade, guard_reason = manager.can_trade(sym, dir_, candle_index, is_near_miss=is_near_miss)
                if not can_trade:
                    logger.warning("[GUARD PREVENTED %s]: %s", sym, guard_reason)
                    continue
                    
                # 2. Correlation Filter Check
                corr_passed, corr_reason = manager.check_correlation(sym, dir_)
                if not corr_passed:
                    logger.warning("[CORRELATION REJECTED %s]: %s", sym, corr_reason)
                    continue
                    
                # 3. USD Diversification Check
                if "USD" in sym:
                    if manager.get_usd_exposure() >= Config.MAX_USD_EXPOSURE:
                        logger.warning("[DIVERSIFICATION %s]: Max USD exposure reached (%d)", sym, Config.MAX_USD_EXPOSURE)
                        continue
                
                # 4. Risk / Position Sizing (Adaptive v4.0)
                assigned_risk = manager.get_adaptive_risk(
                    session=opp.get("session", "UNKNOWN"),
                    trend_strength=opp.get("trend_strength", 0.0),
                    regime_changed=opp.get("regime_changed", False),
                    confidence_level=opp.get("confidence_level", "MEDIUM"),
                )
                
                if is_near_miss:
                    assigned_risk *= 0.5
                    logger.info("[NEAR_MISS RISK REDUCTION] %s risk halved to %.2f%%", sym, assigned_risk)
                
                # Drawdown Survival Mode
                current_dd = manager.get_current_drawdown(get_account_balance())
                if current_dd > Config.DRAWDOWN_SURVIVAL_THRESHOLD:
                    assigned_risk *= Config.SURVIVAL_RISK_MODIFIER
                    logger.warning("[SURVIVAL MODE] %s Risk reduced to %.2f%% (Current DD: %.1f%%)", sym, assigned_risk, current_dd)
                    
                # Portfolio Global Cap
                current_risk = manager.get_current_total_risk()
                if current_risk + assigned_risk > Config.MAX_GLOBAL_RISK_PCT:
                    logger.warning("[RISK CAP] Attempting %s at %.2f%%, but Portfolio Risk is %.2f%%/%.2f%%. Trade Skipped.", 
                                   sym, assigned_risk, current_risk, Config.MAX_GLOBAL_RISK_PCT)
                    continue
                    
                # Passed all checks -> Execute
                logger.info("[PORTFOLIO TRIGGER] Executing top pick: %s %s (Risk: %.2f%%, Score: %.3f)", 
                            dir_, sym, assigned_risk, score)
                            
                import MetaTrader5 as mt5
                info = mt5.symbol_info(sym)
                point = info.point if info else 0.00001
                
                # ===== MICRO ACCOUNT MODE: Dynamic SL/TP/Lot =====
                current_balance = get_account_balance()
                is_micro = (getattr(Config, 'MICRO_ACCOUNT_MODE', False) and 
                            current_balance < getattr(Config, 'MICRO_BALANCE_THRESHOLD', 100.0))
                
                if is_micro:
                    sl_mult = getattr(Config, 'MICRO_SL_ATR_MULT', 1.0)
                    tp1_mult = getattr(Config, 'MICRO_TP1_ATR_MULT', 1.5)
                    tp2_mult = getattr(Config, 'MICRO_TP2_ATR_MULT', 2.5)
                    logger.info("[MICRO MODE] Balance: $%.2f | SL: %.1f*ATR | TP1: %.1f*ATR | TP2: %.1f*ATR",
                                current_balance, sl_mult, tp1_mult, tp2_mult)
                else:
                    sl_mult = Config.SL_ATR_MULT
                    tp1_mult = Config.TP1_ATR_MULT
                    tp2_mult = Config.TP2_ATR_MULT
                
                sl_points = int((atr * sl_mult) / point)
                tp1_points = int((atr * tp1_mult) / point)
                tp2_points = int((atr * tp2_mult) / point)
                
                # Apply base equity reduction
                equity_risk_mult = manager.get_risk_multiplier(get_account_equity())
                final_risk_percent = assigned_risk * equity_risk_mult
                
                signal_time_ms = time.time() * 1000
                
                result = execute_forex_trade(
                    action=dir_,
                    symbol=sym,
                    sl_points=sl_points,
                    tp_points=tp2_points,  # Native MT5 TP set to TP2 so partial close at TP1 can occur
                    risk_multiplier=final_risk_percent,
                    signal_time_ms=signal_time_ms,
                )
                
                if result and result.get("success"):
                    tp2_price = result["filled_price"] + (tp2_points * point) if dir_ == "BUY" else result["filled_price"] - (tp2_points * point)
                    
                    manager.register_trade(
                        ticket=result["ticket"], symbol=sym, direction=dir_, volume=result["volume"],
                        entry_price=result["filled_price"], expected_price=result["expected_price"],
                        sl_price=result["sl_price"], tp1_price=result["tp_price"], tp2_price=tp2_price,
                        signal_time_ms=signal_time_ms, fill_time_ms=result["fill_time_ms"],
                        risk_pct=final_risk_percent, volatility=atr, is_near_miss=is_near_miss
                    )
                    manager.update_signal_tracker(sym, dir_, candle_index)
                    executed_this_cycle += 1
                    
                    # Print Status
                    manager.print_status()
                    
                    # Telegram Notification
                    notifier.trade_opened(
                        direction=dir_, symbol=sym,
                        lstm_prob=opp["lstm_prob"], rf_prob=opp["rf_prob"], final_prob=score,
                        entry_price=result["filled_price"], sl_price=result["sl_price"],
                        tp_price=result["tp_price"], lot_size=result["volume"],
                        penalty=opp["penalty"]
                    )
                else:
                    logger.error("[EXECUTION FAILED] %s %s.", dir_, sym)

        except KeyboardInterrupt:
            print("\n\033[93m[EXIT] Bot stopped by user. Saving state...\033[0m")
            manager._save_state()
            notifier.flush_queue()
            print("\033[93m[EXIT] State saved. Active trades are still managed by MT5.\033[0m")
            break

        except Exception as e:
            logger.error("[MAIN LOOP EXCEPTION] %s", e, exc_info=True)
            time.sleep(5)


if __name__ == "__main__":
    main()

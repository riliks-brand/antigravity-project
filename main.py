"""
Main Loop — Elite v3.0
========================
The master orchestrator that ties all systems together.

Architecture:
  Data (MT5) → Features → LSTM → Hybrid Filters → Trade Manager → Execute

Features:
- Multi-Timeframe analysis
- Hybrid Filter Layer (Trend, Volatility, Range, Session, News, Spread)
- Adaptive probability thresholds
- Memory as probability modifier
- Candle-close confirmation
- Heartbeat monitor
- Comprehensive logging & monitoring
"""

import numpy as np
import pandas as pd
import time
import datetime
import os
import csv
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
        # Only consider losses
        losses_df = losses_df[losses_df.get('pnl', pd.Series(dtype=float)) < 0]
        if losses_df.empty:
            return 0.0, 0.0, -1
    except Exception:
        return 0.0, 0.0, -1

    last_row = processed_df.iloc[-1]

    # Current state vector
    current_state = np.array([
        last_row.get('RSI', 50),
        last_row.get('ATR', 0),
        last_row.get('Volatility', 0),
        last_row.get('ADX', 25),
        last_row.get('BB_position', 0.5),
    ], dtype=float)

    max_sim = 0.0
    match_idx = -1

    # Build loss state vectors from features that were present at entry
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

    # Convert similarity to probability bias
    # Only apply if above threshold
    if max_sim >= Config.MEMORY_SIMILARITY_THRESHOLD:
        # Scale: 60% sim → small bias, 100% sim → full bias
        scale = (max_sim - Config.MEMORY_SIMILARITY_THRESHOLD) / (100 - Config.MEMORY_SIMILARITY_THRESHOLD)
        bias = -scale * Config.MEMORY_BIAS_SCALE  # Negative = reduce confidence
        return bias, max_sim, match_idx

    return 0.0, max_sim, match_idx


def get_adaptive_thresholds(current_atr, atr_series):
    """
    Dynamically adjust probability thresholds based on volatility.
    High volatility → stricter thresholds (more selective)
    Low volatility → looser thresholds
    """
    if not Config.ADAPTIVE_THRESHOLD_ENABLED or atr_series is None or len(atr_series) < 20:
        return Config.PROB_THRESHOLD_BUY, Config.PROB_THRESHOLD_SELL

    atr_mean = atr_series.mean()
    atr_std = atr_series.std()

    if atr_std <= 0:
        return Config.PROB_THRESHOLD_BUY, Config.PROB_THRESHOLD_SELL

    z_score = (current_atr - atr_mean) / atr_std

    # High vol (z > 1): tighten thresholds by up to 5%
    # Low vol (z < -1): loosen thresholds by up to 5%
    adjustment = np.clip(z_score * 0.025, -0.05, 0.05)

    buy_threshold = Config.PROB_THRESHOLD_BUY + adjustment
    sell_threshold = Config.PROB_THRESHOLD_SELL - adjustment

    # Clamp to sane values
    buy_threshold = np.clip(buy_threshold, 0.55, 0.85)
    sell_threshold = np.clip(sell_threshold, 0.15, 0.45)

    return buy_threshold, sell_threshold


def apply_hybrid_filters(processed_df, direction, symbol, server_time=None):
    """
    The Hybrid Filter Layer — rejects bad signals before they reach execution.
    Returns (passed: bool, reason: str)
    """
    last = processed_df.iloc[-1]
    reasons = []

    # 1. ADX Range Filter: reject if market is ranging
    adx_val = last.get('ADX', 30)
    if adx_val < Config.ADX_RANGING_THRESHOLD:
        reasons.append(f"RANGING: ADX={adx_val:.1f} < {Config.ADX_RANGING_THRESHOLD}")

    # 2. Trend Alignment (H1): reject if trading against H1 trend
    h1_trend = last.get('H1_trend', 0)
    if h1_trend != 0:
        if direction == "BUY" and h1_trend == -1:
            reasons.append("COUNTER-TREND: BUY against H1 downtrend")
        elif direction == "SELL" and h1_trend == 1:
            reasons.append("COUNTER-TREND: SELL against H1 uptrend")

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
    print("  🚀 ELITE TRADING BOT v3.0 — PROFESSIONAL MT5 + LSTM ENGINE")
    print("=" * 65)
    print("  📊 Data Source   : MetaTrader 5 (Native)")
    print("  🧠 Model         : LSTM (Multi-Timeframe)")
    print("  ⚙️  Execution     : MT5 Direct (Smart Retry)")
    print("  🛡️  Risk Engine   : Equity Curve + Kill Switch + Cooldown")
    print("  📰 News Filter   : ForexFactory (High Impact)")
    print(f"  💹 Symbol        : {Config.FOREX_SYMBOL}")
    print(f"  📈 Risk/Trade    : {Config.RISK_PERCENT_PER_TRADE}%")
    print(f"  🔴 Daily Max Loss: {Config.MAX_DAILY_LOSS_PCT}%")
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

    # ===== PHASE 3: Import Data & Model modules =====
    from data_loader import fetch_mtf_data, fetch_tick_data, is_market_open
    from features import feature_engineering_pipeline
    from lstm_model import prepare_sequential_data, train_and_evaluate

    # State variables
    last_eval_candle = -1
    last_heartbeat = time.time()
    candle_index = 0

    print("\n\033[92m[STARTUP] ✅ All systems online. Entering main loop.\033[0m\n")

    while True:
        try:
            now = datetime.datetime.utcnow()

            # ===== HEARTBEAT =====
            if time.time() - last_heartbeat > Config.HEARTBEAT_INTERVAL_SECONDS:
                if not heartbeat():
                    logger.error("[Heartbeat] MT5 reconnection failed. Waiting 30s...")
                    time.sleep(30)
                    continue
                last_heartbeat = time.time()

            # ===== KILL SWITCH =====
            if check_kill_switch():
                logger.critical("[KILL SWITCH] Trading halted. Closing all positions...")
                close_all_positions()
                print("\n\033[91m⛔ KILL SWITCH ACTIVATED — Daily loss limit exceeded.\033[0m")
                print("\033[91m⛔ All positions closed. Bot paused until next day.\033[0m\n")
                # Wait until midnight
                tomorrow = (now + datetime.timedelta(days=1)).replace(hour=0, minute=5, second=0)
                wait_seconds = (tomorrow - now).total_seconds()
                logger.info("[KILL SWITCH] Sleeping %.0f seconds until next day.", wait_seconds)
                time.sleep(max(wait_seconds, 60))
                manager.reset_daily_stats(get_account_balance())
                continue

            # ===== DAILY RESET =====
            if now.hour == 0 and now.minute < 2:
                manager.reset_daily_stats(get_account_balance())

            # ===== MARKET OPEN CHECK =====
            if not is_market_open(Config.FOREX_SYMBOL):
                logger.info("[Market] %s is closed. Waiting...", Config.FOREX_SYMBOL)
                time.sleep(60)
                continue

            # ===== TICK MANAGEMENT (every cycle) =====
            tick_data = fetch_tick_data(Config.FOREX_SYMBOL)
            if tick_data and len(manager.active_trades) > 0:
                # Get current ATR for trailing stop calculations
                quick_df = None
                try:
                    from data_loader import fetch_mt5_ohlc
                    quick_df = fetch_mt5_ohlc(Config.FOREX_SYMBOL, Config.TIMEFRAME, 20)
                    if quick_df is not None and 'high' in quick_df.columns:
                        from ta.volatility import AverageTrueRange
                        atr_series = AverageTrueRange(
                            high=quick_df['high'], low=quick_df['low'],
                            close=quick_df['close'], window=14
                        ).average_true_range()
                        current_atr = atr_series.iloc[-1] if len(atr_series) > 0 else 0.001
                    else:
                        current_atr = 0.001
                except Exception:
                    current_atr = 0.001

                manager.on_tick(
                    Config.FOREX_SYMBOL,
                    tick_data['bid'],
                    tick_data['ask'],
                    current_atr,
                )

            # ===== CANDLE CLOSE CONFIRMATION (5-minute boundary) =====
            server_time = get_server_time(Config.FOREX_SYMBOL)
            server_minute = server_time.minute

            # Only evaluate on M5 candle close (minute % 5 == 0) and not repeated
            is_candle_close = (server_minute % 5 == 0) and (server_minute != last_eval_candle)

            if not is_candle_close:
                time.sleep(1)
                continue

            # Mark this candle as evaluated
            last_eval_candle = server_minute
            candle_index += 1

            logger.info("\n" + "=" * 55)
            logger.info("[EVALUATION] Candle #%d | Server Time: %s",
                        candle_index, server_time.strftime('%Y-%m-%d %H:%M:%S'))
            logger.info("=" * 55)

            # ===== FETCH MULTI-TIMEFRAME DATA =====
            mtf_data = fetch_mtf_data(Config.FOREX_SYMBOL)
            if mtf_data is None:
                logger.error("[Data] MTF fetch failed. Skipping cycle.")
                continue

            df_m5 = mtf_data["M5"]
            df_m15 = mtf_data.get("M15", pd.DataFrame())
            df_h1 = mtf_data.get("H1", pd.DataFrame())

            if df_m5 is None or df_m5.empty:
                logger.error("[Data] Primary M5 data empty. Skipping.")
                continue

            # ===== FEATURE ENGINEERING =====
            processed_df = feature_engineering_pipeline(
                df_m5,
                df_confirm=df_m15 if not df_m15.empty else None,
                df_trend=df_h1 if not df_h1.empty else None,
            )

            if processed_df is None or processed_df.empty:
                logger.error("[Features] Pipeline returned empty. Skipping.")
                continue

            # ===== LSTM PREDICTION =====
            try:
                X_train, X_test, y_train, y_test, scaler, train_weights = prepare_sequential_data(processed_df)
                model, history, acc = train_and_evaluate(
                    X_train, X_test, y_train, y_test, sample_weights=train_weights
                )
            except Exception as e:
                logger.error("[LSTM] Training failed: %s", e)
                continue

            latest_features = processed_df.drop(['Target'], axis=1).values
            latest_features_scaled = scaler.transform(latest_features)

            if len(latest_features_scaled) < Config.SEQUENCE_LENGTH:
                logger.warning("[LSTM] Not enough data for sequence. Skipping.")
                continue

            X_live = latest_features_scaled[-Config.SEQUENCE_LENGTH:]
            X_live = np.array([X_live])

            raw_prob = float(model.predict(X_live)[0][0])

            # ===== MEMORY PROBABILITY MODIFIER =====
            memory_bias, sim_pct, sim_idx = compute_memory_similarity(processed_df)
            adjusted_prob = raw_prob + memory_bias
            adjusted_prob = np.clip(adjusted_prob, 0.0, 1.0)

            if abs(memory_bias) > 0.001:
                logger.info("[Memory] Similarity: %.1f%% | Bias: %+.3f | Raw: %.4f → Adjusted: %.4f",
                            sim_pct, memory_bias, raw_prob, adjusted_prob)

            # ===== ADAPTIVE THRESHOLDS =====
            current_atr = processed_df['ATR'].iloc[-1]
            buy_threshold, sell_threshold = get_adaptive_thresholds(
                current_atr, processed_df['ATR']
            )

            # ===== DECISION =====
            direction = None
            if adjusted_prob > buy_threshold:
                direction = "BUY"
            elif adjusted_prob < sell_threshold:
                direction = "SELL"
            else:
                logger.info(
                    "[HOLD] Prob %.4f between thresholds (BUY>%.2f, SELL<%.2f). No trade.",
                    adjusted_prob, buy_threshold, sell_threshold,
                )
                continue

            logger.info("[SIGNAL] %s | Prob: %.4f (raw: %.4f) | Thresholds: BUY>%.2f SELL<%.2f",
                        direction, adjusted_prob, raw_prob, buy_threshold, sell_threshold)

            # ===== HYBRID FILTER LAYER =====
            filter_passed, filter_reason = apply_hybrid_filters(
                processed_df, direction, Config.FOREX_SYMBOL, server_time
            )

            if not filter_passed:
                logger.warning("[FILTERED] %s signal REJECTED: %s", direction, filter_reason)
                manager.log_rejected_trade(direction, filter_reason, adjusted_prob, sim_pct)
                continue

            # ===== TRADE MANAGER GUARD =====
            can_trade, guard_reason = manager.can_trade(direction, candle_index)
            if not can_trade:
                logger.warning("[GUARD] Trade blocked: %s", guard_reason)
                manager.log_rejected_trade(direction, guard_reason, adjusted_prob, sim_pct)
                continue

            # ===== ALL CLEAR — EXECUTE TRADE =====
            logger.info("[EXECUTING] %s %s — All filters passed ✅", direction, Config.FOREX_SYMBOL)

            # Calculate SL/TP in points
            import MetaTrader5 as mt5
            info = mt5.symbol_info(Config.FOREX_SYMBOL)
            point = info.point if info else 0.00001

            sl_points = int((current_atr * Config.SL_ATR_MULT) / point)
            tp1_points = int((current_atr * Config.TP1_ATR_MULT) / point)
            tp2_points = int((current_atr * Config.TP2_ATR_MULT) / point)

            # Equity curve risk adjustment
            risk_mult = manager.get_risk_multiplier(get_account_equity())

            signal_time_ms = time.time() * 1000

            result = execute_forex_trade(
                action=direction,
                symbol=Config.FOREX_SYMBOL,
                sl_points=sl_points,
                tp_points=tp1_points,  # TP set to TP1 initially
                risk_multiplier=risk_mult,
                signal_time_ms=signal_time_ms,
            )

            if result and result.get("success"):
                # Calculate TP2 price for trailing
                tick = mt5.symbol_info_tick(Config.FOREX_SYMBOL)
                if direction == "BUY":
                    tp2_price = result["filled_price"] + (tp2_points * point)
                else:
                    tp2_price = result["filled_price"] - (tp2_points * point)

                # Register with Trade Manager
                manager.register_trade(
                    ticket=result["ticket"],
                    symbol=Config.FOREX_SYMBOL,
                    direction=direction,
                    volume=result["volume"],
                    entry_price=result["filled_price"],
                    expected_price=result["expected_price"],
                    sl_price=result["sl_price"],
                    tp1_price=result["tp_price"],
                    tp2_price=tp2_price,
                    signal_time_ms=signal_time_ms,
                    fill_time_ms=result["fill_time_ms"],
                )

                # Update signal tracker for deduplication
                manager.update_signal_tracker(direction, candle_index)

                # Print status
                manager.print_status()

                # Print stats
                stats = manager.get_stats()
                if stats["total"] > 0:
                    print(f"\n\033[95m{'='*55}\033[0m")
                    print(f"\033[95m       📊 PERFORMANCE STATS\033[0m")
                    print(f"\033[95m{'='*55}\033[0m")
                    print(f"\033[95m  Total Trades  : {stats['total']}\033[0m")
                    print(f"\033[95m  Win Rate      : {stats['win_rate']:.1f}%\033[0m")
                    print(f"\033[95m  Profit Factor : {stats['profit_factor']:.2f}\033[0m")
                    print(f"\033[95m  Max Drawdown  : ${stats['max_dd']:.2f}\033[0m")
                    print(f"\033[95m  Daily P&L     : {'🟢' if stats['daily_pnl'] >= 0 else '🔴'} ${stats['daily_pnl']:.2f}\033[0m")
                    print(f"\033[95m{'='*55}\033[0m\n")

            else:
                logger.error("[EXECUTION FAILED] %s %s — No trade placed.",
                             direction, Config.FOREX_SYMBOL)

        except KeyboardInterrupt:
            print("\n\033[93m[EXIT] Bot stopped by user. Saving state...\033[0m")
            manager._save_state()
            print("\033[93m[EXIT] State saved. Active trades are still managed by MT5.\033[0m")
            break

        except Exception as e:
            logger.error("[MAIN LOOP EXCEPTION] %s", e, exc_info=True)
            time.sleep(5)


if __name__ == "__main__":
    main()

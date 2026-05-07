"""
Smart Exit Evaluator — Phase 2 (AI-Driven Dynamic Exits)
==========================================================
Evaluates open trades against live reversal signals from Phase 1 detectors.
If counter-trend signals accumulate past a danger threshold, recommends
early exit or tighter trailing stop to protect profits.

Usage:
    from smart_exit import evaluate_smart_exit
    should_exit, reason, danger = evaluate_smart_exit(trade_direction, processed_df, trade_info)
"""

import numpy as np
import pandas as pd
import os
import csv
import datetime
import logging
from config import Config

from logging_setup import setup_module_logger
logger = setup_module_logger("SmartExit", Config.LOG_FILE, console_color="\033[93m")


# =========================================
# COUNTER-SIGNAL SCORING TABLES
# =========================================

# Signals that are BEARISH (dangerous for BUY trades)
BEARISH_SIGNALS = {
    # Candlestick Patterns (candles.py)
    "Engulf_Bear":       2.0,
    "EveningStar":       2.5,
    "GravestoneDoji":    1.0,
    "DarkCloudCover":    1.5,
    "ThreeCrows":        2.5,
    "MarubozuBear":      1.5,
    "Hammer_Bear":       1.0,   # Shooting Star
    # Divergence (divergence.py)
    "RSI_BearDiv":       3.0,
    "RSI_HiddenBearDiv": 2.0,
    # Chart Patterns (pattern_detector.py)
    "HS_Flag":           2.5,   # Head & Shoulders
    "DoubleTop_Flag":    2.0,
    "TripleTop_Flag":    2.5,
    "DescTriangle_Flag": 1.5,
    "RisingWedge_Flag":  1.5,
    "BearFlag_Flag":     1.5,
}

# Signals that are BULLISH (dangerous for SELL trades)
BULLISH_SIGNALS = {
    # Candlestick Patterns
    "Engulf_Bull":        2.0,
    "MorningStar":        2.5,
    "DragonflyDoji":      1.0,
    "PiercingLine":       1.5,
    "ThreeSoldiers":      2.5,
    "MarubozuBull":       1.5,
    "Hammer_Bull":        1.0,
    # Divergence
    "RSI_BullDiv":        3.0,
    "RSI_HiddenBullDiv":  2.0,
    # Chart Patterns
    "InvHS_Flag":         2.5,
    "DoubleBottom_Flag":  2.0,
    "TripleBottom_Flag":  2.5,
    "AscTriangle_Flag":   1.5,
    "FallingWedge_Flag":  1.5,
    "BullFlag_Flag":      1.5,
}


# =========================================
# MAIN API: evaluate_smart_exit()
# =========================================

def evaluate_smart_exit(trade_direction, processed_df, trade_info=None):
    """
    Evaluates whether an open trade should be closed early based on
    counter-trend reversal signals from Phase 1 detectors.

    Args:
        trade_direction: "BUY" or "SELL"
        processed_df:    DataFrame from feature_engineering_pipeline()
                         (must contain Phase 1 detector columns)
        trade_info:      Optional dict with trade metadata:
                         {
                             "ticket": int,
                             "symbol": str,
                             "entry_price": float,
                             "candles_open": int,    # How many candles since entry
                             "current_pnl": float,   # Current unrealized P&L
                         }

    Returns:
        tuple: (should_exit: bool, reason: str, danger_score: float)
            - should_exit: True if trade should be closed immediately
            - reason: Human-readable explanation
            - danger_score: Cumulative counter-signal score (0.0 = safe)
    """
    if not getattr(Config, 'SMART_EXIT_ENABLED', True):
        return False, "Smart Exit disabled", 0.0

    if processed_df is None or processed_df.empty:
        return False, "No data", 0.0

    # Get the latest candle's signals
    last = processed_df.iloc[-1]

    # ===== SAFETY GUARD 1: Minimum candles open =====
    min_candles = getattr(Config, 'SMART_EXIT_MIN_CANDLES_OPEN', 3)
    if trade_info and trade_info.get("candles_open", 999) < min_candles:
        return False, f"Trade too young ({trade_info.get('candles_open', 0)}/{min_candles} candles)", 0.0

    # ===== SAFETY GUARD 2: Only exit in profit =====
    if getattr(Config, 'SMART_EXIT_ONLY_IN_PROFIT', True):
        if trade_info and trade_info.get("current_pnl", 0) <= 0:
            return False, "Trade not in profit — letting SL handle it", 0.0

    # ===== CALCULATE DANGER SCORE =====
    danger_score = 0.0
    triggered_signals = []

    if trade_direction == "BUY":
        signal_table = BEARISH_SIGNALS
    else:
        signal_table = BULLISH_SIGNALS

    for signal_name, weight in signal_table.items():
        if signal_name in processed_df.columns:
            value = last.get(signal_name, 0)
            if value and value > 0:
                danger_score += weight
                triggered_signals.append(f"{signal_name}(+{weight})")

    # ===== COMPOSITE SCORE BOOST =====
    # Use the composite scores for additional context
    if trade_direction == "BUY":
        reversal_score = last.get("candle_reversal_score", 0)
        if reversal_score < -1:  # Strong bearish reversal composite
            danger_score += abs(reversal_score) * 0.5
            triggered_signals.append(f"reversal_composite({reversal_score})")

        pattern_bias = last.get("pattern_bias_score", 0)
        if pattern_bias < -1:  # Strong bearish pattern bias
            danger_score += abs(pattern_bias) * 0.5
            triggered_signals.append(f"pattern_bias({pattern_bias})")

        div_score = last.get("divergence_score", 0)
        if div_score < 0:
            danger_score += abs(div_score) * 0.5
            triggered_signals.append(f"div_score({div_score})")

    elif trade_direction == "SELL":
        reversal_score = last.get("candle_reversal_score", 0)
        if reversal_score > 1:
            danger_score += reversal_score * 0.5
            triggered_signals.append(f"reversal_composite(+{reversal_score})")

        pattern_bias = last.get("pattern_bias_score", 0)
        if pattern_bias > 1:
            danger_score += pattern_bias * 0.5
            triggered_signals.append(f"pattern_bias(+{pattern_bias})")

        div_score = last.get("divergence_score", 0)
        if div_score > 0:
            danger_score += div_score * 0.5
            triggered_signals.append(f"div_score(+{div_score})")

    # ===== VOLATILITY COMPRESSION WARNING =====
    # If volatility is compressing, a reversal is more likely to be explosive
    vol_compress = last.get("Volatility_Compress", 0)
    if vol_compress > 0 and danger_score > 0:
        danger_score *= 1.2  # 20% amplification during squeeze
        triggered_signals.append("vol_squeeze_amplify(x1.2)")

    # ===== DECISION =====
    exit_threshold = getattr(Config, 'SMART_EXIT_DANGER_THRESHOLD', 3.0)
    tighten_threshold = getattr(Config, 'SMART_EXIT_TIGHTEN_THRESHOLD', 2.0)

    signal_str = " + ".join(triggered_signals) if triggered_signals else "none"

    if danger_score >= exit_threshold:
        reason = f"DANGER={danger_score:.1f} >= {exit_threshold}: [{signal_str}]"
        should_exit = True
    else:
        reason = f"danger={danger_score:.1f} < {exit_threshold}: [{signal_str}]"
        should_exit = False

    # ===== LOG =====
    if danger_score > 0:
        _log_evaluation(trade_info, trade_direction, danger_score, should_exit, signal_str)

    if should_exit:
        logger.warning(
            "[SMART EXIT] %s %s | Ticket #%s | %s",
            "EXIT" if should_exit else "TIGHTEN",
            trade_direction,
            trade_info.get("ticket", "?") if trade_info else "?",
            reason
        )
    elif danger_score >= tighten_threshold:
        logger.info(
            "[SMART EXIT] TIGHTEN SL for %s | Ticket #%s | danger=%.1f | %s",
            trade_direction,
            trade_info.get("ticket", "?") if trade_info else "?",
            danger_score, signal_str
        )

    return should_exit, reason, danger_score


def should_tighten_sl(danger_score):
    """
    Returns True if the danger score is high enough to tighten the trailing stop,
    but not high enough for a full exit.
    """
    tighten_threshold = getattr(Config, 'SMART_EXIT_TIGHTEN_THRESHOLD', 2.0)
    exit_threshold = getattr(Config, 'SMART_EXIT_DANGER_THRESHOLD', 3.0)
    return tighten_threshold <= danger_score < exit_threshold


def get_tighten_atr_mult():
    """Returns the ATR multiplier to use for tightened trailing stops."""
    return getattr(Config, 'SMART_EXIT_TIGHTEN_ATR_MULT', 0.5)


# =========================================
# LOGGING — Smart Exit Evaluation History
# =========================================

def _log_evaluation(trade_info, direction, danger_score, should_exit, signals_str):
    """Log every non-zero evaluation to CSV for post-analysis."""
    try:
        filepath = os.path.join(os.path.dirname(__file__), "smart_exit_log.csv")
        file_exists = os.path.isfile(filepath)

        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "ticket", "symbol", "direction",
                    "danger_score", "should_exit", "candles_open",
                    "current_pnl", "signals"
                ])
            writer.writerow([
                datetime.datetime.utcnow().isoformat(),
                trade_info.get("ticket", "") if trade_info else "",
                trade_info.get("symbol", "") if trade_info else "",
                direction,
                f"{danger_score:.2f}",
                should_exit,
                trade_info.get("candles_open", "") if trade_info else "",
                f"{trade_info.get('current_pnl', 0):.5f}" if trade_info else "",
                signals_str,
            ])
    except Exception as e:
        logger.error("[SmartExit] Log write failed: %s", e)

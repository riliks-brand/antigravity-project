"""
Ensemble Engine — Elite v3.1
==============================
The Brain: combines LSTM + Random Forest with intelligent voting.

Features:
- Dynamic Weighting based on market state (ADX trending vs ranging)
- Conflict Detection (models strongly disagree → SKIP)
- Disagreement Penalty (prevent overconfidence masking)
- Adaptive Execution Thresholds (volatility-aware)
- Comprehensive Ensemble Decision Logging (CSV)
- Feature Importance extraction from RF
"""

import numpy as np
import os
import csv
import datetime
import logging
from config import Config

logger = logging.getLogger("Ensemble")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(Config.LOG_FILE, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("\033[92m%(asctime)s\033[0m [%(levelname)s] %(message)s"))
    logger.addHandler(_ch)


class EnsembleDecision:
    """Container for ensemble prediction results."""

    def __init__(self):
        self.lstm_prob = 0.5
        self.rf_prob = 0.5
        self.weighted_avg = 0.5
        self.penalty = 0.0
        self.final_prob = 0.5
        self.direction = None       # "BUY", "SELL", or None (HOLD/SKIP)
        self.skip_reason = None     # Why trade was skipped (if any)
        self.lstm_weight = 0.5
        self.rf_weight = 0.5
        self.buy_threshold = Config.PROB_THRESHOLD_BUY
        self.sell_threshold = Config.PROB_THRESHOLD_SELL
        self.market_state = "UNKNOWN"
        self.conflict = False

    def __repr__(self):
        return (
            f"EnsembleDecision(LSTM={self.lstm_prob:.4f} x{self.lstm_weight:.0%}, "
            f"RF={self.rf_prob:.4f} x{self.rf_weight:.0%}, "
            f"Penalty={self.penalty:.4f}, Final={self.final_prob:.4f}, "
            f"Dir={self.direction}, State={self.market_state})"
        )


def get_dynamic_weights(current_adx):
    """
    Calculate model weights based on market state.

    Trending (ADX > 25): LSTM=70%, RF=30%
      → LSTM excels at reading sequential momentum
    Ranging (ADX < 25): LSTM=50%, RF=50%
      → RF handles noise/mean-reversion better
    """
    if current_adx >= Config.ADX_RANGING_THRESHOLD:
        lstm_w = Config.ENSEMBLE_LSTM_WEIGHT_TRENDING
        rf_w = Config.ENSEMBLE_RF_WEIGHT_TRENDING
        state = "TRENDING"
    else:
        lstm_w = Config.ENSEMBLE_LSTM_WEIGHT_RANGING
        rf_w = Config.ENSEMBLE_RF_WEIGHT_RANGING
        state = "RANGING"

    logger.debug("[Ensemble] ADX=%.1f → %s | LSTM=%.0f%% RF=%.0f%%",
                 current_adx, state, lstm_w * 100, rf_w * 100)
    return lstm_w, rf_w, state


def get_adaptive_thresholds(current_atr, atr_series):
    """
    Volatility-aware execution thresholds.

    High volatility → stricter (0.80) — avoid false breakouts
    Low volatility → looser (0.65) — capture slower moves
    """
    if not Config.ADAPTIVE_THRESHOLD_ENABLED or atr_series is None or len(atr_series) < 20:
        return Config.PROB_THRESHOLD_BUY, Config.PROB_THRESHOLD_SELL

    atr_mean = atr_series.mean()
    atr_std = atr_series.std()

    if atr_std <= 0:
        return Config.PROB_THRESHOLD_BUY, Config.PROB_THRESHOLD_SELL

    z_score = (current_atr - atr_mean) / atr_std

    # High vol (z > 1): tighten thresholds
    # Low vol (z < -1): loosen thresholds
    adjustment = np.clip(z_score * 0.025, -0.05, 0.05)

    buy_threshold = Config.PROB_THRESHOLD_BUY + adjustment
    sell_threshold = Config.PROB_THRESHOLD_SELL - adjustment

    # Clamp
    buy_threshold = np.clip(buy_threshold, 0.55, 0.85)
    sell_threshold = np.clip(sell_threshold, 0.15, 0.45)

    return float(buy_threshold), float(sell_threshold)


def ensemble_predict(lstm_prob, rf_prob, current_adx, current_atr, atr_series):
    """
    The core ensemble prediction function.

    Args:
        lstm_prob: LSTM's probability of class 1 (bullish) ∈ [0, 1]
        rf_prob: Random Forest's probability of class 1 (bullish) ∈ [0, 1]
        current_adx: Latest ADX value (for dynamic weighting)
        current_atr: Latest ATR value (for adaptive thresholds)
        atr_series: Historical ATR series (for z-score calculation)

    Returns:
        EnsembleDecision object with all details
    """
    decision = EnsembleDecision()
    decision.lstm_prob = lstm_prob
    decision.rf_prob = rf_prob

    # --- Step 1: Dynamic Weighting ---
    lstm_w, rf_w, state = get_dynamic_weights(current_adx)
    decision.lstm_weight = lstm_w
    decision.rf_weight = rf_w
    decision.market_state = state

    # --- Step 2: Weighted Average ---
    weighted_avg = (lstm_w * lstm_prob) + (rf_w * rf_prob)
    decision.weighted_avg = weighted_avg

    # --- Step 3: Conflict Detection ---
    disagreement = abs(lstm_prob - rf_prob)

    if disagreement >= Config.ENSEMBLE_CONFLICT_THRESHOLD:
        decision.conflict = True
        decision.skip_reason = (
            f"CONFLICT: |LSTM({lstm_prob:.3f}) - RF({rf_prob:.3f})| = "
            f"{disagreement:.3f} >= {Config.ENSEMBLE_CONFLICT_THRESHOLD}"
        )
        decision.final_prob = weighted_avg
        decision.direction = None
        logger.warning("[Ensemble] ⚠️ %s → SKIP TRADE", decision.skip_reason)
        _log_decision(decision)
        return decision

    # --- Step 4: Disagreement Penalty ---
    penalty = disagreement * Config.ENSEMBLE_DISAGREEMENT_PENALTY
    decision.penalty = penalty

    # Apply penalty: reduces confidence when models don't fully agree
    # For BUY signals: push final_prob DOWN (harder to reach threshold)
    # For SELL signals: push final_prob UP (harder to reach threshold)
    if weighted_avg > 0.5:
        final_prob = weighted_avg - penalty
    else:
        final_prob = weighted_avg + penalty

    final_prob = np.clip(final_prob, 0.0, 1.0)
    decision.final_prob = float(final_prob)

    # --- Step 5: Adaptive Thresholds ---
    buy_threshold, sell_threshold = get_adaptive_thresholds(current_atr, atr_series)
    decision.buy_threshold = buy_threshold
    decision.sell_threshold = sell_threshold

    # --- Step 6: Decision ---
    if final_prob > buy_threshold:
        decision.direction = "BUY"
    elif final_prob < sell_threshold:
        decision.direction = "SELL"
    else:
        decision.direction = None
        decision.skip_reason = (
            f"HOLD: Final {final_prob:.4f} between "
            f"BUY>{buy_threshold:.2f} and SELL<{sell_threshold:.2f}"
        )

    # --- Log the full decision ---
    _log_decision(decision)

    # Print
    status = decision.direction or "HOLD"
    conflict_flag = " ⚠️CONFLICT" if decision.conflict else ""
    print(f"\n\033[92m{'='*60}\033[0m")
    print(f"\033[92m       🧠 ENSEMBLE DECISION{conflict_flag}\033[0m")
    print(f"\033[92m{'='*60}\033[0m")
    print(f"\033[92m  Market State  : {state} (ADX: {current_adx:.1f})\033[0m")
    print(f"\033[92m  LSTM          : {lstm_prob:.4f} (weight: {lstm_w:.0%})\033[0m")
    print(f"\033[92m  Random Forest : {rf_prob:.4f} (weight: {rf_w:.0%})\033[0m")
    print(f"\033[92m  Weighted Avg  : {weighted_avg:.4f}\033[0m")
    print(f"\033[92m  Disagreement  : {disagreement:.4f} → Penalty: {penalty:.4f}\033[0m")
    print(f"\033[92m  Final Score   : {final_prob:.4f}\033[0m")
    print(f"\033[92m  Thresholds    : BUY>{buy_threshold:.2f} | SELL<{sell_threshold:.2f}\033[0m")
    print(f"\033[92m  ▶ DECISION    : {status}\033[0m")
    print(f"\033[92m{'='*60}\033[0m\n")

    return decision


# =========================================
# ENSEMBLE LOGGING (CSV)
# =========================================

def _log_decision(decision):
    """Log every ensemble decision to CSV for post-analysis."""
    try:
        filepath = Config.ENSEMBLE_LOG_FILE
        file_exists = os.path.isfile(filepath)

        fieldnames = [
            "timestamp", "market_state", "adx",
            "lstm_prob", "lstm_weight",
            "rf_prob", "rf_weight",
            "weighted_avg", "disagreement", "penalty",
            "final_prob", "buy_threshold", "sell_threshold",
            "direction", "conflict", "skip_reason",
        ]

        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "market_state": decision.market_state,
                "adx": "",  # Will be set from outside if needed
                "lstm_prob": f"{decision.lstm_prob:.6f}",
                "lstm_weight": f"{decision.lstm_weight:.2f}",
                "rf_prob": f"{decision.rf_prob:.6f}",
                "rf_weight": f"{decision.rf_weight:.2f}",
                "weighted_avg": f"{decision.weighted_avg:.6f}",
                "disagreement": f"{abs(decision.lstm_prob - decision.rf_prob):.6f}",
                "penalty": f"{decision.penalty:.6f}",
                "final_prob": f"{decision.final_prob:.6f}",
                "buy_threshold": f"{decision.buy_threshold:.4f}",
                "sell_threshold": f"{decision.sell_threshold:.4f}",
                "direction": decision.direction or "HOLD",
                "conflict": decision.conflict,
                "skip_reason": decision.skip_reason or "",
            })

    except Exception as e:
        logger.error("[Ensemble] Failed to log decision: %s", e)

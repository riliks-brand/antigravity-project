"""
Ensemble Engine — Elite v4.0
==============================
The Brain: Session-Aware, Market-Adaptive Decision Engine.

Architecture:
  Session Context + Market Regime + Model Ensemble = Final Decision

Features:
- Smooth Dynamic Weighting based on continuous trend_strength (NOT binary)
- Session-Aware Strategy Behavior (London/NY/Asia)
- Additive Scoring Model (Base + Session_Bonus + Volatility_Adjustment)
- Adjustment-to-Base Ratio Control (max 10% of base)
- Weak Zone Hysteresis (0.52–0.56 = NO ENTRY)
- Score Floor (< 0.52 = REJECT)
- Regime Conflict Detection (session vs trend mismatch → HOLD)
- ATR Double Filter (ratio + absolute)
- Confidence Classification (HIGH / MEDIUM / LOW)
- Decision Reason on every path (explainability)
- Comprehensive CSV Logging with all fields
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
    """Container for ensemble prediction results — fully explainable."""

    def __init__(self):
        self.lstm_prob = 0.5
        self.rf_prob = 0.5
        self.weighted_avg = 0.5
        self.penalty = 0.0
        self.raw_score = 0.5
        self.final_prob = 0.5
        self.direction = None       # "BUY", "SELL", or None (HOLD/SKIP)
        self.skip_reason = None     # Legacy field (kept for backward compat)
        self.lstm_weight = 0.5
        self.rf_weight = 0.5
        self.buy_threshold = 0.58
        self.sell_threshold = 0.42
        self.market_state = "UNKNOWN"
        self.conflict = False

        # --- New v4.0 Fields ---
        self.session = "UNKNOWN"
        self.trend_strength = 0.0
        self.session_bonus = 0.0
        self.volatility_adjustment = 0.0
        self.regime_conflict = False
        self.decision_reason = ""
        self.confidence_level = "LOW"

    def __repr__(self):
        return (
            f"EnsembleDecision(LSTM={self.lstm_prob:.4f} x{self.lstm_weight:.0%}, "
            f"RF={self.rf_prob:.4f} x{self.rf_weight:.0%}, "
            f"Penalty={self.penalty:.4f}, Raw={self.raw_score:.4f}, Final={self.final_prob:.4f}, "
            f"Dir={self.direction}, Session={self.session}, Trend={self.trend_strength:.2f}, "
            f"Confidence={self.confidence_level}, Reason={self.decision_reason})"
        )


# =========================================
# DYNAMIC MODEL WEIGHTING (Smooth, NOT binary)
# =========================================

def get_dynamic_weights(trend_strength, session):
    """
    Calculate model weights using SMOOTH continuous weighting.

    Formula (EXACT — no simplification):
        LSTM_weight = 0.5 + (trend_strength * 0.3)
        RF_weight   = 0.5 - (trend_strength * 0.3)

    This means:
        trend_strength = 0.0 → LSTM=50%, RF=50%  (pure ranging)
        trend_strength = 0.5 → LSTM=65%, RF=35%  (moderate trend)
        trend_strength = 1.0 → LSTM=80%, RF=20%  (strong trend)

    Session does NOT override this — it only provides bonuses via additive scoring.
    """
    lstm_w = 0.5 + (trend_strength * 0.3)
    rf_w = 0.5 - (trend_strength * 0.3)

    # Determine market state label
    if trend_strength >= 0.6:
        state = "TRENDING"
    elif trend_strength <= 0.2:
        state = "RANGING"
    else:
        state = "TRANSITIONING"

    logger.debug(
        "[Ensemble] trend_strength=%.3f → %s | LSTM=%.1f%% RF=%.1f%% | Session=%s",
        trend_strength, state, lstm_w * 100, rf_w * 100, session
    )
    return lstm_w, rf_w, state


# =========================================
# SESSION BONUS CALCULATION
# =========================================

def _compute_session_bonus(session, trend_strength):
    """
    Compute additive session bonus based on session type and trend alignment.

    Session Strategy:
        London  → Trend Following  → bonus when trend is strong
        NY      → Balanced/Hybrid  → small neutral bonus
        Asia    → Mean Reversion   → bonus when market is ranging

    Returns:
        float: session_bonus, ALWAYS clipped to [-0.03, +0.03]
    """
    raw_bonus = 0.0

    if session == "London":
        # London favors trend: bonus when trend_strength is high
        if trend_strength >= 0.5:
            raw_bonus = 0.02
        elif trend_strength >= 0.3:
            raw_bonus = 0.01
        else:
            raw_bonus = -0.01  # London with no trend = slight penalty
    elif session == "New York":
        # NY is balanced: small bonus if there's moderate clarity
        if trend_strength >= 0.4:
            raw_bonus = 0.01
        else:
            raw_bonus = 0.0
    elif session == "Asia":
        # Asia favors ranging/mean-reversion: bonus when trend is low
        if trend_strength <= 0.3:
            raw_bonus = 0.02
        elif trend_strength <= 0.5:
            raw_bonus = 0.01
        else:
            raw_bonus = -0.01  # Asia with strong trend = unusual, penalize
    else:
        # UNKNOWN session → no bonus
        raw_bonus = 0.0

    # STRICT CLIP: session_bonus is ALWAYS in [-0.03, +0.03]
    return float(np.clip(raw_bonus, -0.03, 0.03))


# =========================================
# VOLATILITY ADJUSTMENT
# =========================================

def _compute_volatility_adjustment(current_atr, atr_series):
    """
    Compute additive volatility adjustment based on ATR z-score.

    High volatility → slight positive adjustment (momentum opportunities)
    Low volatility  → slight negative adjustment (weak moves, avoid)

    Returns:
        float: volatility_adjustment (small value, will be further capped)
    """
    if atr_series is None or len(atr_series) < 20:
        return 0.0

    atr_mean = atr_series.mean()
    atr_std = atr_series.std()

    if atr_std <= 0 or atr_mean <= 0:
        return 0.0

    z_score = (current_atr - atr_mean) / atr_std

    # Map z-score to a small adjustment
    # z > 1: high vol → +0.01 to +0.02
    # z < -1: low vol → -0.01 to -0.02
    adjustment = np.clip(z_score * 0.01, -0.02, 0.02)

    return float(adjustment)


# =========================================
# REGIME CONFLICT DETECTION
# =========================================

def _detect_regime_conflict(session, trend_strength):
    """
    Detect if there's a strong conflict between session expectation and market reality.

    Conflicts:
        - London (expects trend) but trend_strength < 0.2  → CONFLICT
        - Asia (expects range) but trend_strength > 0.8    → CONFLICT

    Returns:
        bool: True if strong conflict detected → must return HOLD
    """
    if session == "London" and trend_strength < 0.2:
        # London should be trending, but market is dead/ranging
        logger.warning(
            "[Ensemble] ⚠️ REGIME CONFLICT: London session but trend_strength=%.2f (ranging)",
            trend_strength
        )
        return True

    if session == "Asia" and trend_strength > 0.8:
        # Asia should be quiet/ranging, but market is strongly trending
        logger.warning(
            "[Ensemble] ⚠️ REGIME CONFLICT: Asia session but trend_strength=%.2f (strong trend)",
            trend_strength
        )
        return True

    return False


# =========================================
# CORE ENSEMBLE PREDICTION
# =========================================

def ensemble_predict(lstm_prob, rf_prob, current_adx, current_atr, atr_series, session="UNKNOWN"):
    """
    The core ensemble prediction function — Session-Aware & Market-Adaptive.

    Args:
        lstm_prob: LSTM's probability of class 1 (bullish) ∈ [0, 1]
        rf_prob: Random Forest's probability of class 1 (bullish) ∈ [0, 1]
        current_adx: Latest ADX value (for trend strength calculation)
        current_atr: Latest ATR value (for volatility filtering)
        atr_series: Historical ATR series (for ratio & z-score calculations)
        session: Current trading session ("London", "New York", "Asia", "UNKNOWN")

    Returns:
        EnsembleDecision object with all details + decision_reason
    """
    decision = EnsembleDecision()
    decision.lstm_prob = lstm_prob
    decision.rf_prob = rf_prob
    decision.session = session

    # =========================================
    # Step 1: Trend Strength Normalization
    # EXACT FORMULA: clip((ADX - 20) / 30, 0, 1)
    # ADX < 20 → 0.0 (no trend)
    # ADX = 35 → 0.5 (moderate)
    # ADX ≥ 50 → 1.0 (strong trend)
    # =========================================
    trend_strength = float(np.clip((current_adx - 20) / 30.0, 0.0, 1.0))
    decision.trend_strength = trend_strength

    # =========================================
    # Step 2: ATR Double Filter (ratio + absolute)
    # BOTH conditions must be checked
    # =========================================
    atr_mean = atr_series.mean() if atr_series is not None and len(atr_series) > 0 else 0
    atr_ratio = (current_atr / atr_mean) if atr_mean > 0 else 1.0

    if atr_ratio < 0.5 or current_atr < Config.ATR_THRESHOLD:
        decision.direction = None
        decision.decision_reason = "LOW_ATR"
        decision.final_prob = 0.5
        decision.confidence_level = "LOW"
        logger.warning(
            "[Ensemble] ⛔ LOW_ATR: atr_ratio=%.3f, current_atr=%.6f, threshold=%.6f → SKIP",
            atr_ratio, current_atr, Config.ATR_THRESHOLD
        )
        _log_decision(decision, current_adx, current_atr)
        return decision

    # =========================================
    # Step 3: Dynamic Weights (smooth, NOT binary)
    # LSTM = 0.5 + (trend_strength * 0.3)
    # RF   = 0.5 - (trend_strength * 0.3)
    # =========================================
    lstm_w, rf_w, market_state = get_dynamic_weights(trend_strength, session)
    decision.lstm_weight = lstm_w
    decision.rf_weight = rf_w
    decision.market_state = market_state

    # =========================================
    # Step 4: Weighted Average (Base Model Score)
    # =========================================
    weighted_avg = (lstm_w * lstm_prob) + (rf_w * rf_prob)
    decision.weighted_avg = weighted_avg

    # =========================================
    # Step 5: Conflict Detection (model disagreement)
    # =========================================
    disagreement = abs(lstm_prob - rf_prob)

    if disagreement >= Config.ENSEMBLE_CONFLICT_THRESHOLD:
        decision.conflict = True
        decision.direction = None
        decision.final_prob = weighted_avg
        decision.decision_reason = "CONFLICT"
        decision.confidence_level = "LOW"
        decision.skip_reason = (
            f"MODEL CONFLICT: |LSTM({lstm_prob:.3f}) - RF({rf_prob:.3f})| = "
            f"{disagreement:.3f} >= {Config.ENSEMBLE_CONFLICT_THRESHOLD}"
        )
        logger.warning("[Ensemble] ⚠️ %s → HOLD", decision.skip_reason)
        _log_decision(decision, current_adx, current_atr)
        return decision

    # =========================================
    # Step 6: Disagreement Penalty
    # =========================================
    penalty = disagreement * Config.ENSEMBLE_DISAGREEMENT_PENALTY
    decision.penalty = penalty

    # Apply penalty: reduces confidence when models don't fully agree
    if weighted_avg > 0.5:
        base_score = weighted_avg - penalty
    else:
        base_score = weighted_avg + penalty

    base_score = float(np.clip(base_score, 0.0, 1.0))

    # =========================================
    # Step 7: Score Floor — signal too weak → REJECT
    # Applied SYMMETRICALLY: distance from 0.5 < 0.02 means no signal
    # For BUY: base_score < 0.52 → reject
    # For SELL: base_score > 0.48 → reject (mirror)
    # =========================================
    distance_from_neutral = abs(base_score - 0.5)
    if distance_from_neutral < 0.02:
        decision.direction = None
        decision.final_prob = base_score
        decision.raw_score = base_score
        decision.decision_reason = "BELOW_THRESHOLD"
        decision.confidence_level = "LOW"
        decision.skip_reason = f"SCORE FLOOR: base_score={base_score:.4f}, distance={distance_from_neutral:.4f} < 0.02"
        logger.info("[Ensemble] ⛔ %s → REJECT", decision.skip_reason)
        _log_decision(decision, current_adx, current_atr)
        return decision

    # =========================================
    # Step 8: Weak Zone Hysteresis (NO ENTRY)
    # Applied SYMMETRICALLY using distance from 0.5
    # Distance 0.02–0.06 = WEAK ZONE (too uncertain)
    # BUY side: base_score 0.52–0.56
    # SELL side: base_score 0.44–0.48
    # =========================================
    if distance_from_neutral < 0.06:
        decision.direction = None
        decision.final_prob = base_score
        decision.raw_score = base_score
        decision.decision_reason = "WEAK_ZONE"
        decision.confidence_level = "LOW"
        decision.skip_reason = f"WEAK ZONE: base_score={base_score:.4f}, distance={distance_from_neutral:.4f} < 0.06"
        logger.info("[Ensemble] ⚠️ %s → NO ENTRY", decision.skip_reason)
        _log_decision(decision, current_adx, current_atr)
        return decision

    # =========================================
    # Step 9: Regime Conflict (session vs trend mismatch → HOLD)
    # NOT a penalty. NOT a score reduction. Direct HOLD.
    # =========================================
    regime_conflict = _detect_regime_conflict(session, trend_strength)
    decision.regime_conflict = regime_conflict

    if regime_conflict:
        decision.direction = None
        decision.final_prob = base_score
        decision.raw_score = base_score
        decision.decision_reason = "CONFLICT"
        decision.confidence_level = "LOW"
        decision.skip_reason = f"REGIME CONFLICT: session={session}, trend_strength={trend_strength:.2f}"
        logger.warning("[Ensemble] ⛔ %s → HOLD", decision.skip_reason)
        _log_decision(decision, current_adx, current_atr)
        return decision

    # =========================================
    # Step 10: Additive Scoring Model (STRICTLY additive, NO multiplication)
    # Final = Base + Session_Bonus + Volatility_Adjustment
    # =========================================

    # 10a. Session Bonus: clipped to [-0.03, +0.03]
    session_bonus = _compute_session_bonus(session, trend_strength)
    decision.session_bonus = session_bonus

    # 10b. Volatility Adjustment
    volatility_adjustment = _compute_volatility_adjustment(current_atr, atr_series)
    decision.volatility_adjustment = volatility_adjustment

    # 10c. Total Adjustment: clipped to [-0.05, +0.05]
    raw_total_adjustment = float(np.clip(session_bonus + volatility_adjustment, -0.05, 0.05))

    # 10d. Adjustment-to-Base Ratio Control: adjustment ≤ 10% of base_score
    # This GUARANTEES boosts cannot create a trade alone
    if raw_total_adjustment > 0:
        total_adjustment = min(raw_total_adjustment, base_score * 0.1)
    else:
        total_adjustment = max(raw_total_adjustment, -(base_score * 0.1))

    # 10e. Compute raw_score and clip
    raw_score = base_score + total_adjustment
    decision.raw_score = raw_score
    final_prob = float(np.clip(raw_score, 0.0, 1.0))
    decision.final_prob = final_prob

    # =========================================
    # Step 11: Dynamic Thresholds (Regime-Aware)
    # EXACT FORMULA: buy_threshold = 0.58 + (1 - trend_strength) * 0.08
    # Strong trend (ts=1.0) → buy_threshold = 0.58
    # No trend (ts=0.0) → buy_threshold = 0.66
    # =========================================
    buy_threshold = 0.58 + (1.0 - trend_strength) * 0.08
    sell_threshold = 1.0 - buy_threshold  # Mirror for sell side

    decision.buy_threshold = buy_threshold
    decision.sell_threshold = sell_threshold

    # =========================================
    # Step 12: Direction Decision
    # =========================================
    if final_prob > buy_threshold:
        decision.direction = "BUY"
        decision.decision_reason = "VALID_SIGNAL"
    elif final_prob < sell_threshold:
        decision.direction = "SELL"
        decision.decision_reason = "VALID_SIGNAL"
    else:
        decision.direction = None
        decision.decision_reason = "BELOW_THRESHOLD"
        decision.skip_reason = (
            f"HOLD: Final {final_prob:.4f} between "
            f"BUY>{buy_threshold:.4f} and SELL<{sell_threshold:.4f}"
        )

    # =========================================
    # Step 13: Confidence Classification
    # >0.65 → HIGH | 0.58–0.65 → MEDIUM | else → LOW
    # =========================================
    distance_from_center = abs(final_prob - 0.5)
    if distance_from_center > 0.15:  # final_prob > 0.65 or < 0.35
        decision.confidence_level = "HIGH"
    elif distance_from_center > 0.08:  # final_prob > 0.58 or < 0.42
        decision.confidence_level = "MEDIUM"
    else:
        decision.confidence_level = "LOW"

    # =========================================
    # Step 14: Log the full decision
    # =========================================
    _log_decision(decision, current_adx, current_atr)

    # =========================================
    # Step 15: Console Print (human-readable)
    # =========================================
    status = decision.direction or "HOLD"
    conflict_flag = " ⚠️CONFLICT" if decision.conflict or decision.regime_conflict else ""
    print(f"\n\033[92m{'='*65}\033[0m")
    print(f"\033[92m       🧠 ENSEMBLE DECISION v4.0{conflict_flag}\033[0m")
    print(f"\033[92m{'='*65}\033[0m")
    print(f"\033[92m  Session       : {session}\033[0m")
    print(f"\033[92m  Market State  : {market_state} (ADX: {current_adx:.1f} → trend_strength: {trend_strength:.3f})\033[0m")
    print(f"\033[92m  LSTM          : {lstm_prob:.4f} (weight: {lstm_w:.0%})\033[0m")
    print(f"\033[92m  Random Forest : {rf_prob:.4f} (weight: {rf_w:.0%})\033[0m")
    print(f"\033[92m  Weighted Avg  : {weighted_avg:.4f}\033[0m")
    print(f"\033[92m  Disagreement  : {disagreement:.4f} → Penalty: {penalty:.4f}\033[0m")
    print(f"\033[92m  Base Score    : {base_score:.4f}\033[0m")
    print(f"\033[92m  Session Bonus : {session_bonus:+.4f}\033[0m")
    print(f"\033[92m  Vol. Adjust   : {volatility_adjustment:+.4f}\033[0m")
    print(f"\033[92m  Total Adjust  : {total_adjustment:+.4f} (capped at 10% of base)\033[0m")
    print(f"\033[92m  Raw Score     : {raw_score:.4f}\033[0m")
    print(f"\033[92m  Final Score   : {final_prob:.4f}\033[0m")
    print(f"\033[92m  Thresholds    : BUY>{buy_threshold:.4f} | SELL<{sell_threshold:.4f}\033[0m")
    print(f"\033[92m  Confidence    : {decision.confidence_level}\033[0m")
    print(f"\033[92m  Reason        : {decision.decision_reason}\033[0m")
    print(f"\033[92m  ▶ DECISION    : {status}\033[0m")
    print(f"\033[92m{'='*65}\033[0m\n")

    return decision


# =========================================
# ENSEMBLE LOGGING (CSV) — FULL, NO SKIPPING
# =========================================

def _log_decision(decision, current_adx=0, current_atr=0):
    """
    Log every ensemble decision to CSV for post-analysis.
    ALL fields are mandatory — no skipping.
    """
    try:
        filepath = Config.ENSEMBLE_LOG_FILE
        file_exists = os.path.isfile(filepath)

        fieldnames = [
            "timestamp", "session", "market_state",
            "adx", "trend_strength", "atr",
            "lstm_prob", "lstm_weight",
            "rf_prob", "rf_weight",
            "weighted_avg", "disagreement", "penalty",
            "session_bonus", "volatility_adjustment",
            "raw_score", "final_score",
            "buy_threshold", "sell_threshold",
            "regime_conflict", "direction",
            "decision_reason", "confidence_level",
            "conflict", "skip_reason",
        ]

        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "session": decision.session,
                "market_state": decision.market_state,
                "adx": f"{current_adx:.2f}",
                "trend_strength": f"{decision.trend_strength:.4f}",
                "atr": f"{current_atr:.6f}",
                "lstm_prob": f"{decision.lstm_prob:.6f}",
                "lstm_weight": f"{decision.lstm_weight:.4f}",
                "rf_prob": f"{decision.rf_prob:.6f}",
                "rf_weight": f"{decision.rf_weight:.4f}",
                "weighted_avg": f"{decision.weighted_avg:.6f}",
                "disagreement": f"{abs(decision.lstm_prob - decision.rf_prob):.6f}",
                "penalty": f"{decision.penalty:.6f}",
                "session_bonus": f"{decision.session_bonus:.4f}",
                "volatility_adjustment": f"{decision.volatility_adjustment:.4f}",
                "raw_score": f"{decision.raw_score:.6f}",
                "final_score": f"{decision.final_prob:.6f}",
                "buy_threshold": f"{decision.buy_threshold:.4f}",
                "sell_threshold": f"{decision.sell_threshold:.4f}",
                "regime_conflict": decision.regime_conflict,
                "direction": decision.direction or "HOLD",
                "decision_reason": decision.decision_reason,
                "confidence_level": decision.confidence_level,
                "conflict": decision.conflict,
                "skip_reason": decision.skip_reason or "",
            })

    except Exception as e:
        logger.error("[Ensemble] Failed to log decision: %s", e)

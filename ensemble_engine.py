"""
Ensemble Engine — Elite v4.1
==============================
The Brain: Session-Aware, Market-Adaptive, Fully Traceable Decision Engine.

Architecture:
  Session Context + Market Regime + Model Ensemble = Final Decision

Features:
- Smooth Dynamic Weighting based on continuous trend_strength (NOT binary)
- Session-Aware Strategy Behavior (London/NY/Asia)
- Additive Scoring Model (Base + Session_Bonus + Volatility_Adjustment)
- Adjustment-to-Base Ratio Control (max 10% of base)
- Session-Aware Weak Zone (Asia=0.05, others=0.04)
- Symmetric Score Floor (distance < 0.015 = REJECT)
- Regime Conflict Detection (session vs trend mismatch → HOLD)
- ATR Double Filter (ratio + absolute)
- Confidence Classification (HIGH / MEDIUM / LOW)
- Decision Reason on every path (explainability)
- Full Diagnostics: side, stage_reached, distance_from_neutral, edge_case
- Comprehensive CSV Logging with all fields
"""

import numpy as np
import pandas as pd
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
    """Container for ensemble prediction results — fully explainable and traceable."""

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

        # --- v4.0 Core Fields ---
        self.session = "UNKNOWN"
        self.trend_strength = 0.0
        self.session_bonus = 0.0
        self.volatility_adjustment = 0.0
        self.regime_conflict = False
        self.decision_reason = ""
        self.confidence_level = "LOW"

        # --- v4.1 Diagnostics Fields ---
        self.distance_from_neutral = 0.0    # abs(base_score - 0.5)
        self.weak_zone_threshold_used = 0.0 # 0.04 or 0.05 (session-dependent)
        self.edge_case = False              # True if borderline decision
        self.side = "NONE"                  # Pre-filter intent: BUY/SELL/NONE
        self.stage_reached = "INIT"         # Deepest pipeline stage reached

    def __repr__(self):
        return (
            f"EnsembleDecision(LSTM={self.lstm_prob:.4f} x{self.lstm_weight:.0%}, "
            f"RF={self.rf_prob:.4f} x{self.rf_weight:.0%}, "
            f"Penalty={self.penalty:.4f}, Raw={self.raw_score:.4f}, Final={self.final_prob:.4f}, "
            f"Dir={self.direction}, Session={self.session}, Trend={self.trend_strength:.2f}, "
            f"Confidence={self.confidence_level}, Reason={self.decision_reason}, "
            f"Side={self.side}, Stage={self.stage_reached})"
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



def _compute_confidence_level(distance_from_neutral):
    """
    Compute confidence level based on distance from 0.5 (neutral).
    Used as a pre-computation for the Double Safety Gate in regime conflict override.
    """
    if distance_from_neutral > 0.15:
        return "HIGH"
    elif distance_from_neutral > 0.08:
        return "MEDIUM"
    else:
        return "LOW"


# =========================================
# REGIME CONFLICT DETECTION (Mode C — Adaptive Override)
# =========================================

def _detect_regime_conflict(session, trend_strength, distance_from_neutral=0.0,
                            atr_normalized=1.0, confidence_level="LOW"):
    """
    Detect if there's a strong conflict between session expectation and market reality.
    Mode C: Allows strong, high-confidence signals to override regime conflicts
    with a dynamic penalty instead of a hard block.

    Conflicts:
        - London (expects trend) but trend_strength < 0.15  → CONFLICT
        - Asia (expects range) but trend_strength > 0.8     → CONFLICT

    Override conditions (ALL must be met):
        - distance_from_neutral > dynamic_distance (volatility-adaptive)
        - confidence_level >= MEDIUM (double safety gate)

    Returns:
        tuple: (is_blocked: bool, regime_penalty: float)
            - (True, 0.0)    → hard block, signal is too weak for override
            - (False, 0.0)   → no conflict detected
            - (False, penalty) → conflict overridden, penalty applied
    """
    conflict_detected = False
    conflict_type = None

    if session == "London" and trend_strength < 0.15:
        conflict_detected = True
        conflict_type = "London_ranging"
    elif session == "Asia" and trend_strength > 0.8:
        conflict_detected = True
        conflict_type = "Asia_trending"

    if not conflict_detected:
        return False, 0.0  # No conflict

    # --- Adaptive Override Logic ---

    # Dynamic distance threshold: adapts to market volatility
    # Higher volatility → requires stronger signal to override
    dynamic_distance = 0.12 + (atr_normalized * 0.05)

    # Double safety gate: distance + confidence must both be sufficient
    confidence_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    confidence_score = confidence_map.get(str(confidence_level).upper(), 1)

    allow_override = (distance_from_neutral > dynamic_distance) and (confidence_score >= 2)

    if allow_override:
        # Dynamic regime penalty: scales based on how "wrong" the regime is
        if conflict_type == "London_ranging":
            regime_penalty = max(0.0, (0.15 - trend_strength) * 0.1)
        else:  # Asia_trending
            regime_penalty = 0.02  # Fixed penalty for Asia override

        logger.info(
            "[Ensemble] ✅ REGIME OVERRIDE: session=%s, ts=%.2f, dist=%.4f > dyn_dist=%.4f, "
            "conf=%s → penalty=%.4f",
            session, trend_strength, distance_from_neutral, dynamic_distance,
            confidence_level, regime_penalty
        )
        return False, regime_penalty  # Allow through with penalty

    # Hard block — signal not strong enough to override
    logger.warning(
        "[Ensemble] ⚠️ REGIME CONFLICT: session=%s, ts=%.2f, dist=%.4f, "
        "dyn_dist=%.4f, conf=%s → HARD BLOCK",
        session, trend_strength, distance_from_neutral, dynamic_distance,
        confidence_level
    )
    return True, 0.0


# =========================================
# CORE ENSEMBLE PREDICTION
# =========================================

def ensemble_predict(
    lstm_prob: float,
    rf_prob: float,
    current_adx: float,
    current_atr: float,
    atr_series: pd.Series,
    session: str = "London",
    diagnostic: bool = False,
    event_boost: float = 0.0,
    h1_trend: int = 0
) -> EnsembleDecision:
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
        decision.stage_reached = "ATR_FILTER"
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
    # Step 5: Conflict Detection (Mode C — Graduated)
    # Hard block at >= 0.60 (always)
    # Moderate zone 0.45-0.60: block only if models disagree on DIRECTION
    # Below 0.45: no conflict (proceed to penalty in Step 6)
    # =========================================
    disagreement = abs(lstm_prob - rf_prob)

    # Hard conflict: disagreement >= 0.60 → always block
    if disagreement >= 0.60:
        decision.conflict = True
        decision.direction = None
        decision.final_prob = weighted_avg
        decision.decision_reason = "CONFLICT"
        decision.stage_reached = "CONFLICT"
        decision.confidence_level = "LOW"
        decision.skip_reason = (
            f"MODEL CONFLICT: |LSTM({lstm_prob:.3f}) - RF({rf_prob:.3f})| = "
            f"{disagreement:.3f} >= 0.60 (hard block)"
        )
        logger.warning("[Ensemble] ⚠️ %s → HOLD", decision.skip_reason)
        _log_decision(decision, current_adx, current_atr)
        return decision

    # Moderate conflict zone: 0.45 <= disagreement < 0.60
    if disagreement >= 0.45:
        same_direction = (lstm_prob > 0.5 and rf_prob > 0.5) or \
                         (lstm_prob < 0.5 and rf_prob < 0.5)
        if not same_direction:
            # Opposite directions + high disagreement → hard block
            decision.conflict = True
            decision.direction = None
            decision.final_prob = weighted_avg
            decision.decision_reason = "CONFLICT"
            decision.stage_reached = "CONFLICT"
            decision.confidence_level = "LOW"
            decision.skip_reason = (
                f"MODEL CONFLICT: |LSTM({lstm_prob:.3f}) - RF({rf_prob:.3f})| = "
                f"{disagreement:.3f} >= 0.45, opposite directions → block"
            )
            logger.warning("[Ensemble] ⚠️ %s → HOLD", decision.skip_reason)
            _log_decision(decision, current_adx, current_atr)
            return decision
        else:
            # Same direction but magnitude differs → proceed with heavy penalty in Step 6
            logger.info(
                "[Ensemble] ℹ️ MODEL MODERATE CONFLICT: |LSTM(%.3f) - RF(%.3f)| = %.3f, "
                "same direction → proceed with penalty",
                lstm_prob, rf_prob, disagreement
            )

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
    # Step 7: Side Tracking + Distance Calculation
    # side = pre-filter intent based on base_score position
    # =========================================
    distance_from_neutral = abs(base_score - 0.5)
    decision.distance_from_neutral = distance_from_neutral

    if base_score > 0.5:
        decision.side = "BUY"
    elif base_score < 0.5:
        decision.side = "SELL"
    else:
        decision.side = "NONE"

    # =========================================
    # Step 8: Weak Zone Threshold (Session-Aware)
    # Phase 2.5 Adjustment: 0.03 for normal, 0.04 for Asia
    # =========================================
    weak_zone_threshold = 0.04 if session == "Asia" else 0.03
    decision.weak_zone_threshold_used = weak_zone_threshold

    # Edge case detection: borderline decisions within 0.005 of threshold
    decision.edge_case = abs(distance_from_neutral - weak_zone_threshold) < 0.005

    # =========================================
    # Step 9: Score Floor — signal too weak → REJECT
    # Applied SYMMETRICALLY: distance from 0.5 < 0.015 means no signal
    # For BUY: base_score in [0.485, 0.515] → reject
    # For SELL: same mirror range → reject
    # =========================================
    if distance_from_neutral < 0.015:
        decision.direction = None
        decision.final_prob = base_score
        decision.raw_score = base_score
        decision.decision_reason = "BELOW_THRESHOLD"
        decision.stage_reached = "SCORE_FLOOR"
        decision.confidence_level = "LOW"
        decision.skip_reason = f"SCORE FLOOR: base_score={base_score:.4f}, distance={distance_from_neutral:.4f} < 0.015"
        if not diagnostic:
            logger.info("[Ensemble] ⛔ %s → REJECT", decision.skip_reason)
        else:
            logger.info("[Ensemble] 📉 [DIAGNOSTIC HOLD] base: %.4f | dist: %.4f | ts: %.2f | reason: SCORE_FLOOR",
                        base_score, distance_from_neutral, trend_strength)
        _log_decision(decision, current_adx, current_atr)
        return decision

    # =========================================
    # Step 10: Weak Zone Hysteresis (NO ENTRY)
    # Applied SYMMETRICALLY using distance from 0.5
    # Session-aware: Asia=0.05, others=0.04
    # BUY side (0.04): base_score 0.50–0.54 | SELL side: 0.46–0.50
    # BUY side (0.05): base_score 0.50–0.55 | SELL side: 0.45–0.50
    # =========================================
    if distance_from_neutral < weak_zone_threshold:
        wz_label = "WEAK_ZONE (Asia stricter)" if session == "Asia" else "WEAK_ZONE (Normal)"
        decision.direction = None
        decision.final_prob = base_score
        decision.raw_score = base_score
        decision.decision_reason = wz_label
        decision.stage_reached = "WEAK_ZONE"
        decision.confidence_level = "LOW"
        decision.skip_reason = f"{wz_label}: base_score={base_score:.4f}, distance={distance_from_neutral:.4f} < {weak_zone_threshold}"
        if not diagnostic:
            logger.info("[Ensemble] ⚠️ %s → NO ENTRY", decision.skip_reason)
        else:
            logger.info("[Ensemble] 📉 [DIAGNOSTIC HOLD] base: %.4f | dist: %.4f | ts: %.2f | reason: WEAK_ZONE",
                        base_score, distance_from_neutral, trend_strength)
        _log_decision(decision, current_adx, current_atr)
        return decision

    # =========================================
    # Step 11: Regime Conflict (Mode C — Adaptive Override)
    # Strong signals with high confidence can pass with a dynamic penalty.
    # Weak signals in wrong regime are still hard blocked.
    # =========================================
    # Pre-compute confidence for the safety gate
    _pre_confidence = _compute_confidence_level(distance_from_neutral)

    # ATR normalization for dynamic distance threshold
    atr_mean = atr_series.mean() if atr_series is not None and len(atr_series) > 0 else current_atr
    atr_normalized = min(1.0, current_atr / atr_mean) if atr_mean > 0 else 1.0

    regime_conflict, regime_penalty = _detect_regime_conflict(
        session, trend_strength, distance_from_neutral,
        atr_normalized, _pre_confidence
    )
    decision.regime_conflict = regime_conflict

    if regime_conflict:
        decision.direction = None
        decision.final_prob = base_score
        decision.raw_score = base_score
        decision.decision_reason = "CONFLICT"
        decision.stage_reached = "CONFLICT"
        decision.confidence_level = "LOW"
        decision.skip_reason = f"REGIME CONFLICT: session={session}, trend_strength={trend_strength:.2f}"
        logger.warning("[Ensemble] ⛔ %s → HOLD", decision.skip_reason)
        _log_decision(decision, current_adx, current_atr)
        return decision

    # If regime was overridden, apply the penalty to base_score
    if regime_penalty > 0:
        if base_score > 0.5:
            base_score -= regime_penalty
        else:
            base_score += regime_penalty
        base_score = float(np.clip(base_score, 0.0, 1.0))
        logger.info("[Ensemble] Regime penalty applied: %.4f → adjusted base_score=%.4f",
                    regime_penalty, base_score)

    # =========================================
    # Step 12: Additive Scoring Model (STRICTLY additive, NO multiplication)
    # Final = Base + Session_Bonus + Volatility_Adjustment + Event_Boost - MTF_Penalty
    # =========================================

    # 12a. Session Bonus: clipped to [-0.03, +0.03]
    session_bonus = _compute_session_bonus(session, trend_strength)
    decision.session_bonus = session_bonus

    # 12b. Volatility Adjustment
    volatility_adjustment = _compute_volatility_adjustment(current_atr, atr_series)
    decision.volatility_adjustment = volatility_adjustment
    
    # 12c. Dynamic Event Boost (Phase 3)
    # Only applied in the direction of the trade
    if decision.side == "BUY" and event_boost > 0:
        actual_event_boost = min(event_boost, 0.04) # Cap boost
    elif decision.side == "SELL" and event_boost > 0:
        actual_event_boost = min(event_boost, 0.04)
    else:
        actual_event_boost = 0.0
        
    # 12d. MTF Soft Penalty (Phase 3)
    against_h1 = (decision.side == "BUY" and h1_trend == -1) or (decision.side == "SELL" and h1_trend == 1)
    mtf_penalty = -0.03 if against_h1 else 0.0

    # 12e. Total Adjustment: clipped to [-0.06, +0.06]
    raw_total_adjustment = float(np.clip(session_bonus + volatility_adjustment + actual_event_boost + mtf_penalty, -0.06, 0.06))

    # 12d. Adjustment-to-Base Ratio Control: adjustment ≤ 10% of base_score
    # This GUARANTEES boosts cannot create a trade alone
    if raw_total_adjustment > 0:
        total_adjustment = min(raw_total_adjustment, base_score * 0.1)
    else:
        total_adjustment = max(raw_total_adjustment, -(base_score * 0.1))

    # 12e. Compute raw_score and clip
    raw_score = base_score + total_adjustment
    decision.raw_score = raw_score
    final_prob = float(np.clip(raw_score, 0.0, 1.0))
    decision.final_prob = final_prob

    # =========================================
    # Step 13: Dynamic Thresholds (Phase 2.5 — Confidence Unlock)
    # =========================================
    if diagnostic:
        # Phase 1 Diagnostic Mode: Flat relaxation
        buy_threshold = 0.65
        sell_threshold = 0.35
    else:
        if trend_strength > 0.35:
            buy_threshold = 0.60
            sell_threshold = 0.40
        elif trend_strength > 0.25:
            buy_threshold = 0.62
            sell_threshold = 0.38
        else:
            buy_threshold = 0.65
            sell_threshold = 0.37  # Adjusted from 0.35 per review

    decision.buy_threshold = buy_threshold
    decision.sell_threshold = sell_threshold

    # =========================================
    # Step 14: Direction Decision + Near-Miss Activation
    # =========================================
    # Calculate distance to threshold
    if final_prob >= 0.5:
        dist_to_thresh = buy_threshold - final_prob
    else:
        dist_to_thresh = final_prob - sell_threshold

    allow_near_miss = False
    if dist_to_thresh > 0:
        is_trend = trend_strength > 0.15
        
        if is_trend and dist_to_thresh < 0.07:
            allow_near_miss = True
        elif not is_trend and dist_to_thresh < 0.03:
            allow_near_miss = True
            
        if _pre_confidence in ["MEDIUM", "HIGH"] and dist_to_thresh < 0.05:
            allow_near_miss = True

    if final_prob > buy_threshold:
        decision.direction = "BUY"
        decision.decision_reason = "VALID_SIGNAL"
        decision.stage_reached = "EXECUTION_READY"
    elif final_prob < sell_threshold:
        decision.direction = "SELL"
        decision.decision_reason = "VALID_SIGNAL"
        decision.stage_reached = "EXECUTION_READY"
    elif allow_near_miss:
        decision.direction = decision.side
        decision.decision_reason = "NEAR_MISS_ACTIVATION"
        decision.stage_reached = "EXECUTION_READY"
        logger.info("[Ensemble] 🚀 [NEAR_MISS ACTIVATED] score: %.4f | ts: %.2f | dist_to_thresh: %.4f", 
                    final_prob, trend_strength, dist_to_thresh)
    else:
        decision.direction = None
        decision.decision_reason = "BELOW_THRESHOLD"
        decision.stage_reached = "THRESHOLD_CHECK"
        decision.skip_reason = (
            f"HOLD: Final {final_prob:.4f} between "
            f"BUY>{buy_threshold:.4f} and SELL<{sell_threshold:.4f}"
        )
        
        # Diagnostics
        if diagnostic:
            _pre_conf = _compute_confidence_level(distance_from_neutral)
            if _pre_conf in ["MEDIUM", "HIGH"] and distance_from_neutral > 0.10:
                logger.info("[Ensemble] 🔍 [NEAR MISS] score: %.4f, dist: %.4f, ts: %.2f | conf: %s | gap_to_buy: %.4f, gap_to_sell: %.4f",
                            final_prob, distance_from_neutral, trend_strength, _pre_conf, 
                            buy_threshold - final_prob, final_prob - sell_threshold)
            logger.info("[Ensemble] 📉 [DIAGNOSTIC HOLD] base: %.4f | dist: %.4f | ts: %.2f | reason: THRESHOLD",
                        base_score, distance_from_neutral, trend_strength)

    # =========================================
    # Step 15: Confidence Classification
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
    # Step 16: Log the full decision
    # =========================================
    _log_decision(decision, current_adx, current_atr)

    # =========================================
    # Step 17: Console Print (human-readable)
    # =========================================
    status = decision.direction or "HOLD"
    conflict_flag = " ⚠️CONFLICT" if decision.conflict or decision.regime_conflict else ""
    edge_flag = " 🔶EDGE" if decision.edge_case else ""
    print(f"\n\033[92m{'='*65}\033[0m")
    print(f"\033[92m       🧠 ENSEMBLE DECISION v4.1{conflict_flag}{edge_flag}\033[0m")
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
    print(f"\033[92m  Distance      : {decision.distance_from_neutral:.4f} (WZ threshold: {decision.weak_zone_threshold_used})\033[0m")
    print(f"\033[92m  Side          : {decision.side}\033[0m")
    print(f"\033[92m  Stage         : {decision.stage_reached}\033[0m")
    print(f"\033[92m  Edge Case     : {decision.edge_case}\033[0m")
    print(f"\033[92m  Confidence    : {decision.confidence_level}\033[0m")
    print(f"\033[92m  Reason        : {decision.decision_reason}\033[0m")
    print(f"\033[92m  ▶ DECISION    : {status}\033[0m")
    print(f"\033[92m{'='*65}\033[0m\n")

    return decision


# =========================================
# ENSEMBLE LOGGING (CSV) — STRICT, NO SILENT FAILURES
# =========================================

# --- Valid stages (enforced at runtime) ---
VALID_STAGES = frozenset([
    "INIT", "SCORE_FLOOR", "WEAK_ZONE", "ATR_FILTER",
    "CONFLICT", "THRESHOLD_CHECK", "EXECUTION_READY",
])


def _log_decision(decision, current_adx=0, current_atr=0):
    """
    Log every ensemble decision to CSV for post-analysis.
    ALL fields are mandatory — no skipping.
    CRITICAL: Failure here raises RuntimeError — no silent failures allowed.
    """
    # --- Stage validation (MANDATORY) ---
    if decision.stage_reached not in VALID_STAGES:
        raise RuntimeError(
            f"CRITICAL: Invalid stage_reached='{decision.stage_reached}'. "
            f"Must be one of {sorted(VALID_STAGES)}"
        )

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
            "distance_from_neutral", "weak_zone_threshold_used", "edge_case",
            "buy_threshold", "sell_threshold",
            "regime_conflict", "direction", "side", "stage_reached",
            "decision_reason", "confidence_level",
            "conflict", "skip_reason",
        ]

        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
                "distance_from_neutral": f"{decision.distance_from_neutral:.6f}",
                "weak_zone_threshold_used": f"{decision.weak_zone_threshold_used:.4f}",
                "edge_case": decision.edge_case,
                "buy_threshold": f"{decision.buy_threshold:.4f}",
                "sell_threshold": f"{decision.sell_threshold:.4f}",
                "regime_conflict": decision.regime_conflict,
                "direction": decision.direction or "HOLD",
                "side": decision.side,
                "stage_reached": decision.stage_reached,
                "decision_reason": decision.decision_reason,
                "confidence_level": decision.confidence_level,
                "conflict": decision.conflict,
                "skip_reason": decision.skip_reason or "",
            })

        # Update runtime metrics
        _metrics.record(decision)

    except Exception as e:
        raise RuntimeError(f"CRITICAL: Logging failed — stopping system. Error: {e}") from e


# =========================================
# RUNTIME METRICS TRACKER — Stage Distribution
# =========================================

class DecisionMetrics:
    """Tracks runtime decision distribution for live observability."""

    def __init__(self):
        self.total_signals = 0
        self.stage_counts = {s: 0 for s in VALID_STAGES}
        self.reason_counts = {}
        self.side_counts = {"BUY": 0, "SELL": 0, "NONE": 0}

    def record(self, decision):
        self.total_signals += 1
        if decision.stage_reached in self.stage_counts:
            self.stage_counts[decision.stage_reached] += 1
        reason = decision.decision_reason
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        if decision.side in self.side_counts:
            self.side_counts[decision.side] += 1

    @property
    def hold_count(self):
        return self.total_signals - self.stage_counts.get("EXECUTION_READY", 0)

    @property
    def execution_ready_rate(self):
        if self.total_signals == 0:
            return 0.0
        return self.stage_counts.get("EXECUTION_READY", 0) / self.total_signals * 100

    def print_summary(self):
        print(f"\n\033[96m{'='*60}\033[0m")
        print(f"\033[96m       📊 ENSEMBLE RUNTIME METRICS\033[0m")
        print(f"\033[96m{'='*60}\033[0m")
        print(f"\033[96m  TOTAL_SIGNALS     : {self.total_signals}\033[0m")
        print(f"\033[96m  HOLD_COUNT        : {self.hold_count}\033[0m")
        print(f"\033[96m  EXECUTION_READY   : {self.stage_counts.get('EXECUTION_READY', 0)} ({self.execution_ready_rate:.1f}%)\033[0m")
        print(f"\033[96m  --- Stage Breakdown ---\033[0m")
        for stage in ["SCORE_FLOOR", "WEAK_ZONE", "ATR_FILTER", "CONFLICT", "THRESHOLD_CHECK", "EXECUTION_READY"]:
            cnt = self.stage_counts.get(stage, 0)
            print(f"\033[96m    {stage:20s}: {cnt}\033[0m")
        print(f"\033[96m  --- Reason Breakdown ---\033[0m")
        for reason, cnt in sorted(self.reason_counts.items(), key=lambda x: -x[1]):
            print(f"\033[96m    {reason:25s}: {cnt}\033[0m")
        print(f"\033[96m  --- Side Distribution ---\033[0m")
        for side, cnt in self.side_counts.items():
            print(f"\033[96m    {side:10s}: {cnt}\033[0m")
        print(f"\033[96m{'='*60}\033[0m\n")

    def reset(self):
        self.__init__()


# Global metrics instance
_metrics = DecisionMetrics()


def get_metrics():
    """Return the global metrics tracker for external access."""
    return _metrics

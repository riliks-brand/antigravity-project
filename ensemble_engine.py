"""
Ensemble Engine â€” Elite v5.0
==============================
XGBoost Edition â€” Ø§Ù„ØªØºÙŠÙŠØ±Ø§Øª Ø¹Ù† v4.2:

Ø§Ù„Ù…Ø³Ø§Ø± C â€” Hybrid: XGBoost Ø¨Ø¯Ù„ LSTM:
  - LSTM accuracy = 50% = noise
  - XGBoost Ø£Ø«Ø¨Øª Ù†ÙØ³Ù‡ ÙÙŠ financial time series Ø¨Ø´ÙƒÙ„ Ù…ØªÙƒØ±Ø±
  - Gradient boosting Ø¹Ù„Ù‰ features Ù…Ù‡Ù†Ø¯Ø³Ø© > sequence learning Ø¹Ù„Ù‰ Ø¨ÙŠØ§Ù†Ø§Øª Ù‚Ù„ÙŠÙ„Ø©

Ø§Ù„ØªØºÙŠÙŠØ±Ø§Øª ÙÙŠ v5.0:
  1. Ø§Ø³ØªØ¨Ø¯Ø§Ù„ LSTM Ø¨Ù€ XGBoost ÙÙŠ Ø§Ù„Ù€ ensemble
  2. XGB-RF ensemble Ø¨Ø¯Ù„ LSTM-RF
  3. Ø§Ù„Ù€ weights Ù…Ø­Ø³ÙˆØ¨Ø© Ø¹Ù„Ù‰ Ù†ÙØ³ Ø§Ù„Ù…Ù†Ø·Ù‚: XGB Ø£Ø³Ø§Ø³ØŒ RF Ù…ÙƒÙ…Ù‘Ù„
     XGB Ø¨ÙŠØ´ÙˆÙ lagged features + tabular context
     RF Ø¨ÙŠØ´ÙˆÙ cross-product interactions + rolling stats
     Ø§Ù„Ø§ØªÙ†ÙŠÙ† Ø¨ÙŠÙƒÙ…Ù„ÙˆØ§ Ø¨Ø¹Ø¶ Ø¨Ø¯Ù„ Ù…Ø§ ÙŠÙƒÙˆÙ†ÙˆØ§ redundant
  4. RF Confidence Gate Ù…Ø­Ø§ÙØ¸ Ø¹Ù„ÙŠÙ‡ (Ø£Ø«Ø¨Øª ÙØ¹Ø§Ù„ÙŠØªÙ‡)
  5. ØªØ­Ø¯ÙŠØ« Ø§Ù„Ù€ labels ÙÙŠ Ø§Ù„Ù€ logs Ù…Ù† LSTM â†’ XGB
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
    """Container for ensemble prediction results â€” fully explainable and traceable."""

    def __init__(self):
        self.xgb_prob = 0.5
        self.rf_prob = 0.5
        self.weighted_avg = 0.5
        self.penalty = 0.0
        self.raw_score = 0.5
        self.final_prob = 0.5
        self.direction = None
        self.skip_reason = None
        self.xgb_weight = 0.55
        self.rf_weight = 0.45
        self.buy_threshold = 0.60
        self.sell_threshold = 0.40
        self.market_state = "UNKNOWN"
        self.conflict = False

        self.session = "UNKNOWN"
        self.trend_strength = 0.0
        self.session_bonus = 0.0
        self.volatility_adjustment = 0.0
        self.dxy_influence = 0.0
        self.regime_conflict = False
        self.decision_reason = ""
        self.confidence_level = "LOW"

        self.distance_from_neutral = 0.0
        self.weak_zone_threshold_used = 0.0
        self.edge_case = False
        self.side = "NONE"
        self.stage_reached = "INIT"

    def __repr__(self):
        return (
            f"EnsembleDecision(XGB={self.xgb_prob:.4f} x{self.xgb_weight:.0%}, "
            f"RF={self.rf_prob:.4f} x{self.rf_weight:.0%}, "
            f"Penalty={self.penalty:.4f}, Raw={self.raw_score:.4f}, Final={self.final_prob:.4f}, "
            f"Dir={self.direction}, Session={self.session}, Trend={self.trend_strength:.2f}, "
            f"Confidence={self.confidence_level}, Reason={self.decision_reason}, "
            f"Side={self.side}, Stage={self.stage_reached})"
        )


# =========================================
# v4.2: RF-FIRST DYNAMIC WEIGHTING
# RF Ù‡Ùˆ Ø§Ù„Ø£Ø³Ø§Ø³ Ø¯Ø§ÙŠÙ…Ø§Ù‹ â€” LSTM modifier Ø¨Ø³
# =========================================

def get_dynamic_weights(trend_strength, session):
    """
    v5.0: XGB-RF dynamic weighting.

    XGBoost Ø¨ÙŠØ´ÙˆÙ lagged + tabular features â€” Ø£Ù‚ÙˆÙ‰ ÙÙŠ capturing non-linear patterns
    RF Ø¨ÙŠØ´ÙˆÙ interaction features â€” complementary perspective
    Ø§Ù„Ø§ØªÙ†ÙŠÙ† Ø¨ÙŠÙƒÙ…Ù„ÙˆØ§ Ø¨Ø¹Ø¶ ÙÙŠ Ø§Ù„Ù€ ensemble.

    XGB accuracy Ù…ØªÙˆÙ‚Ø¹ 55-60% (Ø£Ø­Ø³Ù† Ù…Ù† LSTM Ø¨Ø´ÙƒÙ„ ÙˆØ§Ø¶Ø­)
    Ù„Ø°Ù„Ùƒ XGB ÙŠØ¨Ù‚Ù‰ Ø§Ù„Ù€ primary model:
        trend_strength=0 â†’ XGB=55%, RF=45%  (ranging: RF context Ø£Ù‡Ù…)
        trend_strength=1 â†’ XGB=65%, RF=35%  (trending: XGB lagged features Ø£ÙƒØªØ± ÙØ§Ø¦Ø¯Ø©)
    """
    xgb_w = 0.55 + (trend_strength * 0.10)
    rf_w  = 0.45 - (trend_strength * 0.10)

    # Clamp
    xgb_w = float(np.clip(xgb_w, 0.50, 0.70))
    rf_w  = float(np.clip(rf_w,  0.30, 0.50))

    # Re-normalize
    total = xgb_w + rf_w
    xgb_w /= total
    rf_w  /= total

    if trend_strength >= 0.6:
        state = "TRENDING"
    elif trend_strength <= 0.2:
        state = "RANGING"
    else:
        state = "TRANSITIONING"

    logger.debug(
        "[Ensemble v5.0] trend_strength=%.3f -> %s | XGB=%.1f%% RF=%.1f%% | Session=%s",
        trend_strength, state, xgb_w * 100, rf_w * 100, session
    )
    return xgb_w, rf_w, state


# =========================================
# SESSION BONUS â€” unchanged from v4.1
# =========================================

def _compute_session_bonus(session, trend_strength):
    raw_bonus = 0.0
    if session == "London":
        if trend_strength >= 0.5:
            raw_bonus = 0.02
        elif trend_strength >= 0.3:
            raw_bonus = 0.01
        else:
            raw_bonus = -0.01
    elif session == "New York":
        raw_bonus = 0.01 if trend_strength >= 0.4 else 0.0
    elif session == "Asia":
        if trend_strength <= 0.3:
            raw_bonus = 0.02
        elif trend_strength <= 0.5:
            raw_bonus = 0.01
        else:
            raw_bonus = -0.01
    return float(np.clip(raw_bonus, -0.03, 0.03))


# =========================================
# VOLATILITY ADJUSTMENT â€” unchanged from v4.1
# =========================================

def _compute_volatility_adjustment(current_atr, atr_series):
    if atr_series is None or len(atr_series) < 20:
        return 0.0
    atr_mean = atr_series.mean()
    atr_std  = atr_series.std()
    if atr_std <= 0 or atr_mean <= 0:
        return 0.0
    z_score = (current_atr - atr_mean) / atr_std
    return float(np.clip(z_score * 0.01, -0.02, 0.02))


def _compute_confidence_level(distance_from_neutral):
    if distance_from_neutral > 0.15:
        return "HIGH"
    elif distance_from_neutral > 0.08:
        return "MEDIUM"
    else:
        return "LOW"


# =========================================
# REGIME CONFLICT â€” unchanged from v4.1
# =========================================

def _detect_regime_conflict(session, trend_strength, distance_from_neutral=0.0,
                            atr_normalized=1.0, confidence_level="LOW"):
    conflict_detected = False
    conflict_type = None

    if session == "London" and trend_strength < 0.02:
        conflict_detected = True
        conflict_type = "London_ranging"
    elif session == "Asia" and trend_strength > 0.8:
        conflict_detected = True
        conflict_type = "Asia_trending"

    if not conflict_detected:
        return False, 0.0

    dynamic_distance = 0.05 + (atr_normalized * 0.02)
    confidence_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    confidence_score = confidence_map.get(str(confidence_level).upper(), 1)
    
    # Make override easier (only need LOW confidence or lower distance)
    allow_override = (distance_from_neutral > dynamic_distance * 0.5) or (confidence_score >= 2)

    if allow_override:
        regime_penalty = max(0.0, (0.15 - trend_strength) * 0.1) if conflict_type == "London_ranging" else 0.02
        logger.info("[Ensemble v5.0] [REGIME_OVERRIDE] penalty=%.4f", regime_penalty)
        return False, regime_penalty

    logger.warning("[Ensemble v5.0] [REGIME_CONFLICT] session=%s, ts=%.2f -> HARD BLOCK", session, trend_strength)
    return True, 0.0


# =========================================
# v4.2: RF CONFIDENCE GATE
# Ù„Ùˆ RF ÙÙŠ Ø§Ù„Ù€ noise zone â†’ HOLD Ù…Ø¨Ø§Ø´Ø±Ø©
# Ø¯Ù‡ Ø¨ÙŠÙˆÙØ± ÙˆÙ‚Øª ÙˆÙŠÙ‚Ù„Ù„ Ø§Ù„Ù€ false signals
# =========================================

# Ø§Ù„Ù€ noise zone Ø¨Ù†Ø§Ø¡Ù‹ Ø¹Ù„Ù‰ RF distribution Ø§Ù„ÙØ¹Ù„ÙŠØ©:
# NOISE (0.43-0.57): ~65% Ù…Ù† Ø§Ù„ÙˆÙ‚Øª
# Signal ÙŠØ¨Ø¯Ø£ Ù…Ù† 0.60+ Ù„Ù„Ù€ BUY Ø£Ùˆ 0.40- Ù„Ù„Ù€ SELL
RF_NOISE_UPPER = 0.58   # ÙÙˆÙ‚ Ø¯Ù‡ = RF Ø¨ÙŠÙ‚ÙˆÙ„ BUY Ø¨Ø«Ù‚Ø©  (ÙƒØ§Ù† 0.55)
RF_NOISE_LOWER = 0.42   # ØªØ­Øª Ø¯Ù‡ = RF Ø¨ÙŠÙ‚ÙˆÙ„ SELL Ø¨Ø«Ù‚Ø© (ÙƒØ§Ù† 0.45)


def _rf_confidence_gate(rf_prob, xgb_prob, diagnostic=False):
    """
    v5.0 Gate: Ù„Ùˆ RF ÙÙŠ Ø§Ù„Ù€ noise zone â†’ return False (HOLD)
    Ù„Ùˆ RF Ø®Ø§Ø±Ø¬ Ø§Ù„Ù€ noise zone â†’ return True (Ù…ØªØ§Ø¨Ø¹Ø©)
    Ù„Ùˆ XGBoost Ù‚ÙˆÙŠ Ø¬Ø¯Ø§Ù‹ ÙŠØªÙ… ØªØ¬Ø§ÙˆØ² Ø§Ù„Ù€ gate.
    """
    in_noise = RF_NOISE_LOWER < rf_prob < RF_NOISE_UPPER  # exclusive boundaries
    # v5.1: XGB override threshold lowered to match new buy_threshold (0.56)
    xgb_confident = xgb_prob > 0.57 or xgb_prob < 0.43
    
    if in_noise and not xgb_confident and not diagnostic:
        logger.info(
            "[Ensemble v5.0] [RF_NOISE_GATE] rf_prob=%.4f in noise zone [%.2f, %.2f] -> HOLD",
            rf_prob, RF_NOISE_LOWER, RF_NOISE_UPPER
        )
        return False
    return True


# =========================================
# CORE ENSEMBLE PREDICTION v4.2
# =========================================

def ensemble_predict(
    xgb_prob: float,
    rf_prob: float,
    current_adx: float,
    current_atr: float,
    atr_series: pd.Series,
    session: str = "London",
    diagnostic: bool = False,
    event_boost: float = 0.0,
    h1_trend: int = 0,
    dxy_strength: float = 0.0,
    symbol: str = "EURUSD",
    # backward compat: accept lstm_prob as alias
    lstm_prob: float = None,
) -> EnsembleDecision:

    # Backward compatibility: if old code passes lstm_prob, use it as xgb_prob
    if lstm_prob is not None and xgb_prob == 0.5:
        xgb_prob = lstm_prob

    decision = EnsembleDecision()
    decision.xgb_prob = xgb_prob
    decision.rf_prob = rf_prob
    decision.session = session

    # â”€â”€ Step 1: Trend Strength â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    trend_strength = float(np.clip((current_adx - 15) / 35.0, 0.0, 1.0))  # v5.1: smoother gradient, starts at ADX=15
    decision.trend_strength = trend_strength

    # â”€â”€ Step 2: ATR Double Filter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    atr_mean = atr_series.mean() if atr_series is not None and len(atr_series) > 0 else 0
    atr_ratio = (current_atr / atr_mean) if atr_mean > 0 else 1.0

    if atr_ratio < 0.5 or current_atr < Config.ATR_THRESHOLD:
        decision.direction = None
        decision.decision_reason = "LOW_ATR"
        decision.stage_reached = "ATR_FILTER"
        decision.final_prob = 0.5
        decision.confidence_level = "LOW"
        decision.market_state = "LOW_VOLATILITY"
        logger.warning("[Ensemble v5.0] [LOW_ATR] ratio=%.3f, atr=%.6f -> SKIP", atr_ratio, current_atr)
        _log_decision(decision, current_adx, current_atr)
        return decision

    # â”€â”€ Step 2.5: RF Confidence Gate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not _rf_confidence_gate(rf_prob, xgb_prob, diagnostic):
        decision.direction = None
        decision.decision_reason = "RF_NOISE_ZONE"
        decision.stage_reached = "SCORE_FLOOR"
        decision.final_prob = 0.5
        decision.confidence_level = "LOW"
        decision.market_state = "RF_NOISE"
        decision.skip_reason = f"RF_NOISE_GATE: rf_prob={rf_prob:.4f} in [{RF_NOISE_LOWER}, {RF_NOISE_UPPER}]"
        _log_decision(decision, current_adx, current_atr)
        return decision

    # â”€â”€ Step 3: v5.0 XGB-RF Dynamic Weights â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    xgb_w, rf_w, market_state = get_dynamic_weights(trend_strength, session)
    decision.xgb_weight = xgb_w
    decision.rf_weight = rf_w
    decision.market_state = market_state

    # â”€â”€ Step 4: Weighted Average â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    weighted_avg = (xgb_w * xgb_prob) + (rf_w * rf_prob)
    decision.weighted_avg = weighted_avg

    # â”€â”€ Step 5: Conflict Detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # XGB Ø£Ø®Ø° Ù…ÙƒØ§Ù† LSTM ÙƒÙ€ primary signal
    disagreement = abs(xgb_prob - rf_prob)

    rf_says_buy   = rf_prob  > RF_NOISE_UPPER
    rf_says_sell  = rf_prob  < RF_NOISE_LOWER
    xgb_says_buy  = xgb_prob > 0.60
    xgb_says_sell = xgb_prob < 0.40

    # Hard conflict: XGB Ùˆ RF ÙÙŠ Ø§ØªØ¬Ø§Ù‡ÙŠÙ† Ù…Ø®ØªÙ„ÙÙŠÙ† Ø¨Ø«Ù‚Ø©
    if (rf_says_buy and xgb_says_sell) or (rf_says_sell and xgb_says_buy):
        decision.conflict = True
        decision.direction = None
        decision.final_prob = weighted_avg
        decision.decision_reason = "CONFLICT"
        decision.stage_reached = "CONFLICT"
        decision.confidence_level = "LOW"
        decision.skip_reason = (
            f"XGB_RF_CONFLICT: RF={rf_prob:.3f} vs XGB={xgb_prob:.3f} â€” opposite directions"
        )
        logger.warning("[Ensemble v5.0] âš ï¸ %s -> HOLD", decision.skip_reason)
        _log_decision(decision, current_adx, current_atr)
        return decision

    if disagreement >= 0.60:
        decision.conflict = True
        decision.direction = None
        decision.final_prob = weighted_avg
        decision.decision_reason = "CONFLICT"
        decision.stage_reached = "CONFLICT"
        decision.confidence_level = "LOW"
        decision.skip_reason = f"HIGH_DISAGREEMENT: |XGB-RF|={disagreement:.3f} >= 0.60"
        logger.warning("[Ensemble v5.0] âš ï¸ %s -> HOLD", decision.skip_reason)
        _log_decision(decision, current_adx, current_atr)
        return decision

    # â”€â”€ Step 6: Disagreement Penalty â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    penalty = disagreement * Config.ENSEMBLE_DISAGREEMENT_PENALTY
    decision.penalty = penalty

    if weighted_avg > 0.5:
        base_score = weighted_avg - penalty
    else:
        base_score = weighted_avg + penalty
    base_score = float(np.clip(base_score, 0.0, 1.0))

    # â”€â”€ Step 7: Side Tracking + Distance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    distance_from_neutral = abs(base_score - 0.5)
    decision.distance_from_neutral = distance_from_neutral

    if base_score > 0.5:
        decision.side = "BUY"
    elif base_score < 0.5:
        decision.side = "SELL"
    else:
        decision.side = "NONE"

    # â”€â”€ Step 8: Weak Zone Threshold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    weak_zone_threshold = 0.02 if session == "Asia" else 0.01
    decision.weak_zone_threshold_used = weak_zone_threshold
    decision.edge_case = abs(distance_from_neutral - weak_zone_threshold) < 0.005

    # â”€â”€ Step 9: Score Floor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if distance_from_neutral < 0.002:
        decision.direction = None
        decision.final_prob = base_score
        decision.raw_score = base_score
        decision.decision_reason = "BELOW_THRESHOLD"
        decision.stage_reached = "SCORE_FLOOR"
        decision.confidence_level = "LOW"
        decision.skip_reason = f"SCORE FLOOR: base_score={base_score:.4f}"
        _log_decision(decision, current_adx, current_atr)
        return decision

    # â”€â”€ Step 10: Weak Zone â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if distance_from_neutral < weak_zone_threshold:
        wz_label = "WEAK_ZONE (Asia stricter)" if session == "Asia" else "WEAK_ZONE (Normal)"
        decision.direction = None
        decision.final_prob = base_score
        decision.raw_score = base_score
        decision.decision_reason = wz_label
        decision.stage_reached = "WEAK_ZONE"
        decision.confidence_level = "LOW"
        decision.skip_reason = f"{wz_label}: distance={distance_from_neutral:.4f}"
        if not diagnostic:
            logger.info("[Ensemble v4.2] âš ï¸ %s -> NO ENTRY", decision.skip_reason)
        _log_decision(decision, current_adx, current_atr)
        return decision

    # â”€â”€ Step 11: Regime Conflict â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _pre_confidence = _compute_confidence_level(distance_from_neutral)
    atr_normalized = min(1.0, current_atr / atr_mean) if atr_mean > 0 else 1.0

    regime_conflict, regime_penalty = _detect_regime_conflict(
        session, trend_strength, distance_from_neutral, atr_normalized, _pre_confidence
    )
    decision.regime_conflict = regime_conflict

    if regime_conflict:
        decision.direction = None
        decision.final_prob = base_score
        decision.raw_score = base_score
        decision.decision_reason = "CONFLICT"
        decision.stage_reached = "CONFLICT"
        decision.confidence_level = "LOW"
        decision.skip_reason = f"REGIME CONFLICT: session={session}, ts={trend_strength:.2f}"
        logger.warning("[Ensemble v4.2] â›” %s -> HOLD", decision.skip_reason)
        _log_decision(decision, current_adx, current_atr)
        return decision

    if regime_penalty > 0:
        base_score = base_score - regime_penalty if base_score > 0.5 else base_score + regime_penalty
        base_score = float(np.clip(base_score, 0.0, 1.0))

    # â”€â”€ Step 12: Additive Scoring â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    session_bonus = _compute_session_bonus(session, trend_strength)
    decision.session_bonus = session_bonus

    volatility_adjustment = _compute_volatility_adjustment(current_atr, atr_series)
    decision.volatility_adjustment = volatility_adjustment

    # Event boost
    if decision.side in ("BUY", "SELL") and event_boost > 0:
        actual_event_boost = min(event_boost, 0.04)
    else:
        actual_event_boost = 0.0

    # MTF penalty
    against_h1 = (decision.side == "BUY" and h1_trend == -1) or (decision.side == "SELL" and h1_trend == 1)
    mtf_penalty = -0.03 if against_h1 else 0.0

    # DXY influence
    dxy_influence = 0.0
    usd_sensitive_pairs = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY"]
    if symbol in usd_sensitive_pairs and abs(dxy_strength) >= 0.2:
        raw_dxy_mod = abs(dxy_strength) * 0.02
        if symbol == "USDJPY":
            if decision.side == "BUY":
                dxy_influence = raw_dxy_mod if dxy_strength > 0 else -raw_dxy_mod
            elif decision.side == "SELL":
                dxy_influence = raw_dxy_mod if dxy_strength < 0 else -raw_dxy_mod
        else:
            if decision.side == "BUY":
                dxy_influence = raw_dxy_mod if dxy_strength < 0 else -raw_dxy_mod
            elif decision.side == "SELL":
                dxy_influence = raw_dxy_mod if dxy_strength > 0 else -raw_dxy_mod
    decision.dxy_influence = dxy_influence

    raw_total_adjustment = float(np.clip(
        session_bonus + volatility_adjustment + actual_event_boost + mtf_penalty + dxy_influence,
        -0.04, 0.04
    ))

    if raw_total_adjustment > 0:
        total_adjustment = min(raw_total_adjustment, base_score * 0.08)
    else:
        total_adjustment = max(raw_total_adjustment, -(base_score * 0.08))

    raw_score = base_score + total_adjustment
    decision.raw_score = raw_score
    final_prob = float(np.clip(raw_score, 0.0, 1.0))
    decision.final_prob = final_prob

    # â”€â”€ Step 13: v5.1 Thresholds â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # v5.0: BUY > 0.60-0.62 â†’ ÙƒØ§Ù† ØµØ¹Ø¨ Ø¬Ø¯Ø§Ù‹ ÙŠØªØ­Ù‚Ù‚ (XGB Ø¨ÙŠØ·Ù„Ø¹ 0.55-0.62 Ø¹Ø§Ø¯Ø©Ù‹)
    # v5.1: Ø®ÙØ¶Ù†Ø§ Ø§Ù„Ù€ threshold Ø¨Ù†Ø§Ø¡Ù‹ Ø¹Ù„Ù‰ Ø§Ù„ØªØ­Ù„ÙŠÙ„ Ø§Ù„ÙØ¹Ù„ÙŠ Ù„Ù„Ù€ ensemble_decisions.csv
    #       Ù…ØªÙˆØ³Ø· final_score Ù„Ù„Ù€ THRESHOLD_CHECK = 0.52ØŒ Ø§Ù„Ù€ gap = 0.11
    #       Ø§Ù„Ø­Ù„: Ù†Ø®ÙØ¶ Ø§Ù„Ù€ threshold Ø¨Ù€ 0.05 Ø¹Ø´Ø§Ù† Ù†Ù…Ø³Ùƒ Ø§Ù„Ù€ signals Ø§Ù„Ø­Ù‚ÙŠÙ‚ÙŠØ©
    if diagnostic:
        buy_threshold  = 0.62
        sell_threshold = 0.38
    else:
        if trend_strength > 0.35:
            buy_threshold  = 0.56   # ÙƒØ§Ù† 0.60 â€” Ø®ÙØ¶Ù†Ø§ 0.04
            sell_threshold = 0.44   # ÙƒØ§Ù† 0.40
        elif trend_strength > 0.25:
            buy_threshold  = 0.57   # ÙƒØ§Ù† 0.61
            sell_threshold = 0.43   # ÙƒØ§Ù† 0.39
        else:
            buy_threshold  = 0.58   # ÙƒØ§Ù† 0.62
            sell_threshold = 0.42   # ÙƒØ§Ù† 0.38

    decision.buy_threshold  = buy_threshold
    decision.sell_threshold = sell_threshold

    if final_prob > buy_threshold:
        decision.direction = "BUY"
        decision.decision_reason = "VALID_SIGNAL"
        decision.stage_reached = "EXECUTION_READY"
    elif final_prob < sell_threshold:
        decision.direction = "SELL"
        decision.decision_reason = "VALID_SIGNAL"
        decision.stage_reached = "EXECUTION_READY"
    else:
        decision.direction = None
        decision.decision_reason = "BELOW_THRESHOLD"
        decision.stage_reached = "THRESHOLD_CHECK"
        decision.skip_reason = (
            f"HOLD: Final {final_prob:.4f} between "
            f"BUY>{buy_threshold:.4f} and SELL<{sell_threshold:.4f}"
        )
        if diagnostic:
            _pre_conf = _compute_confidence_level(distance_from_neutral)
            if _pre_conf in ["MEDIUM", "HIGH"] and distance_from_neutral > 0.10:
                logger.info(
                    "[Ensemble v4.2] ðŸ” [NEAR MISS] score:%.4f dist:%.4f ts:%.2f conf:%s",
                    final_prob, distance_from_neutral, trend_strength, _pre_conf
                )

    # â”€â”€ Step 15: Confidence Classification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    distance_from_center = abs(final_prob - 0.5)
    if distance_from_center > 0.15:
        decision.confidence_level = "HIGH"
    elif distance_from_center > 0.08:
        decision.confidence_level = "MEDIUM"
    else:
        decision.confidence_level = "LOW"

    # â”€â”€ Step 16: Log â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _log_decision(decision, current_adx, current_atr)

    # â”€â”€ Step 17: Console Print â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    status = decision.direction or "HOLD"
    conflict_flag = " [CONFLICT]" if decision.conflict or decision.regime_conflict else ""
    edge_flag = " [EDGE]" if decision.edge_case else ""
    print(f"\n\033[92m{'='*65}\033[0m")
    print(f"\033[92m        ENSEMBLE DECISION v5.0 XGB-RF{conflict_flag}{edge_flag}\033[0m")
    print(f"\033[92m{'='*65}\033[0m")
    print(f"\033[92m  Session       : {session}\033[0m")
    print(f"\033[92m  Market State  : {market_state} (ADX: {current_adx:.1f} -> ts: {trend_strength:.3f})\033[0m")
    print(f"\033[92m  XGBoost       : {xgb_prob:.4f} (weight: {xgb_w:.0%}) <- PRIMARY\033[0m")
    print(f"\033[92m  RF            : {rf_prob:.4f} (weight: {rf_w:.0%}) <- COMPLEMENT\033[0m")
    print(f"\033[92m  RF Gate       : {'PASS' if _rf_confidence_gate(rf_prob, xgb_prob, True) else 'NOISE'}\033[0m")
    print(f"\033[92m  Weighted Avg  : {weighted_avg:.4f}\033[0m")
    print(f"\033[92m  Disagreement  : {disagreement:.4f} -> Penalty: {decision.penalty:.4f}\033[0m")
    print(f"\033[92m  Base Score    : {base_score:.4f}\033[0m")
    print(f"\033[92m  Session Bonus : {session_bonus:+.4f}\033[0m")
    print(f"\033[92m  Vol. Adjust   : {volatility_adjustment:+.4f}\033[0m")
    print(f"\033[92m  Final Score   : {final_prob:.4f}\033[0m")
    print(f"\033[92m  Thresholds    : BUY>{buy_threshold:.4f} | SELL<{sell_threshold:.4f}\033[0m")
    print(f"\033[92m  Confidence    : {decision.confidence_level}\033[0m")
    print(f"\033[92m  Reason        : {decision.decision_reason}\033[0m")
    print(f"\033[92m  >> DECISION    : {status}\033[0m")
    print(f"\033[92m{'='*65}\033[0m\n")

    return decision


# =========================================
# ENSEMBLE LOGGING (CSV)
# =========================================

VALID_STAGES = frozenset([
    "INIT", "SCORE_FLOOR", "WEAK_ZONE", "ATR_FILTER",
    "CONFLICT", "THRESHOLD_CHECK", "EXECUTION_READY",
])


def _log_decision(decision, current_adx=0, current_atr=0):
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
            "xgb_prob", "xgb_weight",
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
                "xgb_prob": f"{decision.xgb_prob:.6f}",
                "xgb_weight": f"{decision.xgb_weight:.4f}",
                "rf_prob": f"{decision.rf_prob:.6f}",
                "rf_weight": f"{decision.rf_weight:.4f}",
                "weighted_avg": f"{decision.weighted_avg:.6f}",
                "disagreement": f"{abs(decision.xgb_prob - decision.rf_prob):.6f}",
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

        _metrics.record(decision)

    except Exception as e:
        raise RuntimeError(f"CRITICAL: Logging failed: {e}") from e


# =========================================
# RUNTIME METRICS TRACKER
# =========================================

class DecisionMetrics:
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
        print(f"\033[96m       [METRICS] ENSEMBLE RUNTIME METRICS v5.0\033[0m")
        print(f"\033[96m{'='*60}\033[0m")
        print(f"\033[96m  TOTAL_SIGNALS     : {self.total_signals}\033[0m")
        print(f"\033[96m  HOLD_COUNT        : {self.hold_count}\033[0m")
        print(f"\033[96m  EXECUTION_READY   : {self.stage_counts.get('EXECUTION_READY', 0)} ({self.execution_ready_rate:.1f}%)\033[0m")
        for stage in ["SCORE_FLOOR", "WEAK_ZONE", "ATR_FILTER", "CONFLICT", "THRESHOLD_CHECK", "EXECUTION_READY"]:
            cnt = self.stage_counts.get(stage, 0)
            print(f"\033[96m    {stage:20s}: {cnt}\033[0m")
        for reason, cnt in sorted(self.reason_counts.items(), key=lambda x: -x[1]):
            print(f"\033[96m    {reason:25s}: {cnt}\033[0m")
        print(f"\033[96m{'='*60}\033[0m\n")

    def reset(self):
        self.__init__()


_metrics = DecisionMetrics()


def get_metrics():
    return _metrics





"""
Ensemble Engine — Elite v5.0
==============================
XGBoost Edition — التغييرات عن v4.2:

المسار C — Hybrid: XGBoost بدل LSTM:
  - LSTM accuracy = 50% = noise
  - XGBoost أثبت نفسه في financial time series بشكل متكرر
  - Gradient boosting على features مهندسة > sequence learning على بيانات قليلة

التغييرات في v5.0:
  1. استبدال LSTM بـ XGBoost في الـ ensemble
  2. XGB-RF ensemble بدل LSTM-RF
  3. الـ weights محسوبة على نفس المنطق: XGB أساس، RF مكمّل
     XGB بيشوف lagged features + tabular context
     RF بيشوف cross-product interactions + rolling stats
     الاتنين بيكملوا بعض بدل ما يكونوا redundant
  4. RF Confidence Gate محافظ عليه (أثبت فعاليته)
  5. تحديث الـ labels في الـ logs من LSTM → XGB
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
# RF هو الأساس دايماً — LSTM modifier بس
# =========================================

def get_dynamic_weights(trend_strength, session):
    """
    v5.0: XGB-RF dynamic weighting.

    XGBoost بيشوف lagged + tabular features — أقوى في capturing non-linear patterns
    RF بيشوف interaction features — complementary perspective
    الاتنين بيكملوا بعض في الـ ensemble.

    XGB accuracy متوقع 55-60% (أحسن من LSTM بشكل واضح)
    لذلك XGB يبقى الـ primary model:
        trend_strength=0 → XGB=55%, RF=45%  (ranging: RF context أهم)
        trend_strength=1 → XGB=65%, RF=35%  (trending: XGB lagged features أكتر فائدة)
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
# SESSION BONUS — unchanged from v4.1
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
# VOLATILITY ADJUSTMENT — unchanged from v4.1
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
# REGIME CONFLICT — unchanged from v4.1
# =========================================

def _detect_regime_conflict(session, trend_strength, distance_from_neutral=0.0,
                            atr_normalized=1.0, confidence_level="LOW"):
    conflict_detected = False
    conflict_type = None

    if session == "London" and trend_strength < 0.05:
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
    allow_override = (distance_from_neutral > dynamic_distance) and (confidence_score >= 2)

    if allow_override:
        regime_penalty = max(0.0, (0.15 - trend_strength) * 0.1) if conflict_type == "London_ranging" else 0.02
        logger.info("[Ensemble v4.2] ✅ REGIME OVERRIDE: penalty=%.4f", regime_penalty)
        return False, regime_penalty

    logger.warning("[Ensemble v4.2] ⚠️ REGIME CONFLICT: session=%s, ts=%.2f -> HARD BLOCK", session, trend_strength)
    return True, 0.0


# =========================================
# v4.2: RF CONFIDENCE GATE
# لو RF في الـ noise zone → HOLD مباشرة
# ده بيوفر وقت ويقلل الـ false signals
# =========================================

# الـ noise zone بناءً على RF distribution الفعلية:
# NOISE (0.43-0.57): ~65% من الوقت
# Signal يبدأ من 0.60+ للـ BUY أو 0.40- للـ SELL
RF_NOISE_UPPER = 0.57   # فوق ده = RF بيقول BUY بثقة
RF_NOISE_LOWER = 0.43   # تحت ده = RF بيقول SELL بثقة


def _rf_confidence_gate(rf_prob, diagnostic=False):
    """
    v4.2 Gate: لو RF في الـ noise zone → return False (HOLD)
    لو RF خارج الـ noise zone → return True (متابعة)

    Noise zone: 0.43 ≤ rf_prob ≤ 0.57
    Signal zone: rf_prob > 0.57 (BUY) or rf_prob < 0.43 (SELL)
    """
    in_noise = RF_NOISE_LOWER <= rf_prob <= RF_NOISE_UPPER
    if in_noise and not diagnostic:
        logger.info(
            "[Ensemble v4.2] 🚫 RF_NOISE_GATE: rf_prob=%.4f in noise zone [%.2f, %.2f] -> HOLD",
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

    # ── Step 1: Trend Strength ──────────────────────────────────
    trend_strength = float(np.clip((current_adx - 20) / 30.0, 0.0, 1.0))
    decision.trend_strength = trend_strength

    # ── Step 2: ATR Double Filter ───────────────────────────────
    atr_mean = atr_series.mean() if atr_series is not None and len(atr_series) > 0 else 0
    atr_ratio = (current_atr / atr_mean) if atr_mean > 0 else 1.0

    if atr_ratio < 0.5 or current_atr < Config.ATR_THRESHOLD:
        decision.direction = None
        decision.decision_reason = "LOW_ATR"
        decision.stage_reached = "ATR_FILTER"
        decision.final_prob = 0.5
        decision.confidence_level = "LOW"
        logger.warning("[Ensemble v5.0] ⛔ LOW_ATR: ratio=%.3f, atr=%.6f -> SKIP", atr_ratio, current_atr)
        _log_decision(decision, current_adx, current_atr)
        return decision

    # ── Step 2.5: RF Confidence Gate ────────────────────────────
    if not _rf_confidence_gate(rf_prob, diagnostic):
        decision.direction = None
        decision.decision_reason = "RF_NOISE_ZONE"
        decision.stage_reached = "SCORE_FLOOR"
        decision.final_prob = 0.5
        decision.confidence_level = "LOW"
        decision.skip_reason = f"RF_NOISE_GATE: rf_prob={rf_prob:.4f} in [{RF_NOISE_LOWER}, {RF_NOISE_UPPER}]"
        _log_decision(decision, current_adx, current_atr)
        return decision

    # ── Step 3: v5.0 XGB-RF Dynamic Weights ────────────────────
    xgb_w, rf_w, market_state = get_dynamic_weights(trend_strength, session)
    decision.xgb_weight = xgb_w
    decision.rf_weight = rf_w
    decision.market_state = market_state

    # ── Step 4: Weighted Average ────────────────────────────────
    weighted_avg = (xgb_w * xgb_prob) + (rf_w * rf_prob)
    decision.weighted_avg = weighted_avg

    # ── Step 5: Conflict Detection ──────────────────────────────
    # XGB أخذ مكان LSTM كـ primary signal
    disagreement = abs(xgb_prob - rf_prob)

    rf_says_buy   = rf_prob  > RF_NOISE_UPPER
    rf_says_sell  = rf_prob  < RF_NOISE_LOWER
    xgb_says_buy  = xgb_prob > 0.55
    xgb_says_sell = xgb_prob < 0.45

    # Hard conflict: XGB و RF في اتجاهين مختلفين بثقة
    if (rf_says_buy and xgb_says_sell) or (rf_says_sell and xgb_says_buy):
        decision.conflict = True
        decision.direction = None
        decision.final_prob = weighted_avg
        decision.decision_reason = "CONFLICT"
        decision.stage_reached = "CONFLICT"
        decision.confidence_level = "LOW"
        decision.skip_reason = (
            f"XGB_RF_CONFLICT: RF={rf_prob:.3f} vs XGB={xgb_prob:.3f} — opposite directions"
        )
        logger.warning("[Ensemble v5.0] ⚠️ %s -> HOLD", decision.skip_reason)
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
        logger.warning("[Ensemble v5.0] ⚠️ %s -> HOLD", decision.skip_reason)
        _log_decision(decision, current_adx, current_atr)
        return decision

    # ── Step 6: Disagreement Penalty ───────────────────────────
    penalty = disagreement * Config.ENSEMBLE_DISAGREEMENT_PENALTY
    decision.penalty = penalty

    if weighted_avg > 0.5:
        base_score = weighted_avg - penalty
    else:
        base_score = weighted_avg + penalty
    base_score = float(np.clip(base_score, 0.0, 1.0))

    # ── Step 7: Side Tracking + Distance ───────────────────────
    distance_from_neutral = abs(base_score - 0.5)
    decision.distance_from_neutral = distance_from_neutral

    if base_score > 0.5:
        decision.side = "BUY"
    elif base_score < 0.5:
        decision.side = "SELL"
    else:
        decision.side = "NONE"

    # ── Step 8: Weak Zone Threshold ────────────────────────────
    weak_zone_threshold = 0.02 if session == "Asia" else 0.01
    decision.weak_zone_threshold_used = weak_zone_threshold
    decision.edge_case = abs(distance_from_neutral - weak_zone_threshold) < 0.005

    # ── Step 9: Score Floor ─────────────────────────────────────
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

    # ── Step 10: Weak Zone ──────────────────────────────────────
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
            logger.info("[Ensemble v4.2] ⚠️ %s -> NO ENTRY", decision.skip_reason)
        _log_decision(decision, current_adx, current_atr)
        return decision

    # ── Step 11: Regime Conflict ────────────────────────────────
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
        logger.warning("[Ensemble v4.2] ⛔ %s -> HOLD", decision.skip_reason)
        _log_decision(decision, current_adx, current_atr)
        return decision

    if regime_penalty > 0:
        base_score = base_score - regime_penalty if base_score > 0.5 else base_score + regime_penalty
        base_score = float(np.clip(base_score, 0.0, 1.0))

    # ── Step 12: Additive Scoring ───────────────────────────────
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

    # ── Step 13: v4.2 Thresholds ────────────────────────────────
    # القديم: BUY > 0.55 → كتير noise يعدي
    # الجديد: BUY > 0.60 → يتوافق مع RF Weak BUY zone الحقيقية
    if diagnostic:
        buy_threshold  = 0.65
        sell_threshold = 0.35
    else:
        if trend_strength > 0.35:
            buy_threshold  = 0.60   # كان 0.55
            sell_threshold = 0.40   # كان 0.45
        elif trend_strength > 0.25:
            buy_threshold  = 0.61   # كان 0.56
            sell_threshold = 0.39   # كان 0.44
        else:
            buy_threshold  = 0.62   # كان 0.58
            sell_threshold = 0.38   # كان 0.42

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
                    "[Ensemble v4.2] 🔍 [NEAR MISS] score:%.4f dist:%.4f ts:%.2f conf:%s",
                    final_prob, distance_from_neutral, trend_strength, _pre_conf
                )

    # ── Step 15: Confidence Classification ──────────────────────
    distance_from_center = abs(final_prob - 0.5)
    if distance_from_center > 0.15:
        decision.confidence_level = "HIGH"
    elif distance_from_center > 0.08:
        decision.confidence_level = "MEDIUM"
    else:
        decision.confidence_level = "LOW"

    # ── Step 16: Log ────────────────────────────────────────────
    _log_decision(decision, current_adx, current_atr)

    # ── Step 17: Console Print ──────────────────────────────────
    status = decision.direction or "HOLD"
    conflict_flag = " ⚠️CONFLICT" if decision.conflict or decision.regime_conflict else ""
    edge_flag = " 🔶EDGE" if decision.edge_case else ""
    print(f"\n\033[92m{'='*65}\033[0m")
    print(f"\033[92m        ENSEMBLE DECISION v5.0 XGB-RF{conflict_flag}{edge_flag}\033[0m")
    print(f"\033[92m{'='*65}\033[0m")
    print(f"\033[92m  Session       : {session}\033[0m")
    print(f"\033[92m  Market State  : {market_state} (ADX: {current_adx:.1f} -> ts: {trend_strength:.3f})\033[0m")
    print(f"\033[92m  XGBoost       : {xgb_prob:.4f} (weight: {xgb_w:.0%}) ← PRIMARY\033[0m")
    print(f"\033[92m  RF            : {rf_prob:.4f} (weight: {rf_w:.0%}) ← COMPLEMENT\033[0m")
    print(f"\033[92m  RF Gate       : {'PASS' if _rf_confidence_gate(rf_prob, True) else 'NOISE'}\033[0m")
    print(f"\033[92m  Weighted Avg  : {weighted_avg:.4f}\033[0m")
    print(f"\033[92m  Disagreement  : {disagreement:.4f} -> Penalty: {decision.penalty:.4f}\033[0m")
    print(f"\033[92m  Base Score    : {base_score:.4f}\033[0m")
    print(f"\033[92m  Session Bonus : {session_bonus:+.4f}\033[0m")
    print(f"\033[92m  Vol. Adjust   : {volatility_adjustment:+.4f}\033[0m")
    print(f"\033[92m  Final Score   : {final_prob:.4f}\033[0m")
    print(f"\033[92m  Thresholds    : BUY>{buy_threshold:.4f} | SELL<{sell_threshold:.4f}\033[0m")
    print(f"\033[92m  Confidence    : {decision.confidence_level}\033[0m")
    print(f"\033[92m  Reason        : {decision.decision_reason}\033[0m")
    print(f"\033[92m  ▶ DECISION    : {status}\033[0m")
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
        print(f"\033[96m       📊 ENSEMBLE RUNTIME METRICS v5.0\033[0m")
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

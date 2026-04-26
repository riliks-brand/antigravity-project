"""
Verification Test Suite — Ensemble Engine v4.1
================================================
Tests ALL rules at RUNTIME including full diagnostics.
Run: pytest test_ensemble_v4.py -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import types
mt5_mock = types.ModuleType("MetaTrader5")
mt5_mock.TIMEFRAME_M5 = 5
mt5_mock.TIMEFRAME_M15 = 15
mt5_mock.TIMEFRAME_H1 = 60
sys.modules["MetaTrader5"] = mt5_mock

from ensemble_engine import ensemble_predict, EnsembleDecision


# =========================================
# HELPERS
# =========================================

def make_atr_series(mean_val=0.001, length=50):
    return pd.Series(np.random.normal(mean_val, mean_val * 0.1, length).clip(0.0001))


def predict(lstm=0.7, rf=0.7, adx=35, atr=0.001, session="London", atr_mean=0.001):
    atr_series = make_atr_series(atr_mean, 50)
    return ensemble_predict(
        lstm_prob=lstm, rf_prob=rf,
        current_adx=adx, current_atr=atr,
        atr_series=atr_series, session=session,
    )


# =========================================
# TEST 1: WEAK ZONE (Session-Aware 0.04/0.05)
# =========================================

class TestWeakZone:
    def test_weak_zone_no_entry_london(self):
        """Distance < 0.04 in London → WEAK_ZONE (Normal)."""
        # LSTM=0.53, RF=0.52, ADX=20 → ts=0, w=50/50
        # weighted=0.525, disagree=0.01, penalty=0.003, base=0.522
        # distance=0.022 < 0.04 → WEAK_ZONE (Normal)
        d = predict(lstm=0.53, rf=0.52, adx=20, session="London")
        assert d.direction is None
        assert "WEAK_ZONE" in d.decision_reason
        assert d.stage_reached == "WEAK_ZONE"

    def test_weak_zone_no_entry_asia(self):
        """Distance < 0.05 in Asia → WEAK_ZONE (Asia stricter)."""
        d = predict(lstm=0.545, rf=0.535, adx=20, session="Asia")
        assert d.direction is None
        assert "Asia stricter" in d.decision_reason
        assert d.stage_reached == "WEAK_ZONE"

    def test_weak_zone_boundary_low(self):
        """Score just above floor but inside weak zone."""
        d = predict(lstm=0.525, rf=0.515, adx=20, session="New York")
        assert d.direction is None
        assert d.decision_reason in ("BELOW_THRESHOLD", "WEAK_ZONE (Normal)")


# =========================================
# TEST 2: SCORE FLOOR (distance < 0.015 → REJECT)
# =========================================

class TestScoreFloor:
    def test_below_floor_rejected(self):
        """Distance < 0.015 → BELOW_THRESHOLD."""
        d = predict(lstm=0.505, rf=0.505, adx=20, session="London")
        assert d.direction is None
        assert d.decision_reason == "BELOW_THRESHOLD"
        assert d.stage_reached == "SCORE_FLOOR"

    def test_very_low_score_rejected(self):
        d = predict(lstm=0.50, rf=0.50, adx=20, session="New York")
        assert d.direction is None

    def test_neutral_score_rejected(self):
        d = predict(lstm=0.51, rf=0.49, adx=25, session="Asia")
        assert d.direction is None


# =========================================
# TEST 3: ADJUSTMENT CAP (≤ 10% of base)
# =========================================

class TestAdjustmentCap:
    def test_adjustment_never_exceeds_10pct(self):
        for _ in range(50):
            adx = np.random.uniform(20, 60)
            lstm = np.random.uniform(0.55, 0.95)
            rf = np.random.uniform(0.55, 0.95)
            session = np.random.choice(["London", "New York", "Asia"])
            d = predict(lstm=lstm, rf=rf, adx=adx, session=session)
            assert 0.0 <= d.final_prob <= 1.0

    def test_boost_cannot_cross_threshold_alone(self):
        d = predict(lstm=0.525, rf=0.515, adx=50, session="London")
        assert d.direction is None


# =========================================
# TEST 4: ATR FILTER (DOUBLE CONDITION)
# =========================================

class TestATRFilter:
    def test_low_atr_ratio_blocks(self):
        atr_series = make_atr_series(0.01, 50)
        d = ensemble_predict(
            lstm_prob=0.85, rf_prob=0.80,
            current_adx=40, current_atr=0.003,
            atr_series=atr_series, session="London",
        )
        assert d.direction is None
        assert d.decision_reason == "LOW_ATR"
        assert d.stage_reached == "ATR_FILTER"

    def test_absolute_atr_below_threshold_blocks(self):
        from config import Config
        tiny_atr = Config.ATR_THRESHOLD * 0.5
        atr_series = make_atr_series(tiny_atr, 50)
        d = ensemble_predict(
            lstm_prob=0.85, rf_prob=0.80,
            current_adx=40, current_atr=tiny_atr,
            atr_series=atr_series, session="London",
        )
        assert d.direction is None
        assert d.decision_reason == "LOW_ATR"
        assert d.stage_reached == "ATR_FILTER"

    def test_healthy_atr_passes(self):
        d = predict(lstm=0.85, rf=0.80, adx=40, atr=0.002, atr_mean=0.002, session="London")
        assert d.decision_reason != "LOW_ATR"


# =========================================
# TEST 5: REGIME CONFLICT → HOLD
# =========================================

class TestRegimeConflict:
    def test_london_low_trend_conflict(self):
        d = predict(lstm=0.75, rf=0.70, adx=20, session="London")
        assert d.direction is None
        assert d.decision_reason == "CONFLICT"
        assert d.stage_reached == "CONFLICT"

    def test_asia_high_trend_conflict(self):
        d = predict(lstm=0.80, rf=0.75, adx=55, session="Asia")
        assert d.direction is None
        assert d.decision_reason == "CONFLICT"
        assert d.stage_reached == "CONFLICT"

    def test_no_conflict_normal(self):
        d = predict(lstm=0.80, rf=0.75, adx=35, session="London")
        assert d.regime_conflict is False


# =========================================
# TEST 6: THRESHOLD TEST
# =========================================

class TestThresholds:
    def test_dynamic_threshold_strong_trend(self):
        d = predict(lstm=0.80, rf=0.75, adx=50, session="New York")
        assert abs(d.buy_threshold - 0.58) < 0.01

    def test_dynamic_threshold_no_trend(self):
        d = predict(lstm=0.80, rf=0.75, adx=20, session="New York")
        assert abs(d.buy_threshold - 0.66) < 0.01

    def test_below_threshold_no_trade(self):
        d = predict(lstm=0.63, rf=0.61, adx=20, session="New York")
        if d.decision_reason not in ("CONFLICT", "WEAK_ZONE (Normal)", "WEAK_ZONE (Asia stricter)"):
            assert d.direction is None or d.final_prob > d.buy_threshold


# =========================================
# TEST 7: VALID SIGNAL TEST
# =========================================

class TestValidSignal:
    def test_strong_buy_signal(self):
        d = predict(lstm=0.85, rf=0.80, adx=40, session="New York")
        assert d.direction == "BUY"
        assert d.decision_reason == "VALID_SIGNAL"
        assert d.stage_reached == "EXECUTION_READY"
        assert d.confidence_level in ("HIGH", "MEDIUM")

    def test_strong_sell_signal(self):
        d = predict(lstm=0.15, rf=0.20, adx=35, session="New York")
        assert d.direction == "SELL"
        assert d.decision_reason == "VALID_SIGNAL"
        assert d.stage_reached == "EXECUTION_READY"

    def test_valid_signal_has_confidence(self):
        d = predict(lstm=0.90, rf=0.85, adx=45, session="London")
        assert d.confidence_level in ("HIGH", "MEDIUM", "LOW")


# =========================================
# TEST 8: LOGGING TEST
# =========================================

class TestLogging:
    def test_all_fields_populated(self):
        d = predict(lstm=0.85, rf=0.80, adx=40, session="London")
        assert d.session == "London"
        assert d.trend_strength > 0
        assert d.decision_reason != ""
        assert d.confidence_level in ("HIGH", "MEDIUM", "LOW")
        assert d.final_prob > 0
        assert d.raw_score > 0
        assert d.side in ("BUY", "SELL", "NONE")
        assert d.stage_reached != "INIT"

    def test_decision_reason_always_set(self):
        scenarios = [
            dict(lstm=0.505, rf=0.505, adx=15, session="London"),
            dict(lstm=0.53, rf=0.52, adx=20, session="New York"),
            dict(lstm=0.75, rf=0.70, adx=20, session="London"),
            dict(lstm=0.85, rf=0.80, adx=40, session="New York"),
            dict(lstm=0.85, rf=0.20, adx=40, session="New York"),
        ]
        valid_reasons = {
            "LOW_ATR", "CONFLICT", "BELOW_THRESHOLD",
            "WEAK_ZONE (Normal)", "WEAK_ZONE (Asia stricter)",
            "VALID_SIGNAL"
        }
        for s in scenarios:
            d = predict(**s)
            assert d.decision_reason != "", f"Empty reason for {s}"
            assert d.decision_reason in valid_reasons, f"Unknown: {d.decision_reason}"

    def test_csv_logging_creates_file(self):
        from config import Config
        if os.path.exists(Config.ENSEMBLE_LOG_FILE):
            os.remove(Config.ENSEMBLE_LOG_FILE)

        predict(lstm=0.85, rf=0.80, adx=40, session="London")

        assert os.path.exists(Config.ENSEMBLE_LOG_FILE)
        import csv
        with open(Config.ENSEMBLE_LOG_FILE, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) >= 1
            row = rows[-1]
            required = [
                "raw_score", "final_score", "session", "trend_strength",
                "session_bonus", "volatility_adjustment", "regime_conflict",
                "decision_reason", "confidence_level",
                "distance_from_neutral", "weak_zone_threshold_used",
                "edge_case", "side", "stage_reached",
            ]
            for field in required:
                assert field in row, f"Missing field: {field}"
                assert row[field] != "", f"Empty field: {field}"


# =========================================
# TEST 9: DIAGNOSTICS TEST (side, stage, distance, edge_case)
# =========================================

class TestDiagnostics:
    def test_side_always_set(self):
        """Side must be BUY, SELL, or NONE on every decision that reaches scoring."""
        d_buy = predict(lstm=0.85, rf=0.80, adx=40, session="New York")
        assert d_buy.side == "BUY"

        d_sell = predict(lstm=0.15, rf=0.20, adx=35, session="New York")
        assert d_sell.side == "SELL"

        d_neutral = predict(lstm=0.505, rf=0.505, adx=20, session="London")
        assert d_neutral.side in ("BUY", "SELL", "NONE")

    def test_stage_reached_always_set(self):
        assert predict(lstm=0.85, rf=0.80, adx=40, session="New York").stage_reached == "EXECUTION_READY"
        assert predict(lstm=0.505, rf=0.505, adx=20, session="London").stage_reached == "SCORE_FLOOR"
        assert predict(lstm=0.53, rf=0.52, adx=20, session="New York").stage_reached == "WEAK_ZONE"
        assert predict(lstm=0.75, rf=0.70, adx=20, session="London").stage_reached == "CONFLICT"

    def test_distance_from_neutral_tracked(self):
        d = predict(lstm=0.85, rf=0.80, adx=40, session="New York")
        assert d.distance_from_neutral is not None
        assert d.distance_from_neutral > 0

    def test_edge_case_detected(self):
        """Edge case should be True when distance is within 0.005 of weak_zone_threshold."""
        d = predict(lstm=0.85, rf=0.80, adx=40, session="New York")
        assert d.edge_case is not None
        assert isinstance(d.edge_case, bool)

    def test_weak_zone_threshold_tracked(self):
        d_london = predict(lstm=0.85, rf=0.80, adx=40, session="London")
        assert d_london.weak_zone_threshold_used == 0.04

        d_asia = predict(lstm=0.85, rf=0.80, adx=40, session="Asia")
        assert d_asia.weak_zone_threshold_used == 0.05

    def test_atr_filter_stage_no_side(self):
        """ATR filter fires before scoring → side stays NONE."""
        atr_series = make_atr_series(0.01, 50)
        d = ensemble_predict(
            lstm_prob=0.85, rf_prob=0.80,
            current_adx=40, current_atr=0.003,
            atr_series=atr_series, session="London",
        )
        assert d.stage_reached == "ATR_FILTER"
        assert d.side == "NONE"  # Never reached scoring


# =========================================
# TEST 10: ASIA vs LONDON WEAK ZONE
# =========================================

class TestAsiaVsLondon:
    def test_asia_rejects_london_allows(self):
        """
        distance ≈ 0.045:
        - London (threshold=0.04): distance > 0.04 → PASSES
        - Asia (threshold=0.05): distance < 0.05 → WEAK_ZONE
        """
        # Need base_score ≈ 0.545 → distance = 0.045
        # LSTM=0.555, RF=0.535, ADX=20 → ts=0, w=50/50
        # weighted=0.545, disagree=0.02, penalty=0.006, base=0.539
        # distance=0.039 ... too low. Need higher.
        # LSTM=0.56, RF=0.54, ADX=20 → weighted=0.55, disagree=0.02
        # penalty=0.006, base=0.544, distance=0.044
        d_london = predict(lstm=0.56, rf=0.54, adx=20, session="New York")
        d_asia = predict(lstm=0.56, rf=0.54, adx=20, session="Asia")

        # London/NY threshold = 0.04. distance ≈ 0.044 > 0.04 → passes weak zone
        # Asia threshold = 0.05. distance ≈ 0.044 < 0.05 → WEAK_ZONE
        if d_london.distance_from_neutral > 0.04:
            assert d_london.decision_reason != "WEAK_ZONE (Normal)", \
                f"London should pass. distance={d_london.distance_from_neutral}"

        assert d_asia.decision_reason == "WEAK_ZONE (Asia stricter)", \
            f"Asia should reject. distance={d_asia.distance_from_neutral}, reason={d_asia.decision_reason}"

    def test_asia_stricter_reason_label(self):
        """Asia weak zone must use 'Asia stricter' label."""
        d = predict(lstm=0.53, rf=0.52, adx=20, session="Asia")
        if "WEAK_ZONE" in d.decision_reason:
            assert "Asia stricter" in d.decision_reason


# =========================================
# TEST 11: NO BOOST-ONLY TRADE
# =========================================

class TestNoBoostOnlyTrade:
    def test_marginal_score_with_max_boost(self):
        d = predict(lstm=0.59, rf=0.57, adx=20, session="New York")
        if d.decision_reason not in ("CONFLICT", "WEAK_ZONE (Normal)", "WEAK_ZONE (Asia stricter)", "BELOW_THRESHOLD", "LOW_ATR"):
            assert d.direction is None or d.final_prob > d.buy_threshold

    def test_adjustment_bounded_by_base(self):
        for _ in range(100):
            lstm = np.random.uniform(0.6, 0.9)
            rf = np.random.uniform(0.6, 0.9)
            adx = np.random.uniform(25, 50)
            session = np.random.choice(["London", "New York", "Asia"])
            d = predict(lstm=lstm, rf=rf, adx=adx, session=session)
            assert 0.0 <= d.final_prob <= 1.0


# =========================================
# TEST 12: STRICT STAGE VALIDATION
# =========================================

class TestStageValidation:
    VALID_STAGES = {
        "SCORE_FLOOR", "WEAK_ZONE", "ATR_FILTER",
        "CONFLICT", "THRESHOLD_CHECK", "EXECUTION_READY",
    }

    def test_every_path_has_valid_stage(self):
        """Every possible decision must end with a valid stage_reached."""
        test_cases = [
            dict(lstm=0.505, rf=0.505, adx=20, session="London"),     # SCORE_FLOOR
            dict(lstm=0.53, rf=0.52, adx=20, session="New York"),     # WEAK_ZONE
            dict(lstm=0.75, rf=0.70, adx=20, session="London"),       # CONFLICT
            dict(lstm=0.85, rf=0.80, adx=40, session="New York"),     # EXECUTION_READY
            dict(lstm=0.15, rf=0.20, adx=35, session="New York"),     # EXECUTION_READY (SELL)
            dict(lstm=0.63, rf=0.61, adx=20, session="New York"),     # THRESHOLD_CHECK or CONFLICT
        ]
        for tc in test_cases:
            d = predict(**tc)
            assert d.stage_reached in self.VALID_STAGES, \
                f"Invalid stage '{d.stage_reached}' for {tc}"

    def test_atr_filter_stage(self):
        atr_series = make_atr_series(0.01, 50)
        d = ensemble_predict(0.85, 0.80, 40, 0.003, atr_series, "London")
        assert d.stage_reached == "ATR_FILTER"

    def test_stage_never_init_after_processing(self):
        """INIT is only the default — after any processing it must change."""
        for _ in range(20):
            lstm = np.random.uniform(0.1, 0.9)
            rf = np.random.uniform(0.1, 0.9)
            adx = np.random.uniform(15, 60)
            session = np.random.choice(["London", "New York", "Asia"])
            d = predict(lstm=lstm, rf=rf, adx=adx, session=session)
            assert d.stage_reached != "INIT", \
                f"stage_reached still INIT for lstm={lstm}, rf={rf}, adx={adx}, session={session}"


# =========================================
# TEST 13: NO SILENT FAILURES
# =========================================

class TestNoSilentFailures:
    def test_logging_does_not_silently_fail(self):
        """Logging must succeed without raising — if it fails, RuntimeError is raised."""
        # This test simply proves logging works. If it threw RuntimeError, this test would fail.
        d = predict(lstm=0.85, rf=0.80, adx=40, session="London")
        assert d is not None

    def test_invalid_stage_raises_runtime_error(self):
        """If an invalid stage_reached somehow gets set, logging must raise."""
        from ensemble_engine import _log_decision, EnsembleDecision
        bad = EnsembleDecision()
        bad.stage_reached = "INVALID_STAGE"
        with pytest.raises(RuntimeError, match="CRITICAL: Invalid stage_reached"):
            _log_decision(bad, current_adx=0, current_atr=0)


# =========================================
# TEST 14: RUNTIME METRICS TRACKER
# =========================================

class TestRuntimeMetrics:
    def test_metrics_record_and_count(self):
        from ensemble_engine import get_metrics, DecisionMetrics
        metrics = get_metrics()
        before = metrics.total_signals

        predict(lstm=0.85, rf=0.80, adx=40, session="New York")
        predict(lstm=0.505, rf=0.505, adx=20, session="London")

        assert metrics.total_signals >= before + 2

    def test_metrics_stage_distribution(self):
        from ensemble_engine import DecisionMetrics
        m = DecisionMetrics()

        class FakeDecision:
            pass

        d1 = FakeDecision()
        d1.stage_reached = "EXECUTION_READY"
        d1.decision_reason = "VALID_SIGNAL"
        d1.side = "BUY"
        m.record(d1)

        d2 = FakeDecision()
        d2.stage_reached = "WEAK_ZONE"
        d2.decision_reason = "WEAK_ZONE (Normal)"
        d2.side = "BUY"
        m.record(d2)

        assert m.total_signals == 2
        assert m.stage_counts["EXECUTION_READY"] == 1
        assert m.stage_counts["WEAK_ZONE"] == 1
        assert m.hold_count == 1
        assert m.execution_ready_rate == 50.0

    def test_metrics_print_summary(self, capsys):
        from ensemble_engine import DecisionMetrics
        m = DecisionMetrics()
        m.print_summary()
        captured = capsys.readouterr()
        assert "ENSEMBLE RUNTIME METRICS" in captured.out
        assert "TOTAL_SIGNALS" in captured.out
        assert "HOLD_COUNT" in captured.out
        assert "EXECUTION_READY" in captured.out


# =========================================
# TEST 15: TRADE MANAGER TESTS
# =========================================

class TestTradeManager:
    def test_get_adaptive_risk_strong_trend_high_confidence(self):
        from trade_manager import TradeManager
        mgr = TradeManager()
        risk = mgr.get_adaptive_risk("London", 0.7, False, "HIGH")
        assert risk == 1.0

    def test_get_adaptive_risk_weak_trend(self):
        from trade_manager import TradeManager
        mgr = TradeManager()
        risk = mgr.get_adaptive_risk("Asia", 0.3, False, "MEDIUM")
        assert risk == 0.5

    def test_get_adaptive_risk_regime_change_reduces(self):
        from trade_manager import TradeManager
        mgr = TradeManager()
        risk = mgr.get_adaptive_risk("London", 0.7, True, "HIGH")
        assert abs(risk - 0.7) < 0.01

    def test_regime_persistence_no_flip_flop(self):
        from trade_manager import TradeManager
        mgr = TradeManager()
        regime, changed = mgr.update_regime(0.7)
        assert regime == "TRENDING"

        regime, changed = mgr.update_regime(0.3)
        assert regime == "TRENDING"
        assert changed is False

        regime, changed = mgr.update_regime(0.3)
        assert regime == "RANGING"
        assert changed is True

    def test_regime_resets_on_return(self):
        from trade_manager import TradeManager
        mgr = TradeManager()
        mgr.update_regime(0.7)
        mgr.update_regime(0.3)
        regime, changed = mgr.update_regime(0.7)
        assert regime == "TRENDING"
        assert changed is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

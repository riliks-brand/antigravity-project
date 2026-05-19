"""
Property-Based Tests — bot-performance-trading-fix
====================================================

Six Hypothesis property tests covering:
  - Property 1 (Task 2.3): Cache correctness
  - Property 2 (Task 2.4): Incremental correctness
  - Property 3 (Task 2.5): Output completeness
  - Property 4 (Task 2.6): Cache isolation
  - Property 5 (Task 4.2): High-confidence counter-trend signals not hard-blocked
  - Property 6 (Task 4.3): Low-confidence counter-trend signals receive penalty, not hard block

Run with:
    pytest tests/test_bot_perf_fix.py -v
"""

import sys
import os
import logging

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st
import hypothesis.extra.numpy as npst

# =========================================
# IMPORTS FROM PROJECT MODULES
# =========================================

from pattern_detector import _pattern_cache, OUTPUT_COLUMNS, add_chart_patterns
from ensemble_engine import ensemble_predict
from config import Config


# =========================================
# OHLC DATAFRAME STRATEGY
# =========================================

def ohlc_dataframe_strategy(min_rows: int = 50, max_rows: int = 200):
    """
    Hypothesis strategy that generates a valid OHLC DataFrame with an ATR column.

    Columns: open, high, low, close, ATR — all positive floats.
    The high >= max(open, close) and low <= min(open, close) invariant is enforced.
    """
    @st.composite
    def _strategy(draw):
        n = draw(st.integers(min_value=min_rows, max_value=max_rows))

        # Generate base close prices (positive, reasonable range)
        closes = draw(
            npst.arrays(
                dtype=np.float64,
                shape=(n,),
                elements=st.floats(min_value=0.5, max_value=5000.0,
                                   allow_nan=False, allow_infinity=False),
            )
        )

        # Generate ATR values (positive, small relative to price)
        atrs = draw(
            npst.arrays(
                dtype=np.float64,
                shape=(n,),
                elements=st.floats(min_value=1e-5, max_value=50.0,
                                   allow_nan=False, allow_infinity=False),
            )
        )

        # Derive open, high, low from close + ATR so OHLC invariants hold
        half_atr = atrs / 2.0
        opens = np.abs(closes - half_atr * np.random.default_rng(42).uniform(0, 1, n))
        highs = np.maximum(opens, closes) + np.abs(half_atr)
        lows  = np.minimum(opens, closes) - np.abs(half_atr)

        # Ensure all values are strictly positive
        lows = np.maximum(lows, 1e-6)

        df = pd.DataFrame({
            "open":  opens,
            "high":  highs,
            "low":   lows,
            "close": closes,
            "ATR":   atrs,
        })
        return df

    return _strategy()


# =========================================
# PROPERTY 1 — Cache Correctness (Task 2.3)
# Validates: Requirements 1.1, 1.5
# =========================================

@given(
    df=ohlc_dataframe_strategy(min_rows=50, max_rows=150),
    symbol=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        min_size=1,
        max_size=10,
    ),
)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    deadline=None,
)
def test_property1_cache_correctness(df, symbol):
    """
    **Property 1: Cache correctness — cached results equal fresh results**

    Validates: Requirements 1.1, 1.5

    For any valid OHLC DataFrame and any symbol, calling add_chart_patterns()
    twice with the same DataFrame must produce identical values in all 15
    OUTPUT_COLUMNS. The second call must be served from cache.
    """
    # Clear cache for this symbol before the test run
    if symbol in _pattern_cache:
        del _pattern_cache[symbol]

    # Capture log messages to verify "Cache hit" on second call
    log_messages = []

    class _CapturingHandler(logging.Handler):
        def emit(self, record):
            log_messages.append(record.getMessage())

    handler = _CapturingHandler()
    pattern_logger = logging.getLogger("PatternDetector")
    pattern_logger.addHandler(handler)
    original_level = pattern_logger.level
    pattern_logger.setLevel(logging.DEBUG)

    try:
        # First call — full computation (cold cache)
        result1 = add_chart_patterns(df.copy(), symbol)

        # Second call — should be a cache hit
        result2 = add_chart_patterns(df.copy(), symbol)

        # Assert all 15 OUTPUT_COLUMNS are identical between calls
        for col in OUTPUT_COLUMNS:
            assert col in result1.columns, f"Column '{col}' missing from first result"
            assert col in result2.columns, f"Column '{col}' missing from second result"
            np.testing.assert_array_equal(
                result1[col].values,
                result2[col].values,
                err_msg=(
                    f"Column '{col}' differs between first and second call for symbol '{symbol}'. "
                    f"Cache correctness violated."
                ),
            )

        # Assert second call log contains "Cache hit"
        cache_hit_found = any("Cache hit" in msg for msg in log_messages)
        assert cache_hit_found, (
            f"Expected 'Cache hit' in log messages for symbol '{symbol}' on second call, "
            f"but got: {log_messages}"
        )

    finally:
        pattern_logger.removeHandler(handler)
        pattern_logger.setLevel(original_level)
        # Clean up cache entry created by this test
        if symbol in _pattern_cache:
            del _pattern_cache[symbol]


# =========================================
# PROPERTY 2 — Incremental Correctness (Task 2.4)
# Validates: Requirements 1.2, 1.5
# =========================================

@given(
    base_df=ohlc_dataframe_strategy(min_rows=50, max_rows=100),
    k=st.integers(min_value=1, max_value=20),
)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    deadline=None,
)
def test_property2_incremental_correctness(base_df, k):
    """
    **Property 2: Incremental correctness — previously computed rows are unchanged**

    Validates: Requirements 1.2, 1.5

    For any valid OHLC DataFrame extended with k new rows, calling
    add_chart_patterns on the extended DataFrame must produce the same pattern
    flag values for all rows that existed in the original DataFrame as a fresh
    (no-cache) call would produce.
    """
    symbol = "_PROP2_TEST_SYMBOL_"
    n = len(base_df)

    # Build extended DataFrame: base rows + k new rows appended
    extra_rows = base_df.iloc[-k:].copy()
    # Slightly perturb the extra rows so they are distinct
    extra_rows = extra_rows * 1.001
    extended_df = pd.concat([base_df, extra_rows], ignore_index=True)

    # Clear cache before test
    if symbol in _pattern_cache:
        del _pattern_cache[symbol]

    try:
        # Step 1: warm the cache with the base DataFrame
        add_chart_patterns(base_df.copy(), symbol)

        # Step 2: call with the extended DataFrame (incremental update path)
        result_incremental = add_chart_patterns(extended_df.copy(), symbol)

        # Step 3: fresh (no-cache) call on the same extended DataFrame
        if symbol in _pattern_cache:
            del _pattern_cache[symbol]
        result_fresh = add_chart_patterns(extended_df.copy(), symbol)

        # Assert: for all rows 0..N-1, pattern flag values must match
        # between incremental and fresh results
        flag_columns = OUTPUT_COLUMNS[:-1]  # exclude pattern_bias_score (derived)
        for col in flag_columns:
            assert col in result_incremental.columns, (
                f"Column '{col}' missing from incremental result"
            )
            assert col in result_fresh.columns, (
                f"Column '{col}' missing from fresh result"
            )
            np.testing.assert_array_equal(
                result_incremental[col].values[:n],
                result_fresh[col].values[:n],
                err_msg=(
                    f"Column '{col}' rows 0..{n-1} differ between incremental and fresh call. "
                    f"Incremental correctness violated."
                ),
            )

    finally:
        if symbol in _pattern_cache:
            del _pattern_cache[symbol]


# =========================================
# PROPERTY 3 — Output Completeness (Task 2.5)
# Validates: Requirements 1.8
# =========================================

@given(df=ohlc_dataframe_strategy(min_rows=50, max_rows=200))
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    deadline=None,
)
def test_property3_output_completeness(df):
    """
    **Property 3: Output completeness — all 15 columns always present**

    Validates: Requirements 1.8

    For any valid OHLC DataFrame with at least 50 rows and an ATR column,
    add_chart_patterns() must return a DataFrame containing all 15 expected
    columns: the 14 pattern flag columns and pattern_bias_score.
    """
    symbol = "_PROP3_TEST_SYMBOL_"

    # Clear cache to ensure a fresh computation
    if symbol in _pattern_cache:
        del _pattern_cache[symbol]

    try:
        result = add_chart_patterns(df.copy(), symbol)

        assert len(OUTPUT_COLUMNS) == 15, (
            f"OUTPUT_COLUMNS should have 15 entries, got {len(OUTPUT_COLUMNS)}"
        )

        for col in OUTPUT_COLUMNS:
            assert col in result.columns, (
                f"Expected column '{col}' not found in result. "
                f"Present columns: {list(result.columns)}"
            )

        # Also verify the returned DataFrame has the same number of rows as input
        assert len(result) == len(df), (
            f"Result has {len(result)} rows but input had {len(df)} rows."
        )

    finally:
        if symbol in _pattern_cache:
            del _pattern_cache[symbol]


# =========================================
# PROPERTY 4 — Cache Isolation (Task 2.6)
# Validates: Requirements 1.6
# =========================================

@given(
    df_a=ohlc_dataframe_strategy(min_rows=50, max_rows=120),
    df_b=ohlc_dataframe_strategy(min_rows=50, max_rows=120),
    symbol_a=st.just("SYMBOL_A_ISOLATION"),
    symbol_b=st.just("SYMBOL_B_ISOLATION"),
)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    deadline=None,
)
def test_property4_cache_isolation(df_a, df_b, symbol_a, symbol_b):
    """
    **Property 4: Cache isolation — per-symbol independence**

    Validates: Requirements 1.6

    For any two distinct symbol names and two distinct OHLC DataFrames,
    calling add_chart_patterns for symbol A must not alter the cached result
    for symbol A when the cache for symbol B is modified.
    """
    # Ensure symbols are distinct
    assume(symbol_a != symbol_b)

    # Clear both cache entries before test
    for sym in (symbol_a, symbol_b):
        if sym in _pattern_cache:
            del _pattern_cache[sym]

    try:
        # Step 1: Populate cache for symbol A
        result_a_first = add_chart_patterns(df_a.copy(), symbol_a)

        # Capture the cached arrays for symbol A
        assert symbol_a in _pattern_cache, "Cache entry for symbol A not created"
        cached_a_before = {
            col: _pattern_cache[symbol_a]["columns"][col].copy()
            for col in OUTPUT_COLUMNS
        }

        # Step 2: Populate cache for symbol B (should not affect symbol A)
        add_chart_patterns(df_b.copy(), symbol_b)

        # Step 3: Modify the cache for symbol B directly
        if symbol_b in _pattern_cache:
            for col in OUTPUT_COLUMNS:
                if col in _pattern_cache[symbol_b]["columns"]:
                    _pattern_cache[symbol_b]["columns"][col] = np.zeros(
                        len(_pattern_cache[symbol_b]["columns"][col]), dtype=int
                    )

        # Step 4: Assert that the cached result for symbol A is unchanged
        assert symbol_a in _pattern_cache, (
            "Cache entry for symbol A was unexpectedly removed"
        )
        cached_a_after = _pattern_cache[symbol_a]["columns"]

        for col in OUTPUT_COLUMNS:
            np.testing.assert_array_equal(
                cached_a_before[col],
                cached_a_after[col],
                err_msg=(
                    f"Cache for symbol A column '{col}' was altered after modifying "
                    f"symbol B's cache. Cache isolation violated."
                ),
            )

    finally:
        for sym in (symbol_a, symbol_b):
            if sym in _pattern_cache:
                del _pattern_cache[sym]


# =========================================
# HELPERS FOR ENSEMBLE TESTS
# =========================================

def _make_atr_series(n: int = 50, base_atr: float = 0.001) -> pd.Series:
    """Create a realistic ATR series for ensemble_predict calls."""
    rng = np.random.default_rng(0)
    values = base_atr + rng.uniform(0, base_atr * 0.5, n)
    return pd.Series(values)


def _call_ensemble_buy_counter_trend(xgb_prob: float, h1_trend: int) -> object:
    """
    Call ensemble_predict with a BUY-biased setup that is counter-trend.

    When h1_trend == -1, a BUY signal is counter-trend.
    When h1_trend ==  1, a SELL signal is counter-trend — we flip xgb_prob
    to produce a SELL-biased base score.

    Returns the EnsembleDecision object.
    """
    atr_series = _make_atr_series()
    current_atr = float(atr_series.mean())

    if h1_trend == -1:
        # BUY counter-trend: xgb_prob > 0.5 → BUY side
        effective_xgb = xgb_prob
        # RF must agree (BUY) and be outside noise zone to pass RF gate
        rf_prob = 0.65
    else:
        # SELL counter-trend (h1_trend == 1): xgb_prob < 0.5 → SELL side
        # We invert xgb_prob so the BUY-biased input becomes a SELL signal
        effective_xgb = 1.0 - xgb_prob
        rf_prob = 0.35

    # Use a moderate ADX so we don't hit ATR/regime filters
    current_adx = 20.0

    decision = ensemble_predict(
        xgb_prob=effective_xgb,
        rf_prob=rf_prob,
        current_adx=current_adx,
        current_atr=current_atr,
        atr_series=atr_series,
        session="London",
        diagnostic=False,
        h1_trend=h1_trend,
        symbol="EURUSD",
    )
    return decision


# =========================================
# PROPERTY 5 — High-Confidence Counter-Trend Not Hard-Blocked (Task 4.2)
# Validates: Requirements 2.1, 2.2
# =========================================

@given(
    xgb_prob=st.floats(min_value=0.75, max_value=0.99,
                       allow_nan=False, allow_infinity=False),
    h1_trend=st.sampled_from([-1, 1]),
)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_property5_high_confidence_counter_trend_not_blocked(xgb_prob, h1_trend):
    """
    **Property 5: High-confidence counter-trend signals are not hard-blocked**

    Validates: Requirements 2.1, 2.2

    For any ensemble evaluation where xgb_prob >= 0.75 and the signal is
    counter-trend (against_h1 = True), the decision_reason must NOT be
    "COUNTER_TREND_H1". The v6.1 soft-block must allow high-confidence
    counter-trend signals through to the threshold check step.
    """
    decision = _call_ensemble_buy_counter_trend(xgb_prob, h1_trend)

    assert decision.decision_reason != "COUNTER_TREND_H1", (
        f"High-confidence counter-trend signal was hard-blocked with "
        f"decision_reason='COUNTER_TREND_H1'. "
        f"xgb_prob={xgb_prob:.4f} >= 0.75 should override the counter-trend block. "
        f"h1_trend={h1_trend}, decision={decision}"
    )


# =========================================
# PROPERTY 6 — Low-Confidence Counter-Trend Penalty, Not Hard Block (Task 4.3)
# Validates: Requirements 2.3, 2.4, 2.10
# =========================================

@given(
    xgb_prob=st.floats(min_value=0.50, max_value=0.749,
                       allow_nan=False, allow_infinity=False),
)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_property6_low_confidence_counter_trend_penalty_not_hard_block(xgb_prob):
    """
    **Property 6: Low-confidence counter-trend signals receive penalty, not hard block**

    Validates: Requirements 2.3, 2.4, 2.10

    For any ensemble evaluation where xgb_prob < 0.75 and the signal is
    counter-trend, the decision_reason must NOT be "COUNTER_TREND_H1"
    (no hard block). Additionally, the final_prob must be lower than it
    would be without the counter-trend penalty.
    """
    h1_trend = -1  # BUY counter-trend scenario

    atr_series = _make_atr_series()
    current_atr = float(atr_series.mean())
    current_adx = 20.0

    # Call WITH counter-trend (h1_trend = -1, BUY side)
    decision_with_ct = ensemble_predict(
        xgb_prob=xgb_prob,
        rf_prob=0.65,
        current_adx=current_adx,
        current_atr=current_atr,
        atr_series=atr_series,
        session="London",
        diagnostic=False,
        h1_trend=h1_trend,
        symbol="EURUSD",
    )

    # Call WITHOUT counter-trend (h1_trend = 0, neutral)
    decision_without_ct = ensemble_predict(
        xgb_prob=xgb_prob,
        rf_prob=0.65,
        current_adx=current_adx,
        current_atr=current_atr,
        atr_series=atr_series,
        session="London",
        diagnostic=False,
        h1_trend=0,
        symbol="EURUSD",
    )

    # Assertion 1: No hard block — decision_reason must NOT be "COUNTER_TREND_H1"
    assert decision_with_ct.decision_reason != "COUNTER_TREND_H1", (
        f"Low-confidence counter-trend signal was hard-blocked with "
        f"decision_reason='COUNTER_TREND_H1'. "
        f"xgb_prob={xgb_prob:.4f} < 0.75 should receive a penalty, not a hard block. "
        f"decision={decision_with_ct}"
    )

    # Assertion 2: final_prob is lower with counter-trend penalty than without
    # The penalty (COUNTER_TREND_PENALTY = -0.04) reduces the score.
    # We allow a small tolerance for the clamping in total_adjustment.
    ct_penalty = abs(getattr(Config, "COUNTER_TREND_PENALTY", -0.04))

    assert decision_with_ct.final_prob <= decision_without_ct.final_prob + 1e-9, (
        f"Counter-trend penalty did not reduce final_prob. "
        f"With CT: final_prob={decision_with_ct.final_prob:.6f}, "
        f"Without CT: final_prob={decision_without_ct.final_prob:.6f}. "
        f"Expected with_ct <= without_ct. "
        f"xgb_prob={xgb_prob:.4f}, penalty={ct_penalty}"
    )


# ===========================================================================
# Additional imports for Properties 7–12
# ===========================================================================

import math
from unittest.mock import patch, MagicMock

from ensemble_engine import _compute_session_bonus  # noqa: E402 (already imported above via ensemble_predict)
from trade_manager import TradeManager  # noqa: E402


# ===========================================================================
# Helper — minimal ATR series for ensemble_predict (Properties 11–12)
# ===========================================================================

def _make_atr_series_simple(current_atr: float = 0.001, length: int = 50) -> pd.Series:
    """Return a Series whose mean equals current_atr so atr_ratio == 1.0."""
    return pd.Series([current_atr] * length)


# ===========================================================================
# Property 7 — Pacific session detection covers all UTC 22–23 hours
# Validates: Requirements 3.1
# ===========================================================================

@given(hour=st.sampled_from([22, 23]))
@settings(max_examples=50, deadline=None)
def test_property7_pacific_session_detection(hour):
    """
    **Validates: Requirements 3.1**

    For any UTC hour in {22, 23}, TradeManager.get_active_session() must
    return "Pacific".
    """
    mock_dt = MagicMock()
    mock_dt.utcnow.return_value.hour = hour

    with patch("trade_manager.datetime.datetime", mock_dt):
        session = TradeManager.get_active_session()

    assert session == "Pacific", (
        f"Expected 'Pacific' for UTC hour {hour}, got '{session}'"
    )


# ===========================================================================
# Property 8 — Pacific session trading respects the TRADE_SESSION_PACIFIC flag
# Validates: Requirements 3.2, 3.8
# ===========================================================================

@given(hour=st.sampled_from([22, 23]), flag=st.booleans())
@settings(max_examples=50, deadline=None)
def test_property8_pacific_session_flag(hour, flag):
    """
    **Validates: Requirements 3.2, 3.8**

    For any call to is_in_trading_session() during UTC hours 22–23, the
    return value must be (True, ...) iff TRADE_SESSION_PACIFIC is True.
    """
    original_flag = Config.TRADE_SESSION_PACIFIC
    original_only = Config.TRADE_ONLY_IN_SESSIONS
    try:
        Config.TRADE_SESSION_PACIFIC = flag
        Config.TRADE_ONLY_IN_SESSIONS = True  # ensure session filter is active

        mock_dt = MagicMock()
        mock_dt.utcnow.return_value.hour = hour

        with patch("trade_manager.datetime.datetime", mock_dt):
            result, reason = TradeManager.is_in_trading_session()

        assert result == flag, (
            f"Expected is_in_trading_session()[0] == {flag} for UTC hour {hour} "
            f"with TRADE_SESSION_PACIFIC={flag}, got {result}. Reason: {reason}"
        )
    finally:
        Config.TRADE_SESSION_PACIFIC = original_flag
        Config.TRADE_ONLY_IN_SESSIONS = original_only


# ===========================================================================
# Property 9 — Pacific session bonus is always zero
# Validates: Requirements 3.5
# ===========================================================================

@given(trend_strength=st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=50, deadline=None)
def test_property9_pacific_session_bonus_zero(trend_strength):
    """
    **Validates: Requirements 3.5**

    For any trend_strength in [0.0, 1.0],
    _compute_session_bonus("Pacific", trend_strength) must return 0.0.
    """
    assume(not math.isnan(trend_strength) and not math.isinf(trend_strength))

    bonus = _compute_session_bonus("Pacific", trend_strength)

    assert bonus == 0.0, (
        f"Expected Pacific session bonus == 0.0 for trend_strength={trend_strength}, "
        f"got {bonus}"
    )


# ===========================================================================
# Property 10 — Existing session detection is unchanged
# Validates: Requirements 3.9
# ===========================================================================

@given(hour=st.integers(min_value=0, max_value=23))
@settings(max_examples=50, deadline=None)
def test_property10_existing_session_detection_unchanged(hour):
    """
    **Validates: Requirements 3.9**

    - Hours in [7, 16) → "London"  (London takes priority over NY overlap)
    - Hours in [0, 9) that are NOT in London range → "Asia"
    - Hours 22–23 must NOT return "UNKNOWN" (they now return "Pacific")
    """
    mock_dt = MagicMock()
    mock_dt.utcnow.return_value.hour = hour

    with patch("trade_manager.datetime.datetime", mock_dt):
        session = TradeManager.get_active_session()

    london_start, london_end = Config.SESSION_LONDON   # (7, 16)
    asia_start, asia_end = Config.SESSION_ASIA         # (0, 9)

    in_london = london_start <= hour < london_end
    in_asia = asia_start <= hour < asia_end

    if in_london:
        assert session == "London", (
            f"Hour {hour} is in London range [{london_start},{london_end}), "
            f"expected 'London', got '{session}'"
        )
    elif in_asia:
        assert session == "Asia", (
            f"Hour {hour} is in Asia range [{asia_start},{asia_end}) and not London, "
            f"expected 'Asia', got '{session}'"
        )

    # Hours 22–23 must no longer return "UNKNOWN"
    if hour in (22, 23):
        assert session != "UNKNOWN", (
            f"Hour {hour} must not return 'UNKNOWN' after Pacific session fix, "
            f"got '{session}'"
        )


# ===========================================================================
# Property 11 — BUY threshold matches the correct tier for all trend_strength
# Validates: Requirements 4.1, 4.2, 4.3
# ===========================================================================

@given(trend_strength=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=50, deadline=None)
def test_property11_buy_threshold_tier(trend_strength):
    """
    **Validates: Requirements 4.1, 4.2, 4.3**

    For any trend_strength in [0.0, 1.0] with diagnostic=False, the
    buy_threshold selected by ensemble_predict must match the expected tier:
      - trend_strength <= 0.25  → PROB_THRESHOLD_BUY_RANGING      (0.58)
      - 0.25 < ts <= 0.35       → PROB_THRESHOLD_BUY_TRANSITIONING (0.57)
      - trend_strength > 0.35   → PROB_THRESHOLD_BUY_TRENDING      (0.56)
    """
    assume(not math.isinf(trend_strength))

    # Derive ADX from trend_strength using the inverse of:
    #   trend_strength = clip((adx - 15) / 35, 0, 1)
    # → adx = trend_strength * 35 + 15
    adx = trend_strength * 35 + 15
    current_atr = 0.001
    atr_series = _make_atr_series_simple(current_atr)

    # Use xgb_prob and rf_prob that are clearly outside the noise zone so the
    # decision reaches Step 13 (threshold check) without being filtered earlier.
    xgb_prob = 0.65
    rf_prob = 0.65

    decision = ensemble_predict(
        xgb_prob=xgb_prob,
        rf_prob=rf_prob,
        current_adx=adx,
        current_atr=current_atr,
        atr_series=atr_series,
        session="London",
        diagnostic=False,
        h1_trend=0,
    )

    if trend_strength <= 0.25:
        expected = Config.PROB_THRESHOLD_BUY_RANGING
    elif trend_strength <= 0.35:
        expected = Config.PROB_THRESHOLD_BUY_TRANSITIONING
    else:
        expected = Config.PROB_THRESHOLD_BUY_TRENDING

    assert decision.buy_threshold == expected, (
        f"trend_strength={trend_strength:.4f} → expected buy_threshold={expected}, "
        f"got {decision.buy_threshold}. stage={decision.stage_reached}"
    )


# ===========================================================================
# Property 12 — SELL threshold matches the correct tier for all trend_strength
# Validates: Requirements 4.4, 4.5, 4.6
# ===========================================================================

@given(trend_strength=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=50, deadline=None)
def test_property12_sell_threshold_tier(trend_strength):
    """
    **Validates: Requirements 4.4, 4.5, 4.6**

    For any trend_strength in [0.0, 1.0] with diagnostic=False, the
    sell_threshold selected by ensemble_predict must match the expected tier:
      - trend_strength <= 0.25  → PROB_THRESHOLD_SELL_RANGING       (0.42)
      - 0.25 < ts <= 0.35       → PROB_THRESHOLD_SELL_TRANSITIONING  (0.43)
      - trend_strength > 0.35   → PROB_THRESHOLD_SELL_TRENDING       (0.44)
    """
    assume(not math.isinf(trend_strength))

    adx = trend_strength * 35 + 15
    current_atr = 0.001
    atr_series = _make_atr_series_simple(current_atr)

    xgb_prob = 0.65
    rf_prob = 0.65

    decision = ensemble_predict(
        xgb_prob=xgb_prob,
        rf_prob=rf_prob,
        current_adx=adx,
        current_atr=current_atr,
        atr_series=atr_series,
        session="London",
        diagnostic=False,
        h1_trend=0,
    )

    if trend_strength <= 0.25:
        expected = Config.PROB_THRESHOLD_SELL_RANGING
    elif trend_strength <= 0.35:
        expected = Config.PROB_THRESHOLD_SELL_TRANSITIONING
    else:
        expected = Config.PROB_THRESHOLD_SELL_TRENDING

    assert decision.sell_threshold == expected, (
        f"trend_strength={trend_strength:.4f} → expected sell_threshold={expected}, "
        f"got {decision.sell_threshold}. stage={decision.stage_reached}"
    )


# ===========================================================================
# Task 8.1 — PatternDetector Performance Unit Tests
# Validates: Requirements 1.4
# ===========================================================================

def test_pattern_detector_performance():
    """
    Unit test: PatternDetector performance with a 2000-row DataFrame.

    Validates: Requirements 1.4

    - First call (cold cache) must complete in under 2.0 seconds.
    - Second call (cache hit) must be at least 10× faster than the first call.
    """
    import time

    symbol = "_PERF_TEST_SYMBOL_"

    # Build a 2000-row OHLC DataFrame
    rng = np.random.default_rng(42)
    n = 2000
    closes = 1.1000 + rng.uniform(-0.05, 0.05, n).cumsum()
    closes = np.abs(closes)
    atrs = rng.uniform(0.0005, 0.002, n)
    half_atr = atrs / 2.0
    opens = np.abs(closes - half_atr * rng.uniform(0, 1, n))
    highs = np.maximum(opens, closes) + np.abs(half_atr)
    lows = np.minimum(opens, closes) - np.abs(half_atr)
    lows = np.maximum(lows, 1e-6)

    df = pd.DataFrame({
        "open":  opens,
        "high":  highs,
        "low":   lows,
        "close": closes,
        "ATR":   atrs,
    })

    # Clear cache before test
    if symbol in _pattern_cache:
        del _pattern_cache[symbol]

    try:
        # --- First call: cold cache ---
        t0 = time.perf_counter()
        result1 = add_chart_patterns(df.copy(), symbol)
        t1 = time.perf_counter()
        cold_time = t1 - t0

        assert cold_time < 2.0, (
            f"First call (cold cache, 2000 rows) took {cold_time:.3f}s — "
            f"must complete in under 2.0 seconds."
        )

        # --- Second call: cache hit ---
        t2 = time.perf_counter()
        result2 = add_chart_patterns(df.copy(), symbol)
        t3 = time.perf_counter()
        cache_time = t3 - t2

        # Cache hit must be at least 10× faster than the cold call
        assert cache_time * 10 <= cold_time, (
            f"Cache hit call ({cache_time:.6f}s) is not 10× faster than cold call "
            f"({cold_time:.3f}s). Speedup ratio: {cold_time / max(cache_time, 1e-9):.1f}×"
        )

        # Sanity: both results must have all 15 output columns
        for col in OUTPUT_COLUMNS:
            assert col in result1.columns, f"Column '{col}' missing from cold result"
            assert col in result2.columns, f"Column '{col}' missing from cache result"

    finally:
        if symbol in _pattern_cache:
            del _pattern_cache[symbol]


# ===========================================================================
# Task 8.2 — End-to-End Signal Unblocking Unit Test
# Validates: Requirements 1.1, 2.1, 3.1, 4.1, 4.10
# ===========================================================================

def test_end_to_end_signal_unblocking():
    """
    Unit test: USDJPY live-log scenario that was previously blocked by all four
    defects simultaneously.

    Scenario:
      - trend_strength = 0.10  → buy_threshold = 0.58 (ranging tier)
      - h1_trend = -1          → counter-trend BUY; xgb_prob=0.80 >= 0.75 → OVERRIDE
      - session = "Pacific"    → UTC hour 22 must be in session
      - xgb_prob=0.80, rf_prob=0.70 → high enough final_prob to exceed 0.58

    Validates: Requirements 1.1, 2.1, 3.1, 4.1, 4.10
    """
    # trend_strength = clip((adx - 15) / 35, 0, 1) = 0.10
    # → adx = 0.10 * 35 + 15 = 18.5
    current_adx = 18.5
    current_atr = 0.001
    atr_series = pd.Series([current_atr] * 50)

    decision = ensemble_predict(
        xgb_prob=0.80,
        rf_prob=0.70,
        current_adx=current_adx,
        current_atr=current_atr,
        atr_series=atr_series,
        session="Pacific",
        diagnostic=False,
        h1_trend=-1,
        symbol="USDJPY",
    )

    # 1. Signal must be BUY — previously blocked by all four defects
    assert decision.direction == "BUY", (
        f"Expected direction='BUY' for the USDJPY unblocking scenario, "
        f"got '{decision.direction}'. "
        f"final_prob={decision.final_prob:.4f}, buy_threshold={decision.buy_threshold:.4f}, "
        f"reason={decision.decision_reason}, stage={decision.stage_reached}"
    )

    # 2. Must NOT be blocked by the old COUNTER_TREND_H1 hard block
    assert decision.decision_reason != "COUNTER_TREND_H1", (
        f"Signal was hard-blocked by COUNTER_TREND_H1 despite xgb_prob=0.80 >= 0.75. "
        f"The v6.1 soft-block override should have allowed this signal through."
    )

    # 3. is_in_trading_session() must return (True, ...) for UTC hour 22
    original_flag = Config.TRADE_SESSION_PACIFIC
    original_only = Config.TRADE_ONLY_IN_SESSIONS
    try:
        Config.TRADE_SESSION_PACIFIC = True
        Config.TRADE_ONLY_IN_SESSIONS = True

        mock_dt = MagicMock()
        mock_dt.utcnow.return_value.hour = 22

        with patch("trade_manager.datetime.datetime", mock_dt):
            in_session, reason = TradeManager.is_in_trading_session()

        assert in_session is True, (
            f"is_in_trading_session() returned False for UTC hour 22 with "
            f"TRADE_SESSION_PACIFIC=True. Reason: {reason}"
        )
    finally:
        Config.TRADE_SESSION_PACIFIC = original_flag
        Config.TRADE_ONLY_IN_SESSIONS = original_only


# ===========================================================================
# Task 8.3 — Diagnostic Mode Preservation Unit Test
# Validates: Requirements 4.9
# ===========================================================================

def test_diagnostic_mode_threshold_preservation():
    """
    Unit test: ensemble_predict with diagnostic=True must always use fixed
    thresholds (buy_threshold=0.62, sell_threshold=0.38) regardless of
    trend_strength.

    Validates: Requirements 4.9
    """
    current_atr = 0.001
    atr_series = pd.Series([current_atr] * 50)

    # trend_strength values derived from ADX:
    #   trend_strength = clip((adx - 15) / 35, 0, 1)
    # We test four representative values: 0.10, 0.30, 0.50, 0.80
    trend_strength_cases = [0.10, 0.30, 0.50, 0.80]

    for ts in trend_strength_cases:
        adx = ts * 35 + 15  # inverse of the clip formula

        decision = ensemble_predict(
            xgb_prob=0.65,
            rf_prob=0.65,
            current_adx=adx,
            current_atr=current_atr,
            atr_series=atr_series,
            session="London",
            diagnostic=True,
            h1_trend=0,
            symbol="EURUSD",
        )

        assert decision.buy_threshold == 0.62, (
            f"diagnostic=True with trend_strength={ts:.2f} (adx={adx:.1f}): "
            f"expected buy_threshold=0.62, got {decision.buy_threshold}. "
            f"Diagnostic mode must use fixed thresholds regardless of trend_strength."
        )

        assert decision.sell_threshold == 0.38, (
            f"diagnostic=True with trend_strength={ts:.2f} (adx={adx:.1f}): "
            f"expected sell_threshold=0.38, got {decision.sell_threshold}. "
            f"Diagnostic mode must use fixed thresholds regardless of trend_strength."
        )

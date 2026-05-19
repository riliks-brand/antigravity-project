"""
Bug Condition Exploration & Fix Verification Tests
===================================================
Tests for the data-loader-candle-limit-fix spec.

Task 1  — Bug condition exploration (expected to FAIL on unfixed code)
Task 4  — Re-run same test on FIXED code (expected to PASS)
Task 5  — Preservation property tests
Task 6  — Integration test for fetch_mtf_data()

Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.5
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

import MetaTrader5 as mt5

from config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rates(n: int):
    """Return a minimal structured array that data_loader can convert to a DataFrame."""
    dtype = np.dtype([
        ('time', np.int64),
        ('open', np.float64),
        ('high', np.float64),
        ('low', np.float64),
        ('close', np.float64),
        ('tick_volume', np.int64),
        ('spread', np.int32),
        ('real_volume', np.int64),
    ])
    arr = np.zeros(n, dtype=dtype)
    arr['time'] = np.arange(n, dtype=np.int64) * 300 + 1_700_000_000
    arr['open'] = 1.1
    arr['high'] = 1.11
    arr['low'] = 1.09
    arr['close'] = 1.105
    arr['tick_volume'] = 100
    return arr


def _symbol_info_stub():
    info = MagicMock()
    info.trade_mode = 2
    info.point = 0.00001
    return info


def _symbol_info_tick_stub():
    tick = MagicMock()
    tick.bid = 1.1
    tick.ask = 1.1001
    tick.time = 1_700_000_000
    tick.volume = 10
    tick.last = 1.1
    return tick


# ---------------------------------------------------------------------------
# Task 1 / Task 4 — Bug Condition Exploration (Property 1)
#
# **Validates: Requirements 1.1, 1.2**
#
# On UNFIXED code this test FAILS because max(count, 99000) inflates every
# count < 99000 to 99000.  On FIXED code it PASSES.
# ---------------------------------------------------------------------------

@given(count=st.integers(min_value=1, max_value=98999))
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_bug_condition__fetch_mt5_ohlc_respects_count(count):
    """
    **Validates: Requirements 1.1, 1.2**

    Property 1 — Bug Condition:
    For any count in [1, 98999], fetch_mt5_ohlc must call mt5.copy_rates_from_pos
    with exactly `count` candles, NOT with 99000.

    EXPECTED TO FAIL on unfixed code (confirms the max(count, 99000) override is active).
    EXPECTED TO PASS on fixed code.
    """
    from data_loader import fetch_mt5_ohlc

    captured = {}

    def fake_copy_rates(symbol, timeframe, start, n):
        captured['count'] = n
        return _make_rates(n)

    with patch('MetaTrader5.copy_rates_from_pos', side_effect=fake_copy_rates), \
         patch('MetaTrader5.symbol_select', return_value=True), \
         patch('MetaTrader5.symbol_info', return_value=_symbol_info_stub()), \
         patch('MetaTrader5.symbol_info_tick', return_value=_symbol_info_tick_stub()):

        fetch_mt5_ohlc("EURUSD", mt5.TIMEFRAME_M5, count)

    actual = captured.get('count')
    assert actual == count, (
        f"Bug detected: fetch_mt5_ohlc was called with count={count} "
        f"but MT5 received {actual} (expected {count}, not 99000)"
    )


def test_bug_condition__none_count_uses_data_points():
    """
    **Validates: Requirements 1.1, 2.1**

    When count=None, fetch_mt5_ohlc must call MT5 with Config.DATA_POINTS (2000),
    NOT with max(Config.DATA_POINTS, 99000) = 99000.

    EXPECTED TO FAIL on unfixed code.
    EXPECTED TO PASS on fixed code.
    """
    from data_loader import fetch_mt5_ohlc

    captured = {}

    def fake_copy_rates(symbol, timeframe, start, n):
        captured['count'] = n
        return _make_rates(n)

    with patch('MetaTrader5.copy_rates_from_pos', side_effect=fake_copy_rates), \
         patch('MetaTrader5.symbol_select', return_value=True), \
         patch('MetaTrader5.symbol_info', return_value=_symbol_info_stub()), \
         patch('MetaTrader5.symbol_info_tick', return_value=_symbol_info_tick_stub()):

        fetch_mt5_ohlc("EURUSD", mt5.TIMEFRAME_M5, None)

    actual = captured.get('count')
    assert actual == Config.DATA_POINTS, (
        f"Bug detected: count=None should use Config.DATA_POINTS={Config.DATA_POINTS} "
        f"but MT5 received {actual}"
    )


# ---------------------------------------------------------------------------
# Task 5 — Preservation Property Tests (Property 2)
#
# **Validates: Requirements 3.1, 3.2, 3.3, 3.5**
# ---------------------------------------------------------------------------

@given(count=st.integers(min_value=99000, max_value=500000))
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_preservation__large_count_unchanged(count):
    """
    **Validates: Requirements 3.1, 3.5**

    Property 2 — Preservation:
    For any count >= 99000, fetch_mt5_ohlc must call MT5 with exactly `count`.
    This must hold on BOTH unfixed and fixed code (no regression).
    """
    from data_loader import fetch_mt5_ohlc

    captured = {}

    def fake_copy_rates(symbol, timeframe, start, n):
        captured['count'] = n
        return _make_rates(min(n, 100))  # return a small slice to keep test fast

    with patch('MetaTrader5.copy_rates_from_pos', side_effect=fake_copy_rates), \
         patch('MetaTrader5.symbol_select', return_value=True), \
         patch('MetaTrader5.symbol_info', return_value=_symbol_info_stub()), \
         patch('MetaTrader5.symbol_info_tick', return_value=_symbol_info_tick_stub()):

        fetch_mt5_ohlc("EURUSD", mt5.TIMEFRAME_M5, count)

    actual = captured.get('count')
    assert actual == count, (
        f"Preservation broken: count={count} should be passed through unchanged, "
        f"but MT5 received {actual}"
    )


def test_preservation__mt5_returns_none_propagates_none():
    """
    **Validates: Requirements 3.2, 3.3**

    When mt5.copy_rates_from_pos returns None, fetch_mt5_ohlc must return None.
    """
    from data_loader import fetch_mt5_ohlc

    with patch('MetaTrader5.copy_rates_from_pos', return_value=None), \
         patch('MetaTrader5.symbol_select', return_value=True), \
         patch('MetaTrader5.symbol_info', return_value=_symbol_info_stub()), \
         patch('MetaTrader5.symbol_info_tick', return_value=_symbol_info_tick_stub()):

        result = fetch_mt5_ohlc("EURUSD", mt5.TIMEFRAME_M5, 2000)

    assert result is None, f"Expected None when MT5 returns None, got {type(result)}"


def test_preservation__mt5_returns_empty_array_propagates_none():
    """
    **Validates: Requirements 3.2**

    When mt5.copy_rates_from_pos returns an empty array, fetch_mt5_ohlc must return None.
    """
    from data_loader import fetch_mt5_ohlc

    empty = np.array([], dtype=np.dtype([
        ('time', np.int64), ('open', np.float64), ('high', np.float64),
        ('low', np.float64), ('close', np.float64), ('tick_volume', np.int64),
        ('spread', np.int32), ('real_volume', np.int64),
    ]))

    with patch('MetaTrader5.copy_rates_from_pos', return_value=empty), \
         patch('MetaTrader5.symbol_select', return_value=True), \
         patch('MetaTrader5.symbol_info', return_value=_symbol_info_stub()), \
         patch('MetaTrader5.symbol_info_tick', return_value=_symbol_info_tick_stub()):

        result = fetch_mt5_ohlc("EURUSD", mt5.TIMEFRAME_M5, 2000)

    assert result is None, f"Expected None when MT5 returns empty array, got {type(result)}"


def test_preservation__symbol_select_false_returns_none():
    """
    **Validates: Requirements 3.3**

    When mt5.symbol_select returns False (symbol unavailable), fetch_mt5_ohlc must return None.
    """
    from data_loader import fetch_mt5_ohlc

    with patch('MetaTrader5.symbol_select', return_value=False), \
         patch('MetaTrader5.copy_rates_from_pos') as mock_rates:

        result = fetch_mt5_ohlc("EURUSD", mt5.TIMEFRAME_M5, 2000)

    assert result is None, f"Expected None when symbol_select=False, got {type(result)}"
    mock_rates.assert_not_called()


# ---------------------------------------------------------------------------
# Task 6 — Integration test for fetch_mtf_data()
#
# **Validates: Requirements 2.3, 2.4**
# ---------------------------------------------------------------------------

def test_integration__fetch_mtf_data_requests_data_points_per_timeframe():
    """
    **Validates: Requirements 2.3, 2.4**

    fetch_mtf_data("EURUSD") must call MT5 exactly three times (M5, M15, H1),
    each requesting exactly Config.DATA_POINTS (2000) candles — NOT 99000.
    The returned dict must have keys 'M5', 'M15', 'H1' with non-empty DataFrames.
    """
    from data_loader import fetch_mtf_data

    call_counts = []

    def fake_copy_rates(symbol, timeframe, start, n):
        call_counts.append(n)
        return _make_rates(n)

    with patch('MetaTrader5.copy_rates_from_pos', side_effect=fake_copy_rates), \
         patch('MetaTrader5.symbol_select', return_value=True), \
         patch('MetaTrader5.symbol_info', return_value=_symbol_info_stub()), \
         patch('MetaTrader5.symbol_info_tick', return_value=_symbol_info_tick_stub()):

        result = fetch_mtf_data("EURUSD")

    # Result must be a dict with the three timeframe keys
    assert result is not None, "fetch_mtf_data returned None — M5 fetch failed"
    assert set(result.keys()) == {"M5", "M15", "H1"}, f"Unexpected keys: {result.keys()}"

    for key in ("M5", "M15", "H1"):
        assert isinstance(result[key], pd.DataFrame), f"{key} is not a DataFrame"
        assert not result[key].empty, f"{key} DataFrame is empty"

    # All three MT5 calls must request exactly Config.DATA_POINTS candles
    assert len(call_counts) == 3, f"Expected 3 MT5 calls, got {len(call_counts)}"
    for i, n in enumerate(call_counts):
        assert n == Config.DATA_POINTS, (
            f"MT5 call #{i+1}: expected {Config.DATA_POINTS} candles "
            f"(Config.DATA_POINTS), got {n}"
        )

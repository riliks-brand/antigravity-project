# Implementation Plan: data-loader-candle-limit-fix

## Overview

Surgical two-file fix: remove the `max(count, 99000)` override in `fetch_mt5_ohlc()` and replace
it with `count = count or Config.DATA_POINTS`. Add a clarifying comment to `Config.DATA_POINTS`
in `config.py`. No new modules are introduced.

## Tasks

- [ ] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - `fetch_mt5_ohlc` Ignores Requested Count
  - **CRITICAL**: This test MUST FAIL on unfixed code — failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **GOAL**: Surface counterexamples that demonstrate the `max(count, 99000)` override is active
  - **Scoped PBT Approach**: Scope the property to `count` values in `[1, 98999]` — any value below 99,000 is inflated by the bug
  - File: `tests/test_data_loader_candle_fix.py`
  - Mock `mt5.copy_rates_from_pos` to capture the `count` argument it receives; also mock `mt5.symbol_select`, `mt5.symbol_info`, and `mt5.symbol_info_tick` to return truthy stubs so the function reaches the MT5 call
  - Use Hypothesis `@given(count=st.integers(min_value=1, max_value=98999))` to generate count values
  - Call `fetch_mt5_ohlc("EURUSD", mt5.TIMEFRAME_M5, count)` and assert the captured MT5 call count equals `count` (not 99,000)
  - Also add a concrete case: `fetch_mt5_ohlc("EURUSD", mt5.TIMEFRAME_M5, None)` — assert MT5 is called with `Config.DATA_POINTS` (2,000), not 99,000
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS — MT5 receives 99,000 instead of the requested count (confirms bug)
  - Document counterexamples found (e.g. "count=500 → MT5 called with 99000 instead of 500")
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2_

- [ ] 2. Apply fix to `data_loader.py`
  - Remove the entire `v6.0 FIX` override block (lines ~40–46):
    ```python
    # REMOVE:
    if count is None:
        count = max(Config.DATA_POINTS, 99000)
    else:
        count = max(count, 99000)
    ```
  - Replace with a single default assignment immediately after the `timeframe` default line:
    ```python
    count = count or Config.DATA_POINTS
    ```
  - Update the `fetch_mt5_ohlc` docstring: remove the stale `v6.0 FIX` note; update the `count` param description to read "Number of candles to fetch (default: Config.DATA_POINTS)"
  - _Bug_Condition: isBugCondition(count) where max(count or DATA_POINTS, 99000) != (count or DATA_POINTS)_
  - _Expected_Behavior: MT5 is called with exactly (count or Config.DATA_POINTS) candles_
  - _Preservation: callers passing count >= 99000 continue to receive the full requested count_
  - _Requirements: 2.1, 2.2, 3.1, 3.2, 3.3, 3.5_

- [ ] 3. Apply comment fix to `config.py`
  - Locate the `DATA_POINTS = 2000` line in `config.py`
  - Replace the bare assignment with the annotated version:
    ```python
    DATA_POINTS = 2000  # Live trading candle window (M5/M15/H1 signal generation).
                        # NOT a historical backfill size. Increase only if indicators
                        # require a longer lookback (e.g. 200-period MA on H1).
    ```
  - No logic changes — comment only
  - _Requirements: 2.1_

- [ ] 4. Write fix-checking property tests (Hypothesis)
  - **Property 1: Expected Behavior** - `fetch_mt5_ohlc` Respects the `count` Parameter After Fix
  - **IMPORTANT**: Re-run the SAME test from task 1 — do NOT write a new test
  - The test from task 1 encodes the expected behavior; when it passes, the fix is confirmed
  - Run the bug condition exploration test from task 1 on the FIXED code
  - **EXPECTED OUTCOME**: Test PASSES — MT5 receives exactly the requested count for all values in `[1, 98999]`
  - Also verify the `count=None` case: MT5 is called with `Config.DATA_POINTS` (2,000)
  - Also verify the large-count preservation boundary: `fetch_mt5_ohlc("EURUSD", mt5.TIMEFRAME_M5, 150000)` → MT5 called with 150,000 (unchanged by fix)
  - _Requirements: 2.1, 2.2_

- [ ] 5. Write preservation property tests
  - **Property 2: Preservation** - Large Count Requests Are Unchanged After Fix
  - File: `tests/test_data_loader_candle_fix.py`
  - **IMPORTANT**: Follow observation-first methodology — observe behavior on UNFIXED code for `count >= 99000` inputs first
  - Observe: `fetch_mt5_ohlc("EURUSD", M5, 150000)` on unfixed code → MT5 called with 150,000 (no inflation — already above the floor)
  - Write property-based test: use Hypothesis `@given(count=st.integers(min_value=99000, max_value=500000))` and assert MT5 is called with exactly `count` on the fixed code
  - Also write unit tests for error-path preservation:
    - When `mt5.copy_rates_from_pos` returns `None` → `fetch_mt5_ohlc()` returns `None` (requirement 3.2, 3.3)
    - When `mt5.copy_rates_from_pos` returns an empty array → `fetch_mt5_ohlc()` returns `None` (requirement 3.2)
    - When `mt5.symbol_select` returns `False` → `fetch_mt5_ohlc()` returns `None` (requirement 3.3)
  - Verify all preservation tests PASS on the fixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.5_

- [ ] 6. Write integration test for `fetch_mtf_data()`
  - File: `tests/test_data_loader_candle_fix.py`
  - Mock `fetch_mt5_ohlc` (or mock MT5 at the `mt5.copy_rates_from_pos` level) to capture call arguments
  - Call `fetch_mtf_data("EURUSD")` on the fixed code
  - Assert all three timeframe calls (M5, M15, H1) each request exactly `Config.DATA_POINTS` (2,000) candles — not 99,000
  - Assert the returned dict has keys `"M5"`, `"M15"`, `"H1"` and each value is a non-empty DataFrame
  - _Requirements: 2.3, 2.4_

- [ ] 7. Final checkpoint — Ensure all tests pass
  - Run `pytest tests/test_data_loader_candle_fix.py -v` and confirm all tests pass
  - Verify no regressions in the existing test suite: `pytest tests/ -q`
  - Confirm `data_loader.py` no longer contains `max(count, 99000)` or `max(Config.DATA_POINTS, 99000)`
  - Confirm `config.py` `DATA_POINTS` line now has the clarifying comment
  - Ask the user if any questions arise before closing the spec

## Notes

- Tasks 2 and 3 are independent and can be applied in either order
- Task 4 re-uses the test written in Task 1 — no new test file needed
- All mocking targets `mt5.copy_rates_from_pos` to capture the `count` argument; use `unittest.mock.patch` or `pytest-mock`
- Hypothesis is already present in the project (`.hypothesis/` directory exists); no new install needed
- The `count = count or Config.DATA_POINTS` idiom treats `count=0` as falsy and falls back to `Config.DATA_POINTS` — this is intentional and correct (a zero-candle request is nonsensical)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2", "3"] },
    { "id": 2, "tasks": ["4", "5", "6"] },
    { "id": 3, "tasks": ["7"] }
  ]
}
```

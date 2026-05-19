# Data Loader Candle Limit Fix — Bugfix Design

## Overview

`fetch_mt5_ohlc()` in `data_loader.py` contains a hardcoded override that forces every candle
request to a minimum of 99,000 candles, regardless of the caller-supplied `count` or the
configured `Config.DATA_POINTS = 2000`. The fix removes that override entirely and replaces it
with a single, clean default: `count = count or Config.DATA_POINTS`. A comment is also added to
`config.py` to clarify that `DATA_POINTS` represents the live-trading candle window, not a
historical backfill size.

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug — `fetch_mt5_ohlc()` is called with
  any `count` value (including `None`) and the `max(count, 99000)` block overrides it to 99,000.
- **Property (P)**: The desired behavior — `fetch_mt5_ohlc()` returns a DataFrame whose length
  does not exceed the requested `count` (subject to MT5 availability), and the MT5 API is called
  with exactly the requested count, not 99,000.
- **Preservation**: All existing behaviors for callers that do not rely on the 99,000 override
  must remain unchanged after the fix.
- **`fetch_mt5_ohlc(symbol, timeframe, count)`**: The function in `data_loader.py` that fetches
  OHLC candles from MetaTrader 5. It is the sole location of the defect.
- **`Config.DATA_POINTS`**: Class attribute in `config.py` (currently `2000`) that defines the
  default candle window for live trading signal generation.
- **`fetch_mtf_data(symbol)`**: Caller in `data_loader.py` that invokes `fetch_mt5_ohlc()` three
  times (M5, M15, H1) with `Config.DATA_POINTS` as the explicit `count` argument.

## Bug Details

### Bug Condition

The bug manifests whenever `fetch_mt5_ohlc()` is called — whether `count` is `None` or an
explicit value. The `max(count, 99000)` block unconditionally replaces the intended count with
99,000, causing the MT5 API to be called with 99,000 instead of the requested value.

**Formal Specification:**
```
FUNCTION isBugCondition(count)
  INPUT:  count — the value passed to fetch_mt5_ohlc() (None or integer)
  OUTPUT: boolean

  effective_count := count if count is not None else Config.DATA_POINTS
  RETURN max(effective_count, 99000) != effective_count
         -- i.e., the override inflates the count above what was requested
END FUNCTION
```

Because `Config.DATA_POINTS = 2000` and `max(2000, 99000) = 99000`, the condition is `True`
for every normal live-trading call. It would only be `False` if the caller explicitly passed
`count >= 99000`.

### Examples

- `fetch_mt5_ohlc("EURUSD", M5, None)` → MT5 called with 99,000 (expected: 2,000)
- `fetch_mt5_ohlc("XAUUSD", M5, Config.DATA_POINTS)` → MT5 called with 99,000 (expected: 2,000)
- `fetch_mt5_ohlc("GBPUSD", H1, 500)` → MT5 called with 99,000 (expected: 500)
- `fetch_mt5_ohlc("EURUSD", M5, 150000)` → MT5 called with 150,000 (not a bug — caller requested more than 99,000)

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Callers that pass `count > Config.DATA_POINTS` (e.g. historical backfill scripts) must
  continue to receive the full requested count without truncation.
- When MT5 returns fewer candles than requested (limited history, weekend gap, new symbol),
  `fetch_mt5_ohlc()` must continue to return whatever data MT5 provides without error.
- When a symbol is unavailable or the market is closed, `fetch_mt5_ohlc()` must continue to
  return `None` and log the appropriate error.
- The PatternDetector cache and incremental update logic must remain completely unaffected.
- Changing `Config.DATA_POINTS` to any value must continue to be respected as the default.

**Scope:**
All callers that do NOT rely on the 99,000 minimum (i.e. every normal live-trading call) are
the bug condition. Callers that explicitly request ≥ 99,000 candles are outside the bug
condition and must be preserved unchanged.

## Hypothesized Root Cause

Based on the comment in the source code (`v6.0 FIX: Force minimum 99K candles for all symbols`)
and the Arabic inline comment, the override was introduced as a workaround for an earlier
inconsistency where EURUSD was fetching only 20,000 candles while other symbols fetched 99,000.
The correct fix at that time would have been to standardise on `Config.DATA_POINTS`, but instead
a hard floor of 99,000 was applied globally.

1. **Misdiagnosed Root Cause**: The original inconsistency was likely caused by a per-symbol
   `DATA_POINTS` override that has since been removed. The 99,000 floor was never the right fix.

2. **Stale Override Not Cleaned Up**: The `v6.0 FIX` block was never revisited after the
   underlying inconsistency was resolved, leaving the override permanently active.

3. **`Config.DATA_POINTS` Semantics Unclear**: The config value lacked a comment explaining
   that it is the live-trading window (not a historical backfill size), making it easy for
   developers to distrust it and reach for a hardcoded large number.

4. **No Test Coverage**: There were no tests asserting that `fetch_mt5_ohlc()` respects the
   `count` parameter, so the regression went undetected.

## Correctness Properties

Property 1: Bug Condition — `fetch_mt5_ohlc` Respects the `count` Parameter

_For any_ call to `fetch_mt5_ohlc(symbol, timeframe, count)` where `isBugCondition(count)` is
true (i.e. the requested count is less than 99,000), the fixed function SHALL invoke the MT5
API with exactly `count` candles (or `Config.DATA_POINTS` when `count` is `None`), and SHALL
NOT inflate the request to 99,000.

**Validates: Requirements 2.1, 2.2**

Property 2: Preservation — Large or Explicit Count Requests Are Unchanged

_For any_ call to `fetch_mt5_ohlc(symbol, timeframe, count)` where `isBugCondition(count)` is
false (i.e. `count >= 99000`), the fixed function SHALL produce the same MT5 API call and
return the same result as the original function, preserving all existing large-fetch behaviour.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

## Fix Implementation

### Changes Required

**File**: `data_loader.py`

**Function**: `fetch_mt5_ohlc()`

**Specific Changes**:

1. **Remove the `v6.0 FIX` override block** (lines ~40–46 in the current file):
   ```python
   # REMOVE THIS ENTIRE BLOCK:
   # v6.0 FIX: Force minimum 99K candles for all symbols
   if count is None:
       count = max(Config.DATA_POINTS, 99000)
   else:
       count = max(count, 99000)
   ```

2. **Replace with a single default assignment**:
   ```python
   count = count or Config.DATA_POINTS
   ```

3. **Update the docstring** to remove the stale `v6.0 FIX` reference and accurately describe
   the `count` parameter default.

---

**File**: `config.py`

**Specific Changes**:

4. **Add a clarifying comment** next to `DATA_POINTS`:
   ```python
   DATA_POINTS = 2000  # Live trading candle window (M5/M15/H1 signal generation).
                       # NOT a historical backfill size. Increase only if indicators
                       # require a longer lookback (e.g. 200-period MA on H1).
   ```

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that
demonstrate the bug on the unfixed code, then verify the fix works correctly and preserves
existing behaviour.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix.
Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Mock `mt5.copy_rates_from_pos` to capture the `count` argument it receives.
Call `fetch_mt5_ohlc()` with various `count` values and assert that the MT5 API is called with
the requested count. Run these tests on the UNFIXED code to observe failures.

**Test Cases**:
1. **Default count test**: Call `fetch_mt5_ohlc("EURUSD", M5, None)` — assert MT5 is called
   with 2,000 (will fail on unfixed code: MT5 receives 99,000)
2. **Explicit small count test**: Call `fetch_mt5_ohlc("EURUSD", M5, 500)` — assert MT5 is
   called with 500 (will fail on unfixed code: MT5 receives 99,000)
3. **`Config.DATA_POINTS` passthrough test**: Call with `count=Config.DATA_POINTS` — assert
   MT5 is called with 2,000 (will fail on unfixed code)
4. **Large count edge case**: Call with `count=150000` — assert MT5 is called with 150,000
   (should pass on both unfixed and fixed code — this is the preservation boundary)

**Expected Counterexamples**:
- MT5 API is called with 99,000 instead of the requested count for all inputs < 99,000.
- Confirms root cause: the `max(count, 99000)` block is the sole source of the inflation.

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function invokes
MT5 with the correct count.

**Pseudocode:**
```
FOR ALL count WHERE isBugCondition(count) DO
  mock_mt5_copy_rates_from_pos(capture_args=True)
  fetch_mt5_ohlc_fixed("EURUSD", TIMEFRAME_M5, count)
  actual_count := captured_args.count
  ASSERT actual_count == (count if count is not None else Config.DATA_POINTS)
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold (count ≥ 99,000),
the fixed function produces the same MT5 API call as the original.

**Pseudocode:**
```
FOR ALL count WHERE NOT isBugCondition(count) DO
  ASSERT fetch_mt5_ohlc_original(count) == fetch_mt5_ohlc_fixed(count)
  -- specifically: MT5 is called with the same count in both versions
END FOR
```

**Testing Approach**: Property-based testing is recommended for the fix-checking property
because the count parameter spans a large integer domain. Hypothesis can generate many values
in the range `[1, 98999]` and verify that every one is passed through correctly after the fix.

**Test Cases**:
1. **Preservation of large counts**: Verify `count=150000` still calls MT5 with 150,000 after fix.
2. **Preservation of error paths**: Verify that `None` return from MT5 (symbol unavailable)
   still returns `None` from `fetch_mt5_ohlc()` after fix.
3. **Preservation of partial data**: Verify that when MT5 returns fewer rows than requested,
   the DataFrame is returned as-is (no truncation or error).

### Unit Tests

- Test `fetch_mt5_ohlc(count=None)` calls MT5 with `Config.DATA_POINTS` (2,000)
- Test `fetch_mt5_ohlc(count=500)` calls MT5 with exactly 500
- Test `fetch_mt5_ohlc(count=150000)` calls MT5 with exactly 150,000 (preservation)
- Test that a `None` return from MT5 propagates as `None` from the function
- Test that an empty rates array from MT5 propagates as `None` from the function

### Property-Based Tests

- Generate random `count` values in `[1, 98999]` and verify the fixed function calls MT5 with
  exactly that count (fix-checking property — Property 1)
- Generate random `count` values in `[99000, 500000]` and verify the fixed function calls MT5
  with exactly that count (preservation property — Property 2)
- Generate random `count` values across the full positive integer domain and verify the MT5
  call count always equals `count or Config.DATA_POINTS` (combined property)

### Integration Tests

- Call `fetch_mtf_data("EURUSD")` with MT5 mocked and verify all three timeframe calls
  (M5, M15, H1) each request exactly `Config.DATA_POINTS` candles
- Verify the full portfolio evaluation cycle completes within the expected time budget when
  `Config.DATA_POINTS = 2000` (smoke test against the 3–5 minute regression)

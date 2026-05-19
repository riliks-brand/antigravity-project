# Implementation Plan: bot-performance-trading-fix

## Overview

Four surgical fixes to unblock trade execution: PatternDetector caching (Fix 1), COUNTER_TREND_H1 softening (Fix 2), Pacific session coverage (Fix 3), and BUY/SELL threshold recalibration (Fix 4). All changes are confined to `config.py`, `pattern_detector.py`, `features.py`, `ensemble_engine.py`, and `trade_manager.py`. No new modules are introduced and the main loop in `main.py` requires no changes.

## Tasks

- [x] 1. Add new config parameters for fixes 2, 3, and 4
  - Open `config.py` and append the following parameter groups:
    - **Fix 2 — Counter-trend softening**: `COUNTER_TREND_XGB_OVERRIDE_THRESHOLD = 0.75`, `COUNTER_TREND_PENALTY = -0.04`
    - **Fix 3 — Pacific session**: `SESSION_PACIFIC = (22, 24)`, `TRADE_SESSION_PACIFIC = True`, `PACIFIC_POSITION_SIZE_MODIFIER = 0.5`
    - **Fix 4 — Recalibrated thresholds**: `PROB_THRESHOLD_BUY_RANGING = 0.58`, `PROB_THRESHOLD_BUY_TRANSITIONING = 0.57`, `PROB_THRESHOLD_BUY_TRENDING = 0.56`, `PROB_THRESHOLD_SELL_RANGING = 0.42`, `PROB_THRESHOLD_SELL_TRANSITIONING = 0.43`, `PROB_THRESHOLD_SELL_TRENDING = 0.44`
  - Group each set of params with a comment block identifying the fix number
  - _Requirements: 2.7, 3.3, 3.4, 4.7, 4.8_

- [x] 2. Implement PatternDetector caching (Fix 1)
  - [x] 2.1 Add module-level cache dict and `OUTPUT_COLUMNS` constant to `pattern_detector.py`
    - Define `_pattern_cache: dict[str, dict] = {}` at module level
    - Define `OUTPUT_COLUMNS` list with all 15 column names: the 14 pattern flag columns plus `pattern_bias_score`
    - Define `CACHE_LOOKBACK = 1000` constant
    - _Requirements: 1.1, 1.6, 1.8_

  - [x] 2.2 Update `add_chart_patterns()` signature and add caching logic in `pattern_detector.py`
    - Change signature to `add_chart_patterns(df: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame`
    - Implement the three-branch logic:
      - **Cache hit** (`cached["last_row_count"] == n`): copy cached arrays back into `df`, log "Cache hit — N rows, 0 new rows processed", return early
      - **Incremental update** (`n > cached["last_row_count"]`): compute only `df[max(0, n - CACHE_LOOKBACK):n]`, merge with cached arrays, log new row count and lookback window
      - **Full computation** (no cache or `n < cached["last_row_count"]`): run existing rolling-window logic unchanged
    - After any computation path, store result in `_pattern_cache[symbol]` with `last_row_count`, `last_index`, and `columns` dict
    - Add cache invalidation: if `df.index[-1] != cached["last_index"]` but row count is equal, treat as cache miss and recompute fully
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 1.6, 1.7_

  - [x] 2.3 Write property test for cache correctness (Property 1)
    - **Property 1: Cache correctness — cached results equal fresh results**
    - **Validates: Requirements 1.1, 1.5**
    - File: `tests/test_bot_perf_fix.py`
    - Use Hypothesis `@given` with an OHLC DataFrame strategy and symbol string strategy
    - Clear cache for the symbol before each test run
    - Call `add_chart_patterns(df.copy(), symbol)` twice; assert all 15 `OUTPUT_COLUMNS` are identical between calls
    - Assert second call log contains "Cache hit"

  - [x] 2.4 Write property test for incremental correctness (Property 2)
    - **Property 2: Incremental correctness — previously computed rows are unchanged**
    - **Validates: Requirements 1.2, 1.5**
    - File: `tests/test_bot_perf_fix.py`
    - Generate a base DataFrame of N rows and an extended DataFrame of N+k rows (k ≥ 1)
    - Call `add_chart_patterns` on base (cold cache), then on extended
    - Assert that for all rows 0..N-1, the pattern flag values in the extended result match a fresh (no-cache) call on the same extended DataFrame

  - [x] 2.5 Write property test for output completeness (Property 3)
    - **Property 3: Output completeness — all 15 columns always present**
    - **Validates: Requirements 1.8**
    - File: `tests/test_bot_perf_fix.py`
    - Use Hypothesis with `min_rows=50` OHLC DataFrame strategy
    - Assert all 15 `OUTPUT_COLUMNS` are present in the returned DataFrame for every generated input

  - [x] 2.6 Write property test for cache isolation (Property 4)
    - **Property 4: Cache isolation — per-symbol independence**
    - **Validates: Requirements 1.6**
    - File: `tests/test_bot_perf_fix.py`
    - Generate two distinct symbol names and two distinct OHLC DataFrames
    - Call `add_chart_patterns` for symbol A, then modify the cache for symbol B
    - Assert that the cached result for symbol A is unchanged

- [x] 3. Update `features.py` to pass symbol to `add_chart_patterns()`
  - Locate all call sites of `add_chart_patterns(df)` in `features.py`
  - Update each call to `add_chart_patterns(df, symbol=symbol)`, passing the symbol name that is already available in the calling context
  - Verify no other callers of `add_chart_patterns` exist in the codebase; update any found
  - _Requirements: 1.1, 1.6_

- [x] 4. Implement COUNTER_TREND_H1 softening (Fix 2)
  - [x] 4.1 Replace the hard-block Step 12 in `ensemble_engine.py` with the soft-block logic
    - Remove the existing `# v6.0: H1 counter-trend HARD BLOCK` block that returns HOLD unconditionally
    - Remove the dead-code duplicate `against_h1` assignment and `mtf_penalty = -0.03` that follows it
    - Insert the `# v6.1: H1 counter-trend SOFT BLOCK` replacement:
      - Read `override_threshold` via `getattr(Config, "COUNTER_TREND_XGB_OVERRIDE_THRESHOLD", 0.75)`
      - Read `ct_penalty` via `getattr(Config, "COUNTER_TREND_PENALTY", -0.04)`
      - If `against_h1 and h1_trend != 0`: branch on `xgb_prob >= override_threshold` → `mtf_penalty = 0.0` (log OVERRIDE) vs `mtf_penalty = ct_penalty` (log PENALTY)
      - If not `against_h1` or `h1_trend == 0`: `mtf_penalty = 0.0`
    - Log `xgb_prob`, `h1_trend`, and which branch was taken for every evaluation where `h1_trend != 0`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x] 4.2 Write property test for high-confidence counter-trend override (Property 5)
    - **Property 5: High-confidence counter-trend signals are not hard-blocked**
    - **Validates: Requirements 2.1, 2.2**
    - File: `tests/test_bot_perf_fix.py`
    - Use Hypothesis `@given(xgb_prob=st.floats(min_value=0.75, max_value=0.99), h1_trend=st.sampled_from([-1, 1]))`
    - Call `ensemble_predict` with a BUY side when `h1_trend=-1` and a SELL side when `h1_trend=1`
    - Assert `decision.decision_reason != "COUNTER_TREND_H1"` for all generated inputs

  - [x] 4.3 Write property test for low-confidence counter-trend penalty (Property 6)
    - **Property 6: Low-confidence counter-trend signals receive penalty, not hard block**
    - **Validates: Requirements 2.3, 2.4, 2.10**
    - File: `tests/test_bot_perf_fix.py`
    - Use Hypothesis `@given(xgb_prob=st.floats(min_value=0.50, max_value=0.749))`
    - Assert `decision.decision_reason != "COUNTER_TREND_H1"` (no hard block)
    - Assert `final_prob` is lower than it would be without the counter-trend penalty by exactly `|COUNTER_TREND_PENALTY|`

- [x] 5. Implement Pacific session coverage (Fix 3)
  - [x] 5.1 Update `get_active_session()` in `trade_manager.py` to return `"Pacific"` for UTC hours 22–23
    - After the existing Asia check, add a Pacific branch:
      ```python
      pacific_start, pacific_end = getattr(Config, "SESSION_PACIFIC", (22, 24))
      if pacific_start <= hour or hour < (pacific_end % 24):
          return "Pacific"
      ```
    - Ensure the existing London, New York, and Asia checks remain in their current order and are unchanged
    - _Requirements: 3.1, 3.9_

  - [x] 5.2 Update `is_in_trading_session()` in `trade_manager.py` to handle the Pacific session
    - Before the final fallthrough `return False, "Outside all sessions..."`, insert the Pacific branch:
      - Compute `in_pacific` using the same `SESSION_PACIFIC` range logic as `get_active_session()`
      - If `in_pacific and TRADE_SESSION_PACIFIC`: return `(True, f"In Pacific session (UTC Hour: {hour})")`
      - If `in_pacific and not TRADE_SESSION_PACIFIC`: return `(False, "In Pacific session, but trading is disabled for this session (UTC Hour: {hour})")`
    - _Requirements: 3.2, 3.6, 3.8_

  - [x] 5.3 Apply Pacific position size modifier in `TradeManager.calculate_lot_size()` (or equivalent lot-sizing function)
    - After computing the base `lot_size`, check `if TradeManager.get_active_session() == "Pacific":`
    - Apply `lot_size = lot_size * getattr(Config, "PACIFIC_POSITION_SIZE_MODIFIER", 0.5)`
    - _Requirements: 3.7_

  - [x] 5.4 Add Pacific session bonus to `_compute_session_bonus()` in `ensemble_engine.py`
    - Add `elif session == "Pacific": raw_bonus = 0.0` branch so the function returns a neutral bonus for Pacific
    - _Requirements: 3.5_

  - [x] 5.5 Write property test for Pacific session detection (Property 7)
    - **Property 7: Pacific session detection covers all UTC 22–23 hours**
    - **Validates: Requirements 3.1**
    - File: `tests/test_bot_perf_fix.py`
    - Use Hypothesis `@given(hour=st.sampled_from([22, 23]))`
    - Mock `datetime.datetime.utcnow().hour` to the generated value
    - Assert `TradeManager.get_active_session() == "Pacific"` for all inputs

  - [x] 5.6 Write property test for Pacific session trading flag (Property 8)
    - **Property 8: Pacific session trading respects the TRADE_SESSION_PACIFIC flag**
    - **Validates: Requirements 3.2, 3.8**
    - File: `tests/test_bot_perf_fix.py`
    - Use Hypothesis `@given(hour=st.sampled_from([22, 23]), flag=st.booleans())`
    - Set `Config.TRADE_SESSION_PACIFIC = flag`, mock UTC hour
    - Assert `is_in_trading_session()[0] == flag`

  - [x] 5.7 Write property test for Pacific session bonus (Property 9)
    - **Property 9: Pacific session bonus is always zero**
    - **Validates: Requirements 3.5**
    - File: `tests/test_bot_perf_fix.py`
    - Use Hypothesis `@given(trend_strength=st.floats(min_value=0.0, max_value=1.0))`
    - Assert `_compute_session_bonus("Pacific", trend_strength) == 0.0` for all generated inputs

  - [x] 5.8 Write property test for existing session detection unchanged (Property 10)
    - **Property 10: Existing session detection is unchanged**
    - **Validates: Requirements 3.9**
    - File: `tests/test_bot_perf_fix.py`
    - Use Hypothesis `@given(hour=st.integers(min_value=0, max_value=23))`
    - Assert London for hours in [7, 16), Asia for hours in [0, 9), and that hours 22–23 no longer return "UNKNOWN"

- [x] 6. Implement threshold recalibration (Fix 4)
  - [x] 6.1 Replace Step 13 threshold logic in `ensemble_predict()` in `ensemble_engine.py`
    - Locate the existing Step 13 block that sets `buy_threshold` and `sell_threshold` with hardcoded values
    - Replace with the tiered logic:
      - If `diagnostic=True`: keep `buy_threshold = 0.62`, `sell_threshold = 0.38` (unchanged)
      - If `diagnostic=False` and `trend_strength > 0.35`: read `PROB_THRESHOLD_BUY_TRENDING` (0.56) and `PROB_THRESHOLD_SELL_TRENDING` (0.44)
      - If `diagnostic=False` and `0.25 < trend_strength <= 0.35`: read `PROB_THRESHOLD_BUY_TRANSITIONING` (0.57) and `PROB_THRESHOLD_SELL_TRANSITIONING` (0.43)
      - If `diagnostic=False` and `trend_strength <= 0.25`: read `PROB_THRESHOLD_BUY_RANGING` (0.58) and `PROB_THRESHOLD_SELL_RANGING` (0.42)
    - All reads use `getattr(Config, "PARAM_NAME", default)` for backward compatibility
    - Ensure `buy_threshold` and `sell_threshold` continue to be logged to `ensemble_decisions.csv`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.9, 4.12_

  - [x] 6.2 Write property test for BUY threshold tier selection (Property 11)
    - **Property 11: BUY threshold matches the correct tier for all trend_strength values**
    - **Validates: Requirements 4.1, 4.2, 4.3**
    - File: `tests/test_bot_perf_fix.py`
    - Use Hypothesis `@given(trend_strength=st.floats(min_value=0.0, max_value=1.0))`
    - Derive `adx` from `trend_strength` using the inverse of `clip((adx-15)/35, 0, 1)`
    - Call `ensemble_predict(..., current_adx=adx, diagnostic=False)` and assert the selected `buy_threshold` matches the expected tier value

  - [x] 6.3 Write property test for SELL threshold tier selection (Property 12)
    - **Property 12: SELL threshold matches the correct tier for all trend_strength values**
    - **Validates: Requirements 4.4, 4.5, 4.6**
    - File: `tests/test_bot_perf_fix.py`
    - Use Hypothesis `@given(trend_strength=st.floats(min_value=0.0, max_value=1.0))`
    - Same approach as Property 11 but assert `sell_threshold` matches the expected tier value

- [x] 7. Checkpoint — Ensure all tests pass
  - Run `pytest tests/test_bot_perf_fix.py -v` and confirm all tests pass
  - Verify no regressions in existing test suite: `pytest tests/ -q`
  - Ask the user if any questions arise before proceeding to integration verification

- [x] 8. Integration verification
  - [x] 8.1 Write unit tests for PatternDetector performance
    - File: `tests/test_bot_perf_fix.py`
    - Time a 2000-row call to `add_chart_patterns` and assert completion in under 2.0 seconds
    - Time a second call (cache hit) and assert it is at least 10× faster than the first call
    - _Requirements: 1.4_

  - [x] 8.2 Write unit tests for end-to-end signal unblocking
    - File: `tests/test_bot_perf_fix.py`
    - Construct a scenario matching the USDJPY live log: `final_prob=0.5994`, `trend_strength=0.10`, `h1_trend=-1`, `xgb_prob=0.80`, `session="Pacific"`, UTC hour 22
    - Assert `decision.direction == "BUY"` (previously blocked by all four defects simultaneously)
    - Assert `decision.decision_reason != "COUNTER_TREND_H1"`
    - Assert `is_in_trading_session()` returns `(True, ...)`
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 4.10_

  - [x] 8.3 Write unit tests for diagnostic mode preservation
    - File: `tests/test_bot_perf_fix.py`
    - Call `ensemble_predict(..., diagnostic=True)` with various `trend_strength` values
    - Assert `buy_threshold == 0.62` and `sell_threshold == 0.38` in all cases
    - _Requirements: 4.9_

- [x] 9. Final checkpoint — Ensure all tests pass
  - Run `pytest tests/ -q` and confirm zero failures
  - Confirm `ensemble_decisions.csv` schema is unchanged (all existing columns still present)
  - Confirm `bot.log` shows `[Patterns][SYMBOL] Cache hit` lines on second cycle per symbol
  - Ask the user if any questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP deployment
- All four fixes are independently deployable — Task 1 (config) is the only shared prerequisite
- Fix 1 (caching) requires both `pattern_detector.py` (Tasks 2.1–2.2) and `features.py` (Task 3) to be updated together; deploying only one half will break the symbol-keyed cache
- Property tests use Hypothesis; install with `pip install hypothesis` if not already present
- The `getattr(Config, ..., default)` pattern throughout ensures old `config.py` files without the new params continue to work safely
- Diagnostic mode thresholds (Task 6.1) are intentionally hardcoded — do not read them from Config

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1", "4.1", "5.1", "5.2", "5.3", "5.4", "6.1"] },
    { "id": 2, "tasks": ["2.2", "3"] },
    { "id": 3, "tasks": ["2.3", "2.4", "2.5", "2.6", "4.2", "4.3", "5.5", "5.6", "5.7", "5.8", "6.2", "6.3"] },
    { "id": 4, "tasks": ["8.1", "8.2", "8.3"] }
  ]
}
```

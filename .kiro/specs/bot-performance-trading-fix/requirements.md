# Requirements Document

## Introduction

The trading bot has four critical defects identified from live log analysis that collectively prevent the bot from executing any trades during normal operation. These defects span performance (PatternDetector processing 99,000 candles per cycle), signal filtering (COUNTER_TREND_H1 hard-blocking all BUY signals during H1 downtrends), session coverage (UNKNOWN session blocking off-hours trading), and threshold calibration (BUY threshold too high after isotonic calibration). This document specifies the required fixes, expected post-fix behavior, and regression-prevention criteria for each issue.

## Glossary

- **PatternDetector**: The `pattern_detector.py` module that detects chart patterns (double top/bottom, head & shoulders, triangles, flags, etc.) by running a rolling-window analysis over OHLC candle data.
- **EnsembleEngine**: The `ensemble_engine.py` module that combines XGBoost and Random Forest predictions into a final directional decision (BUY / SELL / HOLD).
- **COUNTER_TREND_H1 Block**: The hard-block logic in `ensemble_engine.py` (Step 12) that returns HOLD when the proposed trade direction opposes the H1 trend.
- **H1 Trend**: The hourly trend direction encoded as `+1` (uptrend), `-1` (downtrend), or `0` (neutral), derived from the H1 timeframe and stored in the `H1_trend` feature column.
- **XGB Confidence**: The raw probability output of the XGBoost model for the current symbol, ranging from 0.0 to 1.0, where values above 0.5 indicate a BUY bias.
- **Session**: The active trading session label — one of `"London"`, `"New York"`, `"Asia"`, or `"UNKNOWN"` — returned by `TradeManager.get_active_session()`.
- **UNKNOWN Session**: The session label returned when the current UTC hour falls outside all defined session windows (London: 07–16, New York: 13–22, Asia: 00–09). At 22:10 UTC this gap covers hours 22–23.
- **BUY Threshold**: The minimum `final_prob` score required for the EnsembleEngine to emit a BUY direction. Currently set to 0.62–0.64 depending on trend strength.
- **Isotonic Calibration**: A post-processing step applied to XGBoost raw probabilities that maps them to a calibrated scale; this tends to compress predictions toward the center (0.5), making raw scores more conservative.
- **Candle Cache**: An in-memory structure that stores the most recently processed candle index and the corresponding pattern results, allowing the PatternDetector to skip recomputation when no new candle has arrived.
- **Pattern Window**: The number of recent candles used for pattern detection. The current implementation uses a 30-candle rolling window but iterates over the full dataset.
- **M5 Candle**: A 5-minute OHLC candle. The bot's primary execution timeframe. A new M5 candle closes every 5 minutes.
- **TradeManager**: The `trade_manager.py` module responsible for session detection, trade guards, risk sizing, and position management.
- **Config**: The `config.py` module containing all tunable parameters including session windows, thresholds, and feature flags.

---

## Requirements

### Requirement 1: PatternDetector Performance Optimization

**User Story:** As a trading bot operator, I want the PatternDetector to process only new candles instead of the full historical dataset on every cycle, so that pattern detection completes in under 2 seconds per symbol and the bot does not miss M5 candle closes.

#### Acceptance Criteria

1. WHEN `add_chart_patterns()` is called with a DataFrame that has the same number of rows as the previous call for the same symbol, THE PatternDetector SHALL return the cached pattern columns without recomputing any rolling-window logic.

2. WHEN `add_chart_patterns()` is called with a DataFrame that has more rows than the previous call for the same symbol, THE PatternDetector SHALL recompute pattern flags only for the new rows (rows beyond the previously cached index) using a lookback window of at most 1000 candles.

3. WHEN `add_chart_patterns()` is called for the first time for a symbol (no cache exists), THE PatternDetector SHALL compute pattern flags for the full DataFrame using the existing rolling-window logic.

4. THE PatternDetector SHALL complete pattern computation for a single symbol within 2 seconds when the input DataFrame contains 2000 or fewer rows.

5. WHEN the PatternDetector cache is populated, THE PatternDetector SHALL produce identical pattern flag values for all previously computed rows as it would without caching (no regression in pattern accuracy).

6. THE PatternDetector SHALL maintain a separate cache entry per symbol so that caching for one symbol does not affect computation for another symbol.

7. WHEN `add_chart_patterns()` completes, THE PatternDetector SHALL log the number of new rows processed and whether the result was served from cache.

#### Unchanged Behavior (Regression Prevention)

8. THE PatternDetector SHALL continue to produce all 14 pattern flag columns (`DoubleTop_Flag`, `DoubleBottom_Flag`, `TripleTop_Flag`, `TripleBottom_Flag`, `HS_Flag`, `InvHS_Flag`, `AscTriangle_Flag`, `DescTriangle_Flag`, `SymTriangle_Flag`, `RisingWedge_Flag`, `FallingWedge_Flag`, `BullFlag_Flag`, `BearFlag_Flag`, `Volatility_Compress`) and the `pattern_bias_score` composite column in every returned DataFrame.

9. THE PatternDetector SHALL continue to use ATR-normalized thresholds for all swing-point comparisons so that pattern sensitivity remains consistent across symbols with different price scales.

---

### Requirement 2: COUNTER_TREND_H1 Block Softening

**User Story:** As a trading bot operator, I want the H1 counter-trend filter to allow high-confidence signals through even when the H1 trend opposes the trade direction, so that the bot does not miss strong pullback entries during H1 downtrends or uptrends.

#### Acceptance Criteria

1. WHEN the EnsembleEngine evaluates a BUY signal and `h1_trend == -1` AND `xgb_prob >= 0.75`, THE EnsembleEngine SHALL NOT block the signal and SHALL proceed to the threshold check step with the base score intact.

2. WHEN the EnsembleEngine evaluates a SELL signal and `h1_trend == 1` AND `xgb_prob >= 0.75`, THE EnsembleEngine SHALL NOT block the signal and SHALL proceed to the threshold check step with the base score intact.

3. WHEN the EnsembleEngine evaluates a BUY signal and `h1_trend == -1` AND `xgb_prob < 0.75`, THE EnsembleEngine SHALL apply a configurable score penalty (default: `−0.04`) to the base score instead of issuing a hard HOLD.

4. WHEN the EnsembleEngine evaluates a SELL signal and `h1_trend == 1` AND `xgb_prob < 0.75`, THE EnsembleEngine SHALL apply a configurable score penalty (default: `−0.04`) to the base score instead of issuing a hard HOLD.

5. WHEN `h1_trend == 0` (neutral), THE EnsembleEngine SHALL apply no counter-trend penalty and SHALL proceed normally.

6. THE EnsembleEngine SHALL log the XGB confidence value, the H1 trend direction, and whether the counter-trend override was applied or the penalty was applied, for every evaluation where `h1_trend != 0`.

7. THE Config module SHALL expose a `COUNTER_TREND_XGB_OVERRIDE_THRESHOLD` parameter (default: `0.75`) and a `COUNTER_TREND_PENALTY` parameter (default: `−0.04`) so that the softening behavior can be tuned without code changes.

#### Unchanged Behavior (Regression Prevention)

8. WHEN `h1_trend == 0`, THE EnsembleEngine SHALL continue to apply no H1-related penalty or block, preserving existing neutral-trend behavior.

9. THE EnsembleEngine SHALL continue to apply all other filters (ATR filter, RF noise gate, conflict detection, regime conflict, weak zone, score floor) independently of the counter-trend logic change.

10. WHEN `xgb_prob < 0.75` and `h1_trend != 0`, the applied penalty SHALL reduce the final score but SHALL NOT force a hard HOLD by itself — the signal may still pass if the penalized score exceeds the buy/sell threshold.

---

### Requirement 3: UNKNOWN Session Trading Coverage

**User Story:** As a trading bot operator, I want the bot to continue evaluating and executing trades during off-hours (UTC 22:00–23:59) instead of blocking all activity due to an UNKNOWN session label, so that valid signals during the Pacific/late-NY overlap are not missed.

#### Acceptance Criteria

1. WHEN `TradeManager.get_active_session()` is called and the UTC hour is 22 or 23, THE TradeManager SHALL return `"Pacific"` instead of `"UNKNOWN"`. At exactly UTC hour 22, the Pacific session takes priority over any overlap with the New York session end boundary.

2. WHEN `TradeManager.is_in_trading_session()` is called and the current session is `"Pacific"`, THE TradeManager SHALL return `(True, reason_string)` only if a new `TRADE_SESSION_PACIFIC` config flag is set to `True`.

3. THE Config module SHALL expose a `SESSION_PACIFIC` tuple (default: `(22, 24)`) defining the UTC hour range for the Pacific session.

4. THE Config module SHALL expose a `TRADE_SESSION_PACIFIC` boolean flag (default: `True`) that enables or disables trading during the Pacific session.

5. WHEN the EnsembleEngine receives `session="Pacific"`, THE EnsembleEngine `_compute_session_bonus()` function SHALL return a session bonus of `0.0` (neutral — no bonus, no penalty) regardless of trend strength.

6. WHEN `TradeManager.is_in_trading_session()` is called and the UTC hour falls outside all defined session windows (London, New York, Asia, Pacific), THE TradeManager SHALL return `(False, "Outside all sessions (UTC Hour: {hour})")`.

7. WHEN the session is `"Pacific"` and `TRADE_SESSION_PACIFIC` is `True`, THE TradeManager SHALL apply a position size modifier of `0.5` (50% of normal size) to all trades executed during this session to reflect reduced liquidity.

8. IF `TRADE_SESSION_PACIFIC` is `False`, THEN THE TradeManager SHALL return `(False, "In Pacific session, but trading is disabled for this session")`, no trades SHALL be executed, and the position size modifier SHALL NOT be applied.

#### Unchanged Behavior (Regression Prevention)

9. THE session detection logic for `"London"` (UTC 07–16), `"New York"` (UTC 13–22), and `"Asia"` (UTC 00–09) SHALL remain unchanged, including their priority order and their respective `TRADE_SESSION_*` toggle flags.

10. THE EnsembleEngine session bonus values for `"London"`, `"New York"`, and `"Asia"` sessions SHALL remain unchanged.

---

### Requirement 4: BUY Threshold Recalibration

**User Story:** As a trading bot operator, I want the BUY threshold to be lowered to account for the conservative probability compression introduced by isotonic calibration, so that valid high-probability signals (such as USDJPY at 0.5994) are not rejected by a threshold that was calibrated for uncalibrated model outputs.

#### Acceptance Criteria

1. THE EnsembleEngine SHALL use a BUY threshold of `0.58` when `trend_strength <= 0.25` (ranging market), replacing the current value of `0.64`.

2. THE EnsembleEngine SHALL use a BUY threshold of `0.57` when `0.25 < trend_strength <= 0.35` (transitioning market), replacing the current value of `0.63`.

3. THE EnsembleEngine SHALL use a BUY threshold of `0.56` when `trend_strength > 0.35` (trending market), replacing the current value of `0.62`.

4. THE EnsembleEngine SHALL use a SELL threshold of `0.42` when `trend_strength <= 0.25`, replacing the current value of `0.36`.

5. THE EnsembleEngine SHALL use a SELL threshold of `0.43` when `0.25 < trend_strength <= 0.35`, replacing the current value of `0.37`.

6. THE EnsembleEngine SHALL use a SELL threshold of `0.44` when `trend_strength > 0.35`, replacing the current value of `0.38`.

7. THE Config module SHALL expose `PROB_THRESHOLD_BUY_RANGING`, `PROB_THRESHOLD_BUY_TRANSITIONING`, and `PROB_THRESHOLD_BUY_TRENDING` parameters so that thresholds can be adjusted without code changes.

8. THE Config module SHALL expose `PROB_THRESHOLD_SELL_RANGING`, `PROB_THRESHOLD_SELL_TRANSITIONING`, and `PROB_THRESHOLD_SELL_TRENDING` parameters symmetrically.

9. WHEN `diagnostic=True`, THE EnsembleEngine SHALL use fixed threshold variable values of exactly `buy_threshold = 0.62` and `sell_threshold = 0.38`, unchanged from the current implementation, so that diagnostic mode comparisons remain consistent.

10. WHEN a signal that previously scored between 0.58 and 0.64 is re-evaluated with the new thresholds, THE EnsembleEngine SHALL emit a BUY direction if the final score exceeds the new threshold and all other filters pass.

#### Unchanged Behavior (Regression Prevention)

11. THE EnsembleEngine SHALL continue to apply all pre-threshold filters (ATR filter, RF noise gate, conflict detection, regime conflict, weak zone, score floor) before the threshold check, so that lowering the threshold does not bypass any safety filter.

12. THE EnsembleEngine SHALL continue to log `buy_threshold` and `sell_threshold` values in the CSV decision log (`ensemble_decisions.csv`) so that threshold changes are fully auditable.

13. THE EnsembleEngine SHALL continue to apply the disagreement penalty before the threshold check, ensuring that low-confidence ensemble agreements do not benefit from the lower threshold.

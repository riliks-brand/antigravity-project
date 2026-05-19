# Bugfix Requirements Document

## Introduction

The trading bot fetches 99,000 candles per symbol per cycle instead of the configured `Config.DATA_POINTS = 2000`. This is caused by a hardcoded override in `data_loader.py`'s `fetch_mt5_ohlc()` function that forces a minimum of 99,000 candles regardless of the configured value. As a result, the PatternDetector takes 45+ seconds per symbol, the feature engineering pipeline processes ~93,000 unnecessary rows, and the full portfolio evaluation cycle takes 3–5 minutes instead of seconds — causing the bot to miss M5 candle closes and valid trade signals.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN `fetch_mt5_ohlc()` is called with `count=None` THEN the system requests `max(Config.DATA_POINTS, 99000)` candles, always resolving to 99,000 regardless of `Config.DATA_POINTS = 2000`

1.2 WHEN `fetch_mt5_ohlc()` is called with an explicit `count` argument (e.g. `Config.DATA_POINTS`) THEN the system overrides it with `max(count, 99000)`, discarding the caller-supplied value and fetching 99,000 candles

1.3 WHEN `fetch_mtf_data()` calls `fetch_mt5_ohlc()` for M5, M15, and H1 timeframes with `Config.DATA_POINTS` THEN the system fetches 99,000 candles per timeframe instead of 2,000, causing the PatternDetector to process ~99,000 rows and take 45+ seconds per symbol

1.4 WHEN the portfolio evaluation cycle runs across 6 symbols THEN the system takes 3–5 minutes to complete instead of seconds, causing the bot to miss M5 candle closes and valid trade signals

### Expected Behavior (Correct)

2.1 WHEN `fetch_mt5_ohlc()` is called with `count=None` THEN the system SHALL use `Config.DATA_POINTS` (2,000) as the default candle count without any minimum override

2.2 WHEN `fetch_mt5_ohlc()` is called with an explicit `count` argument THEN the system SHALL use the caller-supplied value without overriding it with a hardcoded minimum

2.3 WHEN `fetch_mtf_data()` calls `fetch_mt5_ohlc()` for M5, M15, and H1 timeframes with `Config.DATA_POINTS` THEN the system SHALL fetch 2,000 candles per timeframe, allowing the PatternDetector to complete in under 5 seconds per symbol

2.4 WHEN the portfolio evaluation cycle runs across 6 symbols THEN the system SHALL complete in under 30 seconds, allowing the bot to detect and execute valid trade signals at M5 candle closes

### Unchanged Behavior (Regression Prevention)

3.1 WHEN `fetch_mt5_ohlc()` is called with a `count` value greater than `Config.DATA_POINTS` (e.g. for historical backtesting) THEN the system SHALL CONTINUE TO fetch the requested number of candles without truncation

3.2 WHEN `fetch_mt5_ohlc()` is called and MT5 returns fewer candles than requested (e.g. symbol has limited history) THEN the system SHALL CONTINUE TO return whatever data MT5 provides without error

3.3 WHEN `fetch_mt5_ohlc()` is called for a symbol that is unavailable or has a closed market THEN the system SHALL CONTINUE TO return `None` and log the appropriate error

3.4 WHEN the PatternDetector cache is populated from a prior cycle THEN the system SHALL CONTINUE TO use the cached computation for subsequent calls (incremental updates remain unaffected by this fix)

3.5 WHEN `Config.DATA_POINTS` is changed to a value other than 2,000 THEN the system SHALL CONTINUE TO respect the configured value as the default candle count

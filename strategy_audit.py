"""
Strategy Audit — Controlled Experiment: Mode A vs Mode B
=========================================================
Replays ensemble_decisions.csv through two filter configurations
and compares outcomes using direction-aware PnL + drawdown.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
import sys
from datetime import datetime, timedelta

# =========================================
# MODE DEFINITIONS
# =========================================
MODE_A = {
    "name": "Mode A (Current)",
    "score_floor": 0.015,
    "weak_zone_normal": 0.04,
    "weak_zone_asia": 0.05,
    "base_threshold": 0.58,
}

MODE_B = {
    "name": "Mode B (Relaxed)",
    "score_floor": 0.012,
    "weak_zone_normal": 0.035,
    "weak_zone_asia": 0.045,
    "base_threshold": 0.56,
}

ATR_QUALITY_THRESHOLD = 0.0002
FORWARD_CANDLES = 3  # ~15 min horizon on M5


# =========================================
# PRICE DATA FETCHER
# =========================================
def fetch_price_data():
    """Fetch EURUSD M5 data from yfinance for outcome approximation."""
    print("[Audit] Fetching EURUSD price data from yfinance...")
    ticker = "EURUSD=X"
    end = datetime.utcnow()
    start = end - timedelta(days=7)
    df = yf.download(ticker, start=start, end=end, interval="5m", progress=False)
    if df is None or df.empty:
        print("[Audit] ⚠️ yfinance returned no data. Trying 1m interval...")
        df = yf.download(ticker, start=start, end=end, interval="1m", progress=False)
    if df is not None and not df.empty:
        # Handle MultiIndex columns from newer yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        print(f"[Audit] ✅ Got {len(df)} price candles from {df.index[0]} to {df.index[-1]}")
    else:
        print("[Audit] ❌ No price data available.")
    return df


def match_price_outcome(decision_time, side, price_df, n_forward=FORWARD_CANDLES):
    """Direction-aware PnL + max drawdown for a single decision."""
    if price_df is None or price_df.empty:
        return None, None

    dt = pd.Timestamp(decision_time).tz_localize(None) if pd.Timestamp(decision_time).tz else pd.Timestamp(decision_time)
    
    # Find nearest candle
    idx = price_df.index.searchsorted(dt)
    if idx >= len(price_df) - n_forward:
        return None, None

    entry_close = price_df['Close'].iloc[idx]
    future_slice = price_df.iloc[idx+1 : idx+1+n_forward]
    if future_slice.empty:
        return None, None

    exit_close = future_slice['Close'].iloc[-1]

    if side == "BUY":
        pnl = exit_close - entry_close
        max_dd = entry_close - future_slice['Low'].min()
    else:  # SELL
        pnl = entry_close - exit_close
        max_dd = future_slice['High'].max() - entry_close

    return float(pnl), float(max_dd)


# =========================================
# SIMULATION ENGINE
# =========================================
def simulate_mode(df, mode, price_df):
    """Re-evaluate each row under a given mode's filter rules."""
    results = []

    for _, row in df.iterrows():
        distance = row['distance_from_neutral']
        session = row['session']
        trend_strength = row['trend_strength']
        original_stage = row['stage_reached']
        side = row['side']
        base_score = row['raw_score']  # pre-adjustment score
        lstm_prob = row['lstm_prob']
        rf_prob = row['rf_prob']
        disagreement = row['disagreement']
        atr = row['atr']
        confidence = row['confidence_level']

        # Re-derive base_score from weighted_avg and penalty
        weighted_avg = row['weighted_avg']
        penalty = row['penalty']
        if weighted_avg > 0.5:
            recomputed_base = weighted_avg - penalty
        else:
            recomputed_base = weighted_avg + penalty
        recomputed_base = np.clip(recomputed_base, 0.0, 1.0)
        recomputed_distance = abs(recomputed_base - 0.5)

        # --- Filter Pipeline (mirrors ensemble_engine.py) ---

        # ATR Filter (unchanged between modes)
        if original_stage == "ATR_FILTER":
            results.append(_make_result(row, "ATR_FILTER", "LOW_ATR", None, mode, None, None))
            continue

        # Model Conflict (unchanged between modes)
        if row.get('conflict', False) == True or str(row.get('conflict', '')).lower() == 'true':
            results.append(_make_result(row, "CONFLICT", "CONFLICT", None, mode, None, None))
            continue

        # Score Floor
        if recomputed_distance < mode['score_floor']:
            results.append(_make_result(row, "SCORE_FLOOR", "BELOW_THRESHOLD", None, mode, None, None))
            continue

        # Weak Zone
        wz = mode['weak_zone_asia'] if session == "Asia" else mode['weak_zone_normal']
        if recomputed_distance < wz:
            results.append(_make_result(row, "WEAK_ZONE", f"WEAK_ZONE", None, mode, None, None))
            continue

        # Regime Conflict (unchanged)
        if (session == "London" and trend_strength < 0.2) or \
           (session == "Asia" and trend_strength > 0.8):
            results.append(_make_result(row, "CONFLICT", "REGIME_CONFLICT", None, mode, None, None))
            continue

        # Threshold Check
        buy_threshold = mode['base_threshold'] + (1.0 - trend_strength) * 0.08
        sell_threshold = 1.0 - buy_threshold

        # Apply additive adjustments (use stored values)
        final_prob = float(np.clip(row['final_score'], 0.0, 1.0))

        direction = None
        if final_prob > buy_threshold:
            direction = "BUY"
        elif final_prob < sell_threshold:
            direction = "SELL"

        if direction:
            pnl, dd = match_price_outcome(row['timestamp'], direction, price_df)
            results.append(_make_result(row, "EXECUTION_READY", "VALID_SIGNAL", direction, mode, pnl, dd))
        else:
            results.append(_make_result(row, "THRESHOLD_CHECK", "BELOW_THRESHOLD", None, mode, None, None))

    return pd.DataFrame(results)


def _make_result(row, stage, reason, direction, mode, pnl, dd):
    conf_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    return {
        "timestamp": row['timestamp'],
        "session": row['session'],
        "mode": mode['name'],
        "stage": stage,
        "reason": reason,
        "direction": direction or "HOLD",
        "side": row['side'],
        "lstm_prob": row['lstm_prob'],
        "rf_prob": row['rf_prob'],
        "disagreement": row['disagreement'],
        "trend_strength": row['trend_strength'],
        "distance_from_neutral": row['distance_from_neutral'],
        "confidence_level": row['confidence_level'],
        "confidence_num": conf_map.get(str(row['confidence_level']), 0),
        "final_score": row['final_score'],
        "atr": row['atr'],
        "pnl": pnl,
        "max_drawdown": dd,
        "quality_score": conf_map.get(str(row['confidence_level']), 0) + row['trend_strength'] - row['disagreement'],
    }


# =========================================
# ANALYSIS & REPORTING
# =========================================
def analyze_mode(results_df, mode_name):
    """Compute comprehensive metrics for a single mode."""
    total = len(results_df)
    executed = results_df[results_df['stage'] == 'EXECUTION_READY']
    holds = results_df[results_df['stage'] != 'EXECUTION_READY']

    metrics = {
        "mode": mode_name,
        "total_signals": total,
        "hold_count": len(holds),
        "execution_ready": len(executed),
        "execution_rate": f"{len(executed)/total*100:.1f}%" if total > 0 else "0%",
    }

    # Stage breakdown
    stage_counts = results_df['stage'].value_counts().to_dict()
    metrics["stage_breakdown"] = stage_counts

    # Reason breakdown
    reason_counts = results_df['reason'].value_counts().to_dict()
    metrics["reason_breakdown"] = reason_counts

    # Quality metrics for executed trades
    if len(executed) > 0:
        metrics["avg_confidence"] = f"{executed['confidence_num'].mean():.2f}"
        metrics["avg_distance"] = f"{executed['distance_from_neutral'].mean():.4f}"
        metrics["avg_trend_strength"] = f"{executed['trend_strength'].mean():.4f}"
        metrics["avg_disagreement"] = f"{executed['disagreement'].mean():.4f}"
        metrics["avg_quality_score"] = f"{executed['quality_score'].mean():.4f}"

        # PnL metrics (only where we have price data)
        with_pnl = executed.dropna(subset=['pnl'])
        if len(with_pnl) > 0:
            metrics["trades_with_pnl"] = len(with_pnl)
            metrics["avg_pnl"] = f"{with_pnl['pnl'].mean():.6f}"
            metrics["total_pnl"] = f"{with_pnl['pnl'].sum():.6f}"
            wins = with_pnl[with_pnl['pnl'] > 0]
            metrics["win_count"] = len(wins)
            metrics["loss_count"] = len(with_pnl) - len(wins)
            metrics["win_rate"] = f"{len(wins)/len(with_pnl)*100:.1f}%"
            metrics["avg_drawdown"] = f"{with_pnl['max_drawdown'].mean():.6f}"
            metrics["max_drawdown"] = f"{with_pnl['max_drawdown'].max():.6f}"
        else:
            metrics["trades_with_pnl"] = 0
            metrics["note"] = "No price data matched for PnL calculation"

        # Session breakdown
        session_metrics = {}
        for sess in ["London", "New York", "Asia"]:
            sess_exec = executed[executed['session'] == sess]
            sess_all = results_df[results_df['session'] == sess]
            sess_data = {
                "total": len(sess_all),
                "executed": len(sess_exec),
                "rate": f"{len(sess_exec)/len(sess_all)*100:.1f}%" if len(sess_all) > 0 else "0%",
            }
            sess_pnl = sess_exec.dropna(subset=['pnl'])
            if len(sess_pnl) > 0:
                sess_wins = sess_pnl[sess_pnl['pnl'] > 0]
                sess_data["win_rate"] = f"{len(sess_wins)/len(sess_pnl)*100:.1f}%"
                sess_data["avg_pnl"] = f"{sess_pnl['pnl'].mean():.6f}"
                sess_data["avg_quality"] = f"{sess_exec['quality_score'].mean():.4f}"
            session_metrics[sess] = sess_data
        metrics["session_breakdown"] = session_metrics
    else:
        metrics["note"] = "No trades reached EXECUTION_READY"

    return metrics


def delta_analysis(results_a, results_b, price_df):
    """Deep analysis of trades approved in Mode B but rejected in Mode A."""
    merged = results_a[['timestamp', 'stage']].rename(columns={'stage': 'stage_a'}).merge(
        results_b[['timestamp', 'stage', 'direction', 'side', 'session', 'lstm_prob', 'rf_prob',
                    'disagreement', 'trend_strength', 'distance_from_neutral', 'confidence_level',
                    'confidence_num', 'quality_score', 'pnl', 'max_drawdown', 'reason']].rename(
            columns={'stage': 'stage_b'}),
        on='timestamp', how='inner'
    )

    delta = merged[(merged['stage_a'] != 'EXECUTION_READY') & (merged['stage_b'] == 'EXECUTION_READY')]

    if delta.empty:
        return {"count": 0, "note": "No delta trades found"}

    analysis = {
        "count": len(delta),
        "avg_confidence": f"{delta['confidence_num'].mean():.2f}",
        "avg_trend_strength": f"{delta['trend_strength'].mean():.4f}",
        "avg_disagreement": f"{delta['disagreement'].mean():.4f}",
        "avg_distance": f"{delta['distance_from_neutral'].mean():.4f}",
        "avg_quality_score": f"{delta['quality_score'].mean():.4f}",
        "session_distribution": delta['session'].value_counts().to_dict(),
        "previously_blocked_by": {},
    }

    # Original block reasons from Mode A
    block_reasons = results_a[results_a['timestamp'].isin(delta['timestamp'])]['reason'].value_counts().to_dict()
    analysis["original_block_reasons"] = block_reasons

    delta_pnl = delta.dropna(subset=['pnl'])
    if len(delta_pnl) > 0:
        wins = delta_pnl[delta_pnl['pnl'] > 0]
        analysis["trades_with_pnl"] = len(delta_pnl)
        analysis["win_rate"] = f"{len(wins)/len(delta_pnl)*100:.1f}%"
        analysis["avg_pnl"] = f"{delta_pnl['pnl'].mean():.6f}"
        analysis["avg_drawdown"] = f"{delta_pnl['max_drawdown'].mean():.6f}"
    else:
        analysis["pnl_note"] = "No price data matched for delta trades"

    # Individual delta trades detail
    detail_rows = []
    for _, r in delta.iterrows():
        detail_rows.append({
            "timestamp": str(r['timestamp']),
            "session": r['session'],
            "direction": r['direction'],
            "confidence": r['confidence_level'],
            "trend_str": f"{r['trend_strength']:.3f}",
            "disagree": f"{r['disagreement']:.4f}",
            "distance": f"{r['distance_from_neutral']:.4f}",
            "quality": f"{r['quality_score']:.4f}",
            "pnl": f"{r['pnl']:.6f}" if pd.notna(r['pnl']) else "N/A",
            "drawdown": f"{r['max_drawdown']:.6f}" if pd.notna(r['max_drawdown']) else "N/A",
        })
    analysis["detail"] = detail_rows

    return analysis


def distance_distribution(results_df, mode_name):
    """Histogram of distance_from_neutral for executed trades."""
    executed = results_df[results_df['stage'] == 'EXECUTION_READY']
    if executed.empty:
        return {"mode": mode_name, "note": "No executed trades"}

    dist = executed['distance_from_neutral']
    bins = [0, 0.02, 0.04, 0.06, 0.10, 0.15, 0.20, 0.30, 0.50]
    hist = pd.cut(dist, bins=bins, right=False).value_counts().sort_index()
    return {"mode": mode_name, "histogram": {str(k): int(v) for k, v in hist.items()}}


# =========================================
# MAIN EXECUTION
# =========================================
def main():
    print("=" * 65)
    print("  🧪 STRATEGY AUDIT — Mode A vs Mode B Controlled Experiment")
    print("=" * 65)

    # 1. Load CSV
    csv_path = "ensemble_decisions.csv"
    if not os.path.exists(csv_path):
        print(f"[Error] {csv_path} not found!")
        return

    df = pd.read_csv(csv_path)
    print(f"[Audit] Loaded {len(df)} decisions from {csv_path}")

    # Convert types
    for col in ['lstm_prob', 'rf_prob', 'weighted_avg', 'disagreement', 'penalty',
                 'raw_score', 'final_score', 'distance_from_neutral', 'trend_strength',
                 'atr', 'weak_zone_threshold_used', 'session_bonus', 'volatility_adjustment']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Data Quality Filter
    pre_filter = len(df)
    df = df[df['atr'] >= ATR_QUALITY_THRESHOLD]
    post_filter = len(df)
    print(f"[Audit] Data Quality Filter: {pre_filter} → {post_filter} rows (removed {pre_filter - post_filter} low-ATR)")

    # 3. Fetch price data for outcome approximation
    price_df = fetch_price_data()

    # 4. Run simulations
    print("\n[Audit] Running Mode A simulation...")
    results_a = simulate_mode(df, MODE_A, price_df)
    print(f"[Audit] Mode A: {len(results_a)} decisions processed")

    print("[Audit] Running Mode B simulation...")
    results_b = simulate_mode(df, MODE_B, price_df)
    print(f"[Audit] Mode B: {len(results_b)} decisions processed")

    # 5. Analyze
    metrics_a = analyze_mode(results_a, MODE_A['name'])
    metrics_b = analyze_mode(results_b, MODE_B['name'])

    # 6. Delta Analysis
    delta = delta_analysis(results_a, results_b, price_df)

    # 7. Distance Distribution
    dist_a = distance_distribution(results_a, MODE_A['name'])
    dist_b = distance_distribution(results_b, MODE_B['name'])

    # 8. Print Report
    print("\n" + "=" * 65)
    print("  📊 AUDIT REPORT — Mode A vs Mode B")
    print("=" * 65)

    for m in [metrics_a, metrics_b]:
        print(f"\n{'─' * 50}")
        print(f"  📌 {m['mode']}")
        print(f"{'─' * 50}")
        print(f"  Total Signals      : {m['total_signals']}")
        print(f"  HOLD Count         : {m['hold_count']}")
        print(f"  EXECUTION_READY    : {m['execution_ready']} ({m['execution_rate']})")
        if 'stage_breakdown' in m:
            print(f"  --- Stage Breakdown ---")
            for s, c in m['stage_breakdown'].items():
                print(f"    {s:22s}: {c}")
        if 'avg_confidence' in m:
            print(f"  Avg Confidence     : {m['avg_confidence']}")
            print(f"  Avg Distance       : {m['avg_distance']}")
            print(f"  Avg Trend Str.     : {m['avg_trend_strength']}")
            print(f"  Avg Disagreement   : {m['avg_disagreement']}")
            print(f"  Avg Quality Score  : {m['avg_quality_score']}")
        if 'win_rate' in m:
            print(f"  --- PnL Metrics ---")
            print(f"  Trades with PnL    : {m['trades_with_pnl']}")
            print(f"  Win Rate           : {m['win_rate']}")
            print(f"  Avg PnL            : {m['avg_pnl']}")
            print(f"  Total PnL          : {m['total_pnl']}")
            print(f"  Avg Drawdown       : {m['avg_drawdown']}")
            print(f"  Max Drawdown       : {m['max_drawdown']}")
        if 'session_breakdown' in m:
            print(f"  --- Session Breakdown ---")
            for sess, sd in m['session_breakdown'].items():
                wr = sd.get('win_rate', 'N/A')
                ap = sd.get('avg_pnl', 'N/A')
                aq = sd.get('avg_quality', 'N/A')
                print(f"    {sess:12s}: {sd['executed']}/{sd['total']} ({sd['rate']}) WR={wr} PnL={ap} Q={aq}")

    print(f"\n{'=' * 65}")
    print(f"  🔍 DELTA TRADE ANALYSIS (Mode B adds, Mode A blocks)")
    print(f"{'=' * 65}")
    print(f"  Delta Trades       : {delta['count']}")
    if delta['count'] > 0:
        print(f"  Avg Quality Score  : {delta.get('avg_quality_score', 'N/A')}")
        print(f"  Avg Confidence     : {delta.get('avg_confidence', 'N/A')}")
        print(f"  Avg Trend Strength : {delta.get('avg_trend_strength', 'N/A')}")
        print(f"  Avg Disagreement   : {delta.get('avg_disagreement', 'N/A')}")
        print(f"  Session Dist.      : {delta.get('session_distribution', {})}")
        print(f"  Blocked By (A)     : {delta.get('original_block_reasons', {})}")
        if 'win_rate' in delta:
            print(f"  Win Rate           : {delta['win_rate']}")
            print(f"  Avg PnL            : {delta['avg_pnl']}")
            print(f"  Avg Drawdown       : {delta['avg_drawdown']}")
        if 'detail' in delta:
            print(f"\n  --- Individual Delta Trades ---")
            for t in delta['detail'][:15]:  # Show top 15
                print(f"    {t['timestamp'][:19]} | {t['session']:8s} | {t['direction']:4s} | "
                      f"Conf={t['confidence']:6s} | TS={t['trend_str']} | Dis={t['disagree']} | "
                      f"Dist={t['distance']} | Q={t['quality']} | PnL={t['pnl']} | DD={t['drawdown']}")

    print(f"\n{'=' * 65}")
    print(f"  📊 DISTANCE DISTRIBUTION (Executed Trades)")
    print(f"{'=' * 65}")
    for d in [dist_a, dist_b]:
        print(f"\n  {d['mode']}:")
        if 'histogram' in d:
            for bucket, count in d['histogram'].items():
                bar = "█" * count
                print(f"    {bucket:20s}: {count:3d} {bar}")

    # 9. Save full report as JSON
    report = {
        "mode_a": metrics_a,
        "mode_b": metrics_b,
        "delta_analysis": delta,
        "distance_distribution_a": dist_a,
        "distance_distribution_b": dist_b,
    }
    report_path = "audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[Audit] Full report saved to {report_path}")

    print(f"\n{'=' * 65}")
    print(f"  ✅ AUDIT COMPLETE")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

"""
Strategy Audit — Mode A vs Mode C
"""
import pandas as pd
import numpy as np
import yfinance as yf
import json, os
from datetime import datetime, timedelta

MODE_A = {"name": "Mode A (Current)", "score_floor": 0.015, "weak_zone_normal": 0.04,
           "weak_zone_asia": 0.05, "base_threshold": 0.58, "regime": "strict", "conflict": "binary"}
MODE_C = {"name": "Mode C (Targeted)", "score_floor": 0.015, "weak_zone_normal": 0.04,
           "weak_zone_asia": 0.05, "base_threshold": 0.58, "regime": "adaptive", "conflict": "graduated"}

ATR_QUALITY_THRESHOLD = 0.0002
FORWARD_CANDLES = 3

def fetch_price_data():
    print("[Audit] Fetching EURUSD price data...")
    ticker = "EURUSD=X"
    end = datetime.utcnow()
    start = end - timedelta(days=7)
    df = yf.download(ticker, start=start, end=end, interval="5m", progress=False)
    if df is None or df.empty:
        df = yf.download(ticker, start=start, end=end, interval="1m", progress=False)
    if df is not None and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        print(f"[Audit] Got {len(df)} price candles")
    return df

def match_price_outcome(decision_time, side, price_df):
    if price_df is None or price_df.empty: return None, None
    dt = pd.Timestamp(decision_time).tz_localize(None) if pd.Timestamp(decision_time).tz else pd.Timestamp(decision_time)
    idx = price_df.index.searchsorted(dt)
    if idx >= len(price_df) - FORWARD_CANDLES: return None, None
    entry = price_df['Close'].iloc[idx]
    future = price_df.iloc[idx+1:idx+1+FORWARD_CANDLES]
    if future.empty: return None, None
    exit_p = future['Close'].iloc[-1]
    if side == "BUY":
        return float(exit_p - entry), float(entry - future['Low'].min())
    else:
        return float(entry - exit_p), float(future['High'].max() - entry)

def simulate_mode(df, mode, price_df):
    results = []
    for _, row in df.iterrows():
        session = row['session']
        ts = row['trend_strength']
        weighted_avg = row['weighted_avg']
        penalty = row['penalty']
        lstm_prob = row['lstm_prob']
        rf_prob = row['rf_prob']
        disagreement = row['disagreement']
        atr = row['atr']
        original_stage = row['stage_reached']

        if weighted_avg > 0.5: base = weighted_avg - penalty
        else: base = weighted_avg + penalty
        base = np.clip(base, 0.0, 1.0)
        dist = abs(base - 0.5)

        # ATR Filter
        if original_stage == "ATR_FILTER":
            results.append(_r(row, "ATR_FILTER", "LOW_ATR", None, mode, None, None)); continue

        # Model Conflict
        if mode['conflict'] == 'binary':
            if disagreement >= 0.50:
                results.append(_r(row, "CONFLICT", "MODEL_CONFLICT", None, mode, None, None)); continue
        else:  # graduated
            if disagreement >= 0.60:
                results.append(_r(row, "CONFLICT", "MODEL_CONFLICT", None, mode, None, None)); continue
            if disagreement >= 0.45:
                same_dir = (lstm_prob > 0.5 and rf_prob > 0.5) or (lstm_prob < 0.5 and rf_prob < 0.5)
                if not same_dir:
                    results.append(_r(row, "CONFLICT", "MODEL_CONFLICT", None, mode, None, None)); continue

        # Score Floor
        if dist < mode['score_floor']:
            results.append(_r(row, "SCORE_FLOOR", "BELOW_THRESHOLD", None, mode, None, None)); continue

        # Weak Zone
        wz = mode['weak_zone_asia'] if session == "Asia" else mode['weak_zone_normal']
        if dist < wz:
            results.append(_r(row, "WEAK_ZONE", "WEAK_ZONE", None, mode, None, None)); continue

        # Regime Conflict
        if mode['regime'] == 'strict':
            if (session == "London" and ts < 0.2) or (session == "Asia" and ts > 0.8):
                results.append(_r(row, "CONFLICT", "REGIME_CONFLICT", None, mode, None, None)); continue
        else:  # adaptive
            conflict = False
            if session == "London" and ts < 0.15: conflict = True
            elif session == "Asia" and ts > 0.8: conflict = True
            if conflict:
                atr_norm = min(1.0, atr / 0.001) if atr > 0 else 1.0
                dyn_dist = 0.12 + (atr_norm * 0.05)
                conf = "HIGH" if dist > 0.15 else ("MEDIUM" if dist > 0.08 else "LOW")
                conf_score = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(conf, 1)
                if not (dist > dyn_dist and conf_score >= 2):
                    results.append(_r(row, "CONFLICT", "REGIME_CONFLICT", None, mode, None, None)); continue
                # Override: apply penalty
                regime_pen = max(0.0, (0.15 - ts) * 0.1) if session == "London" else 0.02
                if base > 0.5: base -= regime_pen
                else: base += regime_pen
                base = np.clip(base, 0.0, 1.0)

        # Threshold
        if mode['regime'] == 'adaptive' and ts >= 0.7:
            buy_th = 0.56 + (1.0 - ts) * 0.06
        else:
            buy_th = mode['base_threshold'] + (1.0 - ts) * 0.08
        sell_th = 1.0 - buy_th

        final = float(np.clip(row['final_score'], 0.0, 1.0))
        direction = None
        if final > buy_th: direction = "BUY"
        elif final < sell_th: direction = "SELL"

        if direction:
            pnl, dd = match_price_outcome(row['timestamp'], direction, price_df)
            results.append(_r(row, "EXECUTION_READY", "VALID_SIGNAL", direction, mode, pnl, dd))
        else:
            results.append(_r(row, "THRESHOLD_CHECK", "BELOW_THRESHOLD", None, mode, None, None))
    return pd.DataFrame(results)

def _r(row, stage, reason, direction, mode, pnl, dd):
    cm = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    return {"timestamp": row['timestamp'], "session": row['session'], "mode": mode['name'],
            "stage": stage, "reason": reason, "direction": direction or "HOLD", "side": row['side'],
            "lstm_prob": row['lstm_prob'], "rf_prob": row['rf_prob'], "disagreement": row['disagreement'],
            "trend_strength": row['trend_strength'], "distance_from_neutral": row['distance_from_neutral'],
            "confidence_level": row['confidence_level'], "confidence_num": cm.get(str(row['confidence_level']), 0),
            "final_score": row['final_score'], "atr": row['atr'], "pnl": pnl, "max_drawdown": dd,
            "quality_score": cm.get(str(row['confidence_level']), 0) + row['trend_strength'] - row['disagreement']}

def analyze(res, name):
    t = len(res); ex = res[res['stage']=='EXECUTION_READY']; h = t - len(ex)
    m = {"mode": name, "total": t, "holds": h, "executed": len(ex),
         "rate": f"{len(ex)/t*100:.1f}%" if t else "0%",
         "stages": res['stage'].value_counts().to_dict()}
    if len(ex):
        m["avg_conf"] = f"{ex['confidence_num'].mean():.2f}"
        m["avg_dist"] = f"{ex['distance_from_neutral'].mean():.4f}"
        m["avg_ts"] = f"{ex['trend_strength'].mean():.4f}"
        m["avg_dis"] = f"{ex['disagreement'].mean():.4f}"
        m["avg_q"] = f"{ex['quality_score'].mean():.4f}"
        wp = ex.dropna(subset=['pnl'])
        if len(wp):
            w = wp[wp['pnl']>0]
            m["pnl_n"] = len(wp); m["wr"] = f"{len(w)/len(wp)*100:.1f}%"
            m["avg_pnl"] = f"{wp['pnl'].mean():.6f}"; m["tot_pnl"] = f"{wp['pnl'].sum():.6f}"
            m["avg_dd"] = f"{wp['max_drawdown'].mean():.6f}"; m["max_dd"] = f"{wp['max_drawdown'].max():.6f}"
        sess = {}
        for s in ["London", "New York", "Asia"]:
            se = ex[ex['session']==s]; sa = res[res['session']==s]
            sd = {"exec": len(se), "total": len(sa), "rate": f"{len(se)/len(sa)*100:.1f}%" if len(sa) else "0%"}
            sp = se.dropna(subset=['pnl'])
            if len(sp):
                sw = sp[sp['pnl']>0]
                sd["wr"] = f"{len(sw)/len(sp)*100:.1f}%"; sd["pnl"] = f"{sp['pnl'].mean():.6f}"
                sd["q"] = f"{se['quality_score'].mean():.4f}"
            sess[s] = sd
        m["sessions"] = sess
    return m

def delta(ra, rc, pdf):
    mg = ra[['timestamp','stage']].rename(columns={'stage':'sa'}).merge(
        rc[['timestamp','stage','direction','side','session','lstm_prob','rf_prob',
            'disagreement','trend_strength','distance_from_neutral','confidence_level',
            'confidence_num','quality_score','pnl','max_drawdown','reason']].rename(columns={'stage':'sc'}),
        on='timestamp', how='inner')
    d = mg[(mg['sa']!='EXECUTION_READY')&(mg['sc']=='EXECUTION_READY')]
    if d.empty: return {"count": 0}
    a = {"count": len(d), "avg_q": f"{d['quality_score'].mean():.4f}",
         "avg_conf": f"{d['confidence_num'].mean():.2f}", "avg_ts": f"{d['trend_strength'].mean():.4f}",
         "avg_dis": f"{d['disagreement'].mean():.4f}", "avg_dist": f"{d['distance_from_neutral'].mean():.4f}",
         "sessions": d['session'].value_counts().to_dict(),
         "blocked_by": ra[ra['timestamp'].isin(d['timestamp'])]['reason'].value_counts().to_dict()}
    dp = d.dropna(subset=['pnl'])
    if len(dp):
        w = dp[dp['pnl']>0]
        a["wr"] = f"{len(w)/len(dp)*100:.1f}%"; a["avg_pnl"] = f"{dp['pnl'].mean():.6f}"
        a["avg_dd"] = f"{dp['max_drawdown'].mean():.6f}"
    a["detail"] = []
    for _, r in d.iterrows():
        a["detail"].append({"ts": str(r['timestamp'])[:19], "sess": r['session'], "dir": r['direction'],
            "conf": r['confidence_level'], "trend": f"{r['trend_strength']:.3f}",
            "dis": f"{r['disagreement']:.4f}", "dist": f"{r['distance_from_neutral']:.4f}",
            "q": f"{r['quality_score']:.4f}",
            "pnl": f"{r['pnl']:.6f}" if pd.notna(r['pnl']) else "N/A",
            "dd": f"{r['max_drawdown']:.6f}" if pd.notna(r['max_drawdown']) else "N/A"})
    return a

def dist_hist(res, name):
    ex = res[res['stage']=='EXECUTION_READY']
    if ex.empty: return {"mode": name}
    bins = [0, 0.02, 0.04, 0.06, 0.10, 0.15, 0.20, 0.30, 0.50]
    h = pd.cut(ex['distance_from_neutral'], bins=bins, right=False).value_counts().sort_index()
    return {"mode": name, "hist": {str(k): int(v) for k, v in h.items()}}

def pr(m):
    print(f"\n{'_'*50}")
    print(f"  {m['mode']}")
    print(f"{'_'*50}")
    print(f"  Total: {m['total']} | Executed: {m['executed']} ({m['rate']}) | Holds: {m['holds']}")
    for s, c in m.get('stages', {}).items(): print(f"    {s:22s}: {c}")
    for k in ['avg_conf','avg_dist','avg_ts','avg_dis','avg_q']:
        if k in m: print(f"  {k:20s}: {m[k]}")
    for k in ['wr','avg_pnl','tot_pnl','avg_dd','max_dd']:
        if k in m: print(f"  {k:20s}: {m[k]}")
    for s, sd in m.get('sessions', {}).items():
        wr = sd.get('wr','N/A'); p = sd.get('pnl','N/A'); q = sd.get('q','N/A')
        print(f"    {s:12s}: {sd['exec']}/{sd['total']} ({sd['rate']}) WR={wr} PnL={p} Q={q}")

def main():
    print("="*65)
    print("  STRATEGY AUDIT -- Mode A vs Mode C")
    print("="*65)
    df = pd.read_csv("ensemble_decisions.csv")
    print(f"[Audit] Loaded {len(df)} decisions")
    for c in ['lstm_prob','rf_prob','weighted_avg','disagreement','penalty','raw_score',
              'final_score','distance_from_neutral','trend_strength','atr']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    pre = len(df); df = df[df['atr'] >= ATR_QUALITY_THRESHOLD]; post = len(df)
    print(f"[Audit] Quality filter: {pre} -> {post} (removed {pre-post})")
    pdf = fetch_price_data()

    print("\n[Audit] Running Mode A..."); ra = simulate_mode(df, MODE_A, pdf)
    print(f"[Audit] Running Mode C..."); rc = simulate_mode(df, MODE_C, pdf)

    ma = analyze(ra, MODE_A['name']); mc = analyze(rc, MODE_C['name'])
    pr(ma); pr(mc)

    print(f"\n{'='*65}\n  DELTA ANALYSIS (Mode C adds, Mode A blocks)\n{'='*65}")
    d = delta(ra, rc, pdf)
    print(f"  Delta Trades: {d['count']}")
    if d['count'] > 0:
        for k in ['avg_q','avg_conf','avg_ts','avg_dis','avg_dist']:
            if k in d: print(f"  {k:20s}: {d[k]}")
        print(f"  Sessions: {d.get('sessions',{})}")
        print(f"  Blocked by (A): {d.get('blocked_by',{})}")
        for k in ['wr','avg_pnl','avg_dd']:
            if k in d: print(f"  {k:20s}: {d[k]}")
        if 'detail' in d:
            print(f"\n  --- Delta Trades Detail ---")
            for t in d['detail'][:20]:
                print(f"    {t['ts']} | {t['sess']:8s} | {t['dir']:4s} | C={t['conf']:6s} | "
                      f"TS={t['trend']} | Dis={t['dis']} | Dist={t['dist']} | Q={t['q']} | "
                      f"PnL={t['pnl']} | DD={t['dd']}")

    print(f"\n{'='*65}\n  DISTANCE DISTRIBUTION\n{'='*65}")
    for dh in [dist_hist(ra, MODE_A['name']), dist_hist(rc, MODE_C['name'])]:
        print(f"\n  {dh['mode']}:")
        for b, c in dh.get('hist', {}).items():
            print(f"    {b:20s}: {c:3d} {'#'*c}")

    rpt = {"mode_a": ma, "mode_c": mc, "delta": d}
    with open("audit_report_c.json", "w") as f: json.dump(rpt, f, indent=2, default=str)
    print(f"\n[Audit] Report saved to audit_report_c.json")
    print(f"\n{'='*65}\n  AUDIT COMPLETE\n{'='*65}")

if __name__ == "__main__":
    main()

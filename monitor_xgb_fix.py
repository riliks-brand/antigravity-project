"""
XGBoost Fix Performance Monitor - v6.1

This script monitors the effectiveness of the XGBoost BUY bias fix by analyzing:
1. Prediction distribution (BUY/SELL/NOISE zones)
2. Trading performance metrics (win rate, profit factor, drawdown)
3. Trade direction balance (BUY vs SELL)
4. Temporal analysis (performance over time)

Usage:
    python monitor_xgb_fix.py                    # Full analysis
    python monitor_xgb_fix.py --quick            # Quick summary only
    python monitor_xgb_fix.py --symbol EURUSD    # Specific symbol
    python monitor_xgb_fix.py --days 7           # Last 7 days only
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import argparse
from collections import defaultdict

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text:^70}{Colors.END}")
    print(f"{Colors.CYAN}{'='*70}{Colors.END}\n")

def print_section(text):
    """Print a section header"""
    print(f"\n{Colors.BLUE}{'─'*70}{Colors.END}")
    print(f"{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BLUE}{'─'*70}{Colors.END}")

def colorize_metric(value, threshold_good, threshold_bad, reverse=False):
    """Colorize a metric based on thresholds"""
    if reverse:
        if value <= threshold_good:
            return f"{Colors.GREEN}{value}{Colors.END}"
        elif value <= threshold_bad:
            return f"{Colors.YELLOW}{value}{Colors.END}"
        else:
            return f"{Colors.RED}{value}{Colors.END}"
    else:
        if value >= threshold_good:
            return f"{Colors.GREEN}{value}{Colors.END}"
        elif value >= threshold_bad:
            return f"{Colors.YELLOW}{value}{Colors.END}"
        else:
            return f"{Colors.RED}{value}{Colors.END}"

def load_ensemble_decisions(days=None):
    """Load ensemble decisions from CSV"""
    filepath = 'ensemble_decisions.csv'
    
    if not os.path.exists(filepath):
        print(f"{Colors.RED}❌ Error: {filepath} not found{Colors.END}")
        print(f"{Colors.YELLOW}Make sure the bot has been running and generating decisions.{Colors.END}")
        return None
    
    try:
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by days if specified
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            df = df[df['timestamp'] >= cutoff]
        
        if len(df) == 0:
            print(f"{Colors.YELLOW}⚠️  No data found in the specified time range{Colors.END}")
            return None
            
        return df
    except Exception as e:
        print(f"{Colors.RED}❌ Error loading {filepath}: {e}{Colors.END}")
        return None

def load_trading_history(days=None):
    """Load trading history from CSV"""
    filepath = 'trading_history.csv'
    
    if not os.path.exists(filepath):
        print(f"{Colors.YELLOW}⚠️  Warning: {filepath} not found{Colors.END}")
        print(f"{Colors.YELLOW}Trading performance metrics will not be available.{Colors.END}")
        return None
    
    try:
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by days if specified
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            df = df[df['timestamp'] >= cutoff]
        
        return df
    except Exception as e:
        print(f"{Colors.YELLOW}⚠️  Warning: Error loading {filepath}: {e}{Colors.END}")
        return None

def analyze_prediction_distribution(df, symbol=None):
    """Analyze XGBoost prediction distribution"""
    print_section("📊 XGBoost Prediction Distribution Analysis")
    
    if symbol:
        # Filter by symbol if specified (assuming there's a symbol column)
        if 'symbol' in df.columns:
            df = df[df['symbol'] == symbol]
            print(f"Analyzing symbol: {Colors.BOLD}{symbol}{Colors.END}\n")
    
    # Calculate distribution
    xgb_probs = df['xgb_prob'].dropna()
    
    if len(xgb_probs) == 0:
        print(f"{Colors.RED}❌ No XGBoost probability data found{Colors.END}")
        return
    
    pct_buy = (xgb_probs > 0.6).mean() * 100
    pct_sell = (xgb_probs < 0.4).mean() * 100
    pct_noise = ((xgb_probs >= 0.4) & (xgb_probs <= 0.6)).mean() * 100
    
    # Expected ranges (after fix)
    expected_buy = (30, 40)
    expected_sell = (30, 40)
    expected_noise = (20, 40)
    
    print(f"Total decisions analyzed: {Colors.BOLD}{len(xgb_probs):,}{Colors.END}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
    
    print("Distribution:")
    
    # BUY zone
    buy_status = "✅" if expected_buy[0] <= pct_buy <= expected_buy[1] else "⚠️" if pct_buy < 50 else "❌"
    buy_color = Colors.GREEN if expected_buy[0] <= pct_buy <= expected_buy[1] else Colors.YELLOW if pct_buy < 50 else Colors.RED
    print(f"  {buy_status} BUY (>0.6):      {buy_color}{pct_buy:5.1f}%{Colors.END}  (expected: {expected_buy[0]}-{expected_buy[1]}%)")
    
    # SELL zone
    sell_status = "✅" if expected_sell[0] <= pct_sell <= expected_sell[1] else "⚠️" if pct_sell > 10 else "❌"
    sell_color = Colors.GREEN if expected_sell[0] <= pct_sell <= expected_sell[1] else Colors.YELLOW if pct_sell > 10 else Colors.RED
    print(f"  {sell_status} SELL (<0.4):     {sell_color}{pct_sell:5.1f}%{Colors.END}  (expected: {expected_sell[0]}-{expected_sell[1]}%)")
    
    # NOISE zone
    noise_status = "✅" if expected_noise[0] <= pct_noise <= expected_noise[1] else "⚠️" if pct_noise > 15 else "❌"
    noise_color = Colors.GREEN if expected_noise[0] <= pct_noise <= expected_noise[1] else Colors.YELLOW if pct_noise > 15 else Colors.RED
    print(f"  {noise_status} NOISE (0.4-0.6): {noise_color}{pct_noise:5.1f}%{Colors.END}  (expected: {expected_noise[0]}-{expected_noise[1]}%)")
    
    # Overall assessment
    print(f"\n{Colors.BOLD}Assessment:{Colors.END}")
    if (expected_buy[0] <= pct_buy <= expected_buy[1] and 
        expected_sell[0] <= pct_sell <= expected_sell[1] and 
        expected_noise[0] <= pct_noise <= expected_noise[1]):
        print(f"  {Colors.GREEN}✅ Distribution is BALANCED - Fix is working correctly!{Colors.END}")
    elif pct_buy > 70:
        print(f"  {Colors.RED}❌ Severe BUY bias detected - Fix may not be working{Colors.END}")
        print(f"  {Colors.YELLOW}   → Check if models were retrained with v6.1 fix{Colors.END}")
    elif pct_buy > 50:
        print(f"  {Colors.YELLOW}⚠️  Moderate BUY bias - Improvement but not optimal{Colors.END}")
        print(f"  {Colors.YELLOW}   → May need more time to stabilize{Colors.END}")
    else:
        print(f"  {Colors.GREEN}✅ Distribution is improving{Colors.END}")
    
    # Histogram
    print(f"\n{Colors.BOLD}Probability Distribution:{Colors.END}")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(xgb_probs, bins=bins)
    max_count = hist.max()
    
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        count = hist[i]
        pct = count / len(xgb_probs) * 100
        bar_length = int(count / max_count * 40) if max_count > 0 else 0
        bar = '█' * bar_length
        print(f"  {low:.1f}-{high:.1f}: {bar:40s} {count:6,} ({pct:5.1f}%)")

def analyze_ensemble_decisions(df):
    """Analyze ensemble decision outcomes"""
    print_section("🎯 Ensemble Decision Analysis")
    
    # Direction distribution
    directions = df['direction'].value_counts()
    total = len(df)
    
    print(f"Total decisions: {Colors.BOLD}{total:,}{Colors.END}\n")
    
    print("Decision breakdown:")
    for direction, count in directions.items():
        pct = count / total * 100
        if direction == 'HOLD' or direction is None or pd.isna(direction):
            color = Colors.CYAN
            label = "HOLD"
        elif direction == 'BUY':
            color = Colors.GREEN
            label = "BUY"
        elif direction == 'SELL':
            color = Colors.RED
            label = "SELL"
        else:
            color = Colors.YELLOW
            label = str(direction)
        
        print(f"  {color}{label:6s}{Colors.END}: {count:6,} ({pct:5.1f}%)")
    
    # Skip reasons analysis
    if 'skip_reason' in df.columns:
        skip_reasons = df[df['direction'].isna() | (df['direction'] == 'HOLD')]['skip_reason'].value_counts()
        if len(skip_reasons) > 0:
            print(f"\n{Colors.BOLD}Top skip reasons:{Colors.END}")
            for reason, count in skip_reasons.head(5).items():
                if pd.notna(reason):
                    print(f"  • {reason}: {count:,}")

def analyze_trading_performance(df):
    """Analyze trading performance metrics"""
    print_section("💰 Trading Performance Analysis")
    
    if df is None or len(df) == 0:
        print(f"{Colors.YELLOW}No trading history data available{Colors.END}")
        return
    
    # Filter closed trades only
    closed = df[df['status'] == 'CLOSED'].copy() if 'status' in df.columns else df.copy()
    
    if len(closed) == 0:
        print(f"{Colors.YELLOW}No closed trades found{Colors.END}")
        return
    
    # Calculate metrics
    total_trades = len(closed)
    wins = closed[closed['profit'] > 0]
    losses = closed[closed['profit'] <= 0]
    
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    total_profit = wins['profit'].sum() if len(wins) > 0 else 0
    total_loss = abs(losses['profit'].sum()) if len(losses) > 0 else 0
    net_profit = total_profit - total_loss
    profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
    
    # Direction balance
    buy_trades = closed[closed['direction'] == 'BUY']
    sell_trades = closed[closed['direction'] == 'SELL']
    buy_pct = len(buy_trades) / total_trades * 100 if total_trades > 0 else 0
    sell_pct = len(sell_trades) / total_trades * 100 if total_trades > 0 else 0
    
    print(f"Total closed trades: {Colors.BOLD}{total_trades:,}{Colors.END}")
    print(f"Time range: {closed['timestamp'].min()} to {closed['timestamp'].max()}\n")
    
    # Win rate
    print(f"{Colors.BOLD}Win Rate:{Colors.END}")
    win_rate_str = f"{win_rate:.1f}%"
    win_rate_colored = colorize_metric(win_rate, 50, 40, reverse=False)
    print(f"  {win_rate_colored} ({win_count} wins / {loss_count} losses)")
    
    if win_rate >= 50:
        print(f"  {Colors.GREEN}✅ Excellent win rate!{Colors.END}")
    elif win_rate >= 40:
        print(f"  {Colors.YELLOW}⚠️  Acceptable win rate{Colors.END}")
    else:
        print(f"  {Colors.RED}❌ Low win rate - needs improvement{Colors.END}")
    
    # Profit metrics
    print(f"\n{Colors.BOLD}Profit Metrics:{Colors.END}")
    print(f"  Total Profit:  {Colors.GREEN}${total_profit:,.2f}{Colors.END}")
    print(f"  Total Loss:    {Colors.RED}${total_loss:,.2f}{Colors.END}")
    net_color = Colors.GREEN if net_profit > 0 else Colors.RED
    print(f"  Net Profit:    {net_color}${net_profit:,.2f}{Colors.END}")
    
    pf_colored = colorize_metric(profit_factor, 1.5, 1.0, reverse=False)
    print(f"  Profit Factor: {pf_colored}")
    
    if profit_factor >= 1.5:
        print(f"  {Colors.GREEN}✅ Strong profit factor!{Colors.END}")
    elif profit_factor >= 1.0:
        print(f"  {Colors.YELLOW}⚠️  Profitable but could be better{Colors.END}")
    else:
        print(f"  {Colors.RED}❌ Unprofitable - losing money{Colors.END}")
    
    # Direction balance
    print(f"\n{Colors.BOLD}Trade Direction Balance:{Colors.END}")
    print(f"  BUY:  {buy_pct:5.1f}% ({len(buy_trades)} trades)")
    print(f"  SELL: {sell_pct:5.1f}% ({len(sell_trades)} trades)")
    
    balance_diff = abs(buy_pct - sell_pct)
    if balance_diff < 20:
        print(f"  {Colors.GREEN}✅ Well balanced (diff: {balance_diff:.1f}%){Colors.END}")
    elif balance_diff < 40:
        print(f"  {Colors.YELLOW}⚠️  Moderate imbalance (diff: {balance_diff:.1f}%){Colors.END}")
    else:
        print(f"  {Colors.RED}❌ Severe imbalance (diff: {balance_diff:.1f}%){Colors.END}")
        if buy_pct > sell_pct:
            print(f"  {Colors.YELLOW}   → Too many BUY trades - BUY bias may still exist{Colors.END}")

def analyze_temporal_trends(df_decisions, df_trades):
    """Analyze performance trends over time"""
    print_section("📈 Temporal Trend Analysis")
    
    if df_decisions is None or len(df_decisions) == 0:
        print(f"{Colors.YELLOW}No decision data available{Colors.END}")
        return
    
    # Group by day
    df_decisions['date'] = df_decisions['timestamp'].dt.date
    daily_stats = []
    
    for date, group in df_decisions.groupby('date'):
        xgb_probs = group['xgb_prob'].dropna()
        if len(xgb_probs) == 0:
            continue
        
        pct_buy = (xgb_probs > 0.6).mean() * 100
        pct_sell = (xgb_probs < 0.4).mean() * 100
        pct_noise = ((xgb_probs >= 0.4) & (xgb_probs <= 0.6)).mean() * 100
        
        daily_stats.append({
            'date': date,
            'buy_pct': pct_buy,
            'sell_pct': pct_sell,
            'noise_pct': pct_noise,
            'decisions': len(group)
        })
    
    if len(daily_stats) == 0:
        print(f"{Colors.YELLOW}Not enough data for temporal analysis{Colors.END}")
        return
    
    df_daily = pd.DataFrame(daily_stats).sort_values('date')
    
    print(f"Analyzing {len(df_daily)} days of data\n")
    print(f"{'Date':<12} {'Decisions':>10} {'BUY%':>8} {'SELL%':>8} {'NOISE%':>8} {'Status':<10}")
    print("─" * 70)
    
    for _, row in df_daily.tail(14).iterrows():  # Last 14 days
        date_str = str(row['date'])
        decisions = int(row['decisions'])
        buy_pct = row['buy_pct']
        sell_pct = row['sell_pct']
        noise_pct = row['noise_pct']
        
        # Status indicator
        if 30 <= buy_pct <= 40 and 30 <= sell_pct <= 40:
            status = f"{Colors.GREEN}✅ Balanced{Colors.END}"
        elif buy_pct > 60:
            status = f"{Colors.RED}❌ BUY bias{Colors.END}"
        else:
            status = f"{Colors.YELLOW}⚠️  Improving{Colors.END}"
        
        print(f"{date_str:<12} {decisions:>10,} {buy_pct:>7.1f}% {sell_pct:>7.1f}% {noise_pct:>7.1f}% {status}")
    
    # Trend analysis
    if len(df_daily) >= 3:
        recent_buy = df_daily.tail(3)['buy_pct'].mean()
        older_buy = df_daily.head(3)['buy_pct'].mean() if len(df_daily) >= 6 else recent_buy
        
        print(f"\n{Colors.BOLD}Trend:{Colors.END}")
        if recent_buy < older_buy - 10:
            print(f"  {Colors.GREEN}✅ BUY bias is decreasing (improving){Colors.END}")
        elif recent_buy > older_buy + 10:
            print(f"  {Colors.RED}❌ BUY bias is increasing (worsening){Colors.END}")
        else:
            print(f"  {Colors.YELLOW}→ Stable (no significant trend){Colors.END}")

def generate_summary_report(df_decisions, df_trades):
    """Generate overall summary report"""
    print_header("📋 SUMMARY REPORT")
    
    if df_decisions is None:
        print(f"{Colors.RED}❌ Cannot generate report - no decision data{Colors.END}")
        return
    
    # Calculate key metrics
    xgb_probs = df_decisions['xgb_prob'].dropna()
    pct_buy = (xgb_probs > 0.6).mean() * 100
    pct_sell = (xgb_probs < 0.4).mean() * 100
    pct_noise = ((xgb_probs >= 0.4) & (xgb_probs <= 0.6)).mean() * 100
    
    # Overall status
    print(f"{Colors.BOLD}XGBoost Fix Status:{Colors.END}")
    
    if 30 <= pct_buy <= 40 and 30 <= pct_sell <= 40 and 20 <= pct_noise <= 40:
        print(f"  {Colors.GREEN}✅ FIX IS WORKING - Distribution is balanced!{Colors.END}")
        status_score = 100
    elif pct_buy > 70:
        print(f"  {Colors.RED}❌ FIX NOT WORKING - Severe BUY bias persists{Colors.END}")
        print(f"  {Colors.YELLOW}   Action required: Verify models were retrained with v6.1{Colors.END}")
        status_score = 0
    elif pct_buy > 50:
        print(f"  {Colors.YELLOW}⚠️  PARTIAL FIX - Moderate BUY bias remains{Colors.END}")
        print(f"  {Colors.YELLOW}   May need more time or further investigation{Colors.END}")
        status_score = 50
    else:
        print(f"  {Colors.GREEN}✅ FIX IS WORKING - Distribution is improving{Colors.END}")
        status_score = 75
    
    # Trading performance
    if df_trades is not None and len(df_trades) > 0:
        closed = df_trades[df_trades['status'] == 'CLOSED'] if 'status' in df_trades.columns else df_trades
        if len(closed) > 0:
            wins = closed[closed['profit'] > 0]
            win_rate = len(wins) / len(closed) * 100
            
            print(f"\n{Colors.BOLD}Trading Performance:{Colors.END}")
            if win_rate >= 50:
                print(f"  {Colors.GREEN}✅ Win rate: {win_rate:.1f}% - Excellent!{Colors.END}")
            elif win_rate >= 40:
                print(f"  {Colors.YELLOW}⚠️  Win rate: {win_rate:.1f}% - Acceptable{Colors.END}")
            else:
                print(f"  {Colors.RED}❌ Win rate: {win_rate:.1f}% - Needs improvement{Colors.END}")
    
    # Recommendations
    print(f"\n{Colors.BOLD}Recommendations:{Colors.END}")
    
    if status_score == 100:
        print(f"  • {Colors.GREEN}Continue monitoring - system is working well{Colors.END}")
        print(f"  • {Colors.GREEN}Focus on optimizing other parameters{Colors.END}")
    elif status_score >= 50:
        print(f"  • {Colors.YELLOW}Monitor for 2-3 more days to see if it stabilizes{Colors.END}")
        print(f"  • {Colors.YELLOW}Check if recent market conditions are unusual{Colors.END}")
    else:
        print(f"  • {Colors.RED}Verify models were retrained with v6.1 fix{Colors.END}")
        print(f"  • {Colors.RED}Check training logs for calibration effectiveness{Colors.END}")
        print(f"  • {Colors.RED}Consider re-running train_offline.py{Colors.END}")
    
    print(f"\n{Colors.CYAN}{'─'*70}{Colors.END}")
    print(f"{Colors.CYAN}Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    print(f"{Colors.CYAN}{'─'*70}{Colors.END}\n")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Monitor XGBoost fix performance')
    parser.add_argument('--quick', action='store_true', help='Quick summary only')
    parser.add_argument('--symbol', type=str, help='Analyze specific symbol only')
    parser.add_argument('--days', type=int, help='Analyze last N days only')
    args = parser.parse_args()
    
    print_header("🔍 XGBoost BUY Bias Fix Monitor v6.1")
    
    # Load data
    print(f"{Colors.CYAN}Loading data...{Colors.END}")
    df_decisions = load_ensemble_decisions(days=args.days)
    df_trades = load_trading_history(days=args.days)
    
    if df_decisions is None:
        print(f"\n{Colors.RED}❌ Cannot proceed without ensemble decisions data{Colors.END}")
        print(f"{Colors.YELLOW}Make sure the bot is running and ensemble_decisions.csv exists{Colors.END}")
        return 1
    
    if args.quick:
        # Quick summary only
        generate_summary_report(df_decisions, df_trades)
    else:
        # Full analysis
        analyze_prediction_distribution(df_decisions, symbol=args.symbol)
        analyze_ensemble_decisions(df_decisions)
        
        if df_trades is not None:
            analyze_trading_performance(df_trades)
        
        analyze_temporal_trends(df_decisions, df_trades)
        generate_summary_report(df_decisions, df_trades)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

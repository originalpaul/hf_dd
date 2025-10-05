#!/usr/bin/env python3
"""
Generate comparison charts showing gross returns vs net-of-fees returns
for all funds alongside S&P 500 benchmark.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "charts")

DEFAULT_GROSS_CSV = os.path.join(DATA_DIR, "fund_returns_clean.csv")
DEFAULT_NET_CSV = os.path.join(DATA_DIR, "fund_returns_net_all_fees.csv")

FUNDS = ["Fund 1", "Fund 2", "Fund 3"]
BENCHMARK = "S&P 500 w/ Div"


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(gross_csv: str, net_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both gross and net returns data."""
    print(f"Loading gross returns from: {gross_csv}")
    df_gross = pd.read_csv(gross_csv, parse_dates=["Date"]).set_index("Date").sort_index()
    
    print(f"Loading net returns from: {net_csv}")
    df_net = pd.read_csv(net_csv, parse_dates=["Date"]).set_index("Date").sort_index()
    
    print(f"Gross data shape: {df_gross.shape}")
    print(f"Net data shape: {df_net.shape}")
    print(f"Date range: {df_gross.index.min()} to {df_gross.index.max()}")
    
    return df_gross, df_net


def plot_cumulative_comparison(df_gross: pd.DataFrame, df_net: pd.DataFrame, output_path: str):
    """Plot cumulative returns comparison: gross vs net vs benchmark."""
    plt.figure(figsize=(14, 8))
    
    # Calculate cumulative returns
    cum_gross = (1 + df_gross).cumprod()
    cum_net = (1 + df_net).cumprod()
    
    # Plot benchmark
    plt.plot(cum_gross.index, cum_gross[BENCHMARK], 
             label=f"{BENCHMARK} (Benchmark)", 
             linewidth=2, color='black', linestyle='--')
    
    # Plot funds - gross returns
    colors = ['blue', 'red', 'green']
    for i, fund in enumerate(FUNDS):
        if fund in cum_gross.columns:
            plt.plot(cum_gross.index, cum_gross[fund], 
                    label=f"{fund} (Gross)", 
                    linewidth=2, color=colors[i], linestyle='-', alpha=0.7)
    
    # Plot funds - net returns
    for i, fund in enumerate(FUNDS):
        if fund in cum_net.columns:
            plt.plot(cum_net.index, cum_net[fund], 
                    label=f"{fund} (Net of Fees)", 
                    linewidth=2, color=colors[i], linestyle=':', alpha=0.9)
    
    plt.title("Cumulative Returns: Gross vs Net of Fees\n(Starting at $1.00)", 
              fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Wealth", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cumulative comparison chart to: {output_path}")


def plot_drawdown_comparison(df_gross: pd.DataFrame, df_net: pd.DataFrame, output_path: str):
    """Plot drawdown comparison: gross vs net vs benchmark."""
    plt.figure(figsize=(14, 8))
    
    # Calculate drawdowns
    def calc_drawdown(returns):
        wealth = (1 + returns).cumprod()
        peak = wealth.cummax()
        return wealth / peak - 1
    
    dd_gross = calc_drawdown(df_gross)
    dd_net = calc_drawdown(df_net)
    
    # Plot benchmark drawdown
    plt.plot(dd_gross.index, dd_gross[BENCHMARK], 
             label=f"{BENCHMARK} (Benchmark)", 
             linewidth=2, color='black', linestyle='--')
    
    # Plot funds - gross drawdowns
    colors = ['blue', 'red', 'green']
    for i, fund in enumerate(FUNDS):
        if fund in dd_gross.columns:
            plt.plot(dd_gross.index, dd_gross[fund], 
                    label=f"{fund} (Gross)", 
                    linewidth=2, color=colors[i], linestyle='-', alpha=0.7)
    
    # Plot funds - net drawdowns
    for i, fund in enumerate(FUNDS):
        if fund in dd_net.columns:
            plt.plot(dd_net.index, dd_net[fund], 
                    label=f"{fund} (Net of Fees)", 
                    linewidth=2, color=colors[i], linestyle=':', alpha=0.9)
    
    plt.title("Drawdown Comparison: Gross vs Net of Fees", 
              fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Drawdown", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, 0.1)  # Focus on drawdown range
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved drawdown comparison chart to: {output_path}")


def plot_annual_returns_comparison(df_gross: pd.DataFrame, df_net: pd.DataFrame, output_path: str):
    """Plot annual returns comparison bar chart."""
    # Calculate annual returns
    def calc_annual_returns(df):
        annual_returns = {}
        for col in df.columns:
            if col in FUNDS + [BENCHMARK]:
                # Resample to annual and calculate total return
                annual = (1 + df[col]).resample('Y').prod() - 1
                annual_returns[col] = annual
        return annual_returns
    
    gross_annual = calc_annual_returns(df_gross)
    net_annual = calc_annual_returns(df_net)
    
    # Get years
    years = gross_annual[BENCHMARK].index.year
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    # Plot each fund + benchmark
    colors = ['blue', 'red', 'green']
    for i, fund in enumerate(FUNDS + [BENCHMARK]):
        if i >= 4:
            break
            
        ax = axes[i]
        
        # Plot gross returns
        if fund in gross_annual:
            ax.bar([y - 0.2 for y in years], gross_annual[fund].values, 
                   width=0.4, label=f"{fund} (Gross)", 
                   color=colors[i % len(colors)], alpha=0.7)
        
        # Plot net returns
        if fund in net_annual:
            ax.bar([y + 0.2 for y in years], net_annual[fund].values, 
                   width=0.4, label=f"{fund} (Net of Fees)", 
                   color=colors[i % len(colors)], alpha=0.9, linestyle='--')
        
        ax.set_title(f"{fund} Annual Returns", fontweight='bold')
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual Return")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(years[::2])  # Show every other year
    
    plt.suptitle("Annual Returns Comparison: Gross vs Net of Fees", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved annual returns comparison chart to: {output_path}")


def print_summary_stats(df_gross: pd.DataFrame, df_net: pd.DataFrame):
    """Print summary statistics comparing gross vs net returns."""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY: GROSS vs NET OF FEES")
    print("="*80)
    
    def calc_cagr(returns):
        return ((1 + returns).prod()) ** (12 / len(returns)) - 1
    
    def calc_max_dd(returns):
        wealth = (1 + returns).cumprod()
        peak = wealth.cummax()
        return (wealth / peak - 1).min()
    
    def calc_sharpe(returns, rf=0.02):
        excess_returns = returns - rf/12  # Monthly risk-free rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(12)
    
    print(f"{'Metric':<15} {'Fund 1':<20} {'Fund 2':<20} {'Fund 3':<20} {'S&P 500':<20}")
    print("-" * 95)
    
    # CAGR
    print(f"{'CAGR (Gross)':<15}", end="")
    for fund in FUNDS + [BENCHMARK]:
        if fund in df_gross.columns:
            cagr = calc_cagr(df_gross[fund])
            print(f"{cagr:>8.1%}{'':>11}", end="")
    print()
    
    print(f"{'CAGR (Net)':<15}", end="")
    for fund in FUNDS + [BENCHMARK]:
        if fund in df_net.columns:
            cagr = calc_cagr(df_net[fund])
            print(f"{cagr:>8.1%}{'':>11}", end="")
    print()
    
    # Max Drawdown
    print(f"{'Max DD (Gross)':<15}", end="")
    for fund in FUNDS + [BENCHMARK]:
        if fund in df_gross.columns:
            mdd = calc_max_dd(df_gross[fund])
            print(f"{mdd:>8.1%}{'':>11}", end="")
    print()
    
    print(f"{'Max DD (Net)':<15}", end="")
    for fund in FUNDS + [BENCHMARK]:
        if fund in df_net.columns:
            mdd = calc_max_dd(df_net[fund])
            print(f"{mdd:>8.1%}{'':>11}", end="")
    print()
    
    # Sharpe Ratio
    print(f"{'Sharpe (Gross)':<15}", end="")
    for fund in FUNDS + [BENCHMARK]:
        if fund in df_gross.columns:
            sharpe = calc_sharpe(df_gross[fund])
            print(f"{sharpe:>8.2f}{'':>11}", end="")
    print()
    
    print(f"{'Sharpe (Net)':<15}", end="")
    for fund in FUNDS + [BENCHMARK]:
        if fund in df_net.columns:
            sharpe = calc_sharpe(df_net[fund])
            print(f"{sharpe:>8.2f}{'':>11}", end="")
    print()


def main(gross_csv: str, net_csv: str):
    """Generate all comparison charts and statistics."""
    ensure_output_dir()
    
    # Load data
    df_gross, df_net = load_data(gross_csv, net_csv)
    
    # Generate charts
    print("\nGenerating comparison charts...")
    
    # Cumulative returns comparison
    cum_path = os.path.join(OUTPUT_DIR, "gross_vs_net_cumulative_returns.png")
    plot_cumulative_comparison(df_gross, df_net, cum_path)
    
    # Drawdown comparison
    dd_path = os.path.join(OUTPUT_DIR, "gross_vs_net_drawdowns.png")
    plot_drawdown_comparison(df_gross, df_net, dd_path)
    
    # Annual returns comparison
    annual_path = os.path.join(OUTPUT_DIR, "gross_vs_net_annual_returns.png")
    plot_annual_returns_comparison(df_gross, df_net, annual_path)
    
    # Print summary statistics
    print_summary_stats(df_gross, df_net)
    
    print(f"\nAll charts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gross vs net returns comparison charts.")
    parser.add_argument("--gross_csv", default=DEFAULT_GROSS_CSV, 
                       help="CSV with gross returns.")
    parser.add_argument("--net_csv", default=DEFAULT_NET_CSV, 
                       help="CSV with net-of-fees returns.")
    args = parser.parse_args()
    
    main(args.gross_csv, args.net_csv)

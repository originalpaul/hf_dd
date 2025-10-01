# scripts/generate_reports.py
"""
Generate summary tables and charts for Fund 1/2/3 vs S&P 500.

Requirements:
- data/fund_returns_clean.csv  (with columns: Date, Fund 1, Fund 2, Fund 3, S&P 500 w/ Div)
- src/performance_metrics.py   (the module you already created)

Outputs:
- outputs/tables/summary_stats.csv
- outputs/charts/cumulative_returns.png
- outputs/charts/drawdown_Fund_1.png
- outputs/charts/drawdown_Fund_2.png
- outputs/charts/drawdown_Fund_3.png
- outputs/charts/drawdown_S&P_500_w_Div.png
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# If running as a script, make sure Python can find src/performance_metrics.py
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from performance_metrics import summary_table, cumulative_returns, drawdown_series

DEFAULT_DATA = os.path.join("data", "fund_returns_clean.csv")
OUT_TABLE = os.path.join("outputs", "tables", "summary_stats.csv")
OUT_CHARTS_DIR = os.path.join("outputs", "charts")

FUNDS = ["Fund 1", "Fund 2", "Fund 3"]
BENCH = "S&P 500 w/ Div"
ALL_COLS = FUNDS + [BENCH]


def ensure_dirs():
    os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)
    os.makedirs(OUT_CHARTS_DIR, exist_ok=True)


def load_returns(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    # keep only expected columns if present
    missing = [c for c in ALL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")
    return df[ALL_COLS].astype(float)


def save_summary_table(df: pd.DataFrame, rf_annual: float = 0.02) -> pd.DataFrame:
    tbl = summary_table(df, bench_col=BENCH, rf=rf_annual)
    tbl.to_csv(OUT_TABLE)
    return tbl


def plot_cumulative(df: pd.DataFrame, out_path: str):
    cum = (1.0 + df).cumprod()
    plt.figure(figsize=(10, 6))
    cum.plot(ax=plt.gca(), linewidth=1.5)
    plt.title("Cumulative Wealth (Starting at 1.0)")
    plt.xlabel("Date")
    plt.ylabel("Wealth Index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_drawdowns(df: pd.DataFrame):
    # One drawdown chart per series
    for col in df.columns:
        dd = drawdown_series(df[col])
        plt.figure(figsize=(10, 4))
        dd.plot(ax=plt.gca(), linewidth=1.5)
        plt.title(f"Drawdown — {col}")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        safe_name = col.replace(" ", "_").replace("/", "_").replace("%", "pct")
        out = os.path.join(OUT_CHARTS_DIR, f"drawdown_{safe_name}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()


def main(csv_path: str, rf_annual: float):
    ensure_dirs()
    df = load_returns(csv_path)
    # Save summary table
    tbl = save_summary_table(df, rf_annual=rf_annual)
    print("Saved summary table to:", OUT_TABLE)
    print(tbl.round(4))

    # Save cumulative returns plot
    cum_path = os.path.join(OUT_CHARTS_DIR, "cumulative_returns.png")
    plot_cumulative(df, cum_path)
    print("Saved cumulative returns chart to:", cum_path)

    # Save drawdown plots
    plot_drawdowns(df)
    print("Saved drawdown charts to:", OUT_CHARTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hedge fund DD tables & charts.")
    parser.add_argument("--csv", default=DEFAULT_DATA, help="Path to returns CSV.")
    parser.add_argument("--rf", type=float, default=0.02, help="Annual risk-free rate (e.g., 0.02 = 2%).")
    args = parser.parse_args()
    main(args.csv, args.rf)

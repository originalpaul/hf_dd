#!/usr/bin/env python3
"""
Apply both management fees and performance fees to fund returns.
Performance fees use high-water mark logic (only charged on new highs).
"""

import os
import argparse
import pandas as pd
import numpy as np


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_CSV = os.path.join(DATA_DIR, "fund_returns_clean.csv")
DEFAULT_OUT = os.path.join(DATA_DIR, "fund_returns_net_all_fees.csv")

# Fee structure from case study
FUND_FEES = {
    "Fund 1": {"mgmt": 0.015, "perf": 0.20},  # 1.5% mgmt, 20% perf
    "Fund 2": {"mgmt": 0.035, "perf": 0.35},  # 3.5% mgmt, 35% perf  
    "Fund 3": {"mgmt": 0.0175, "perf": 0.20}, # 1.75% mgmt, 20% perf
}


def infer_periods_per_year(idx: pd.Index) -> int:
    """Infer periods per year from DatetimeIndex frequency."""
    freq = pd.infer_freq(idx)
    if not freq:
        return 12
    f = freq.upper()
    if f.startswith("M"):
        return 12
    if f.startswith("W"):
        return 52
    if f.startswith("B") or f.startswith("D"):
        return 252
    if f.startswith("Q"):
        return 4
    if f.startswith("A") or f.startswith("Y"):
        return 1
    return 12


def apply_management_fee(returns: pd.Series, annual_mgmt_fee: float, periods_per_year: int) -> pd.Series:
    """Apply management fee (fixed deduction each period)."""
    period_mgmt_fee = annual_mgmt_fee / periods_per_year
    return returns - period_mgmt_fee


def apply_performance_fee(returns: pd.Series, perf_fee_rate: float, periods_per_year: int) -> pd.Series:
    """
    Apply performance fee with high-water mark logic.
    Performance fees are typically calculated annually, so we'll apply them at year-end.
    """
    if periods_per_year != 12:  # Only handle monthly data for now
        print(f"Warning: Performance fee calculation assumes monthly data (got {periods_per_year} periods/year)")
        return returns
    
    # Calculate cumulative wealth
    wealth = (1 + returns).cumprod()
    
    # Find year-end periods
    year_ends = wealth.index[wealth.index.month == 12]
    
    # Initialize high-water mark
    high_water_mark = 1.0
    net_returns = returns.copy()
    
    for year_end in year_ends:
        # Get year-end wealth
        year_end_wealth = wealth.loc[year_end]
        
        # Check if we have a new high
        if year_end_wealth > high_water_mark:
            # Calculate performance fee on the gain above high-water mark
            gain_above_hwm = year_end_wealth - high_water_mark
            performance_fee = gain_above_hwm * perf_fee_rate
            
            # Convert fee back to return impact
            # If wealth was W and fee is F, new wealth is W - F
            # So return impact is (W - F) / W - 1 = -F / W
            fee_return_impact = -performance_fee / year_end_wealth
            
            # Apply the fee to the year-end return
            net_returns.loc[year_end] += fee_return_impact
            
            # Update high-water mark
            high_water_mark = year_end_wealth - performance_fee
    
    return net_returns


def apply_all_fees(df: pd.DataFrame, fund_fees: dict) -> pd.DataFrame:
    """
    Apply both management and performance fees to fund returns.
    """
    df_net = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DateTimeIndex (parsed dates)")
    
    periods_per_year = infer_periods_per_year(df.index)
    
    for fund_name, fees in fund_fees.items():
        if fund_name not in df.columns:
            print(f"Warning: {fund_name} not found in data, skipping")
            continue
            
        print(f"Applying fees to {fund_name}: {fees['mgmt']:.1%} mgmt, {fees['perf']:.1%} perf")
        
        # Apply management fee (fixed deduction each period)
        df_net[fund_name] = apply_management_fee(
            df[fund_name], 
            fees['mgmt'], 
            periods_per_year
        )
        
        # Apply performance fee (high-water mark logic)
        df_net[fund_name] = apply_performance_fee(
            df_net[fund_name], 
            fees['perf'], 
            periods_per_year
        )
    
    return df_net


def main(csv_in: str, csv_out: str):
    """Load returns, apply fees, and save net returns."""
    print(f"Loading returns from: {csv_in}")
    df = pd.read_csv(csv_in, parse_dates=["Date"]).set_index("Date").sort_index()
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {list(df.columns)}")
    
    # Apply fees
    df_net = apply_all_fees(df, FUND_FEES)
    
    # Save results
    df_net.to_csv(csv_out, index=True)
    print(f"Saved net-of-all-fees returns to: {csv_out}")
    
    # Show impact summary
    print("\nFee Impact Summary:")
    for fund_name in FUND_FEES.keys():
        if fund_name in df.columns:
            gross_cagr = ((1 + df[fund_name]).prod()) ** (12 / len(df)) - 1
            net_cagr = ((1 + df_net[fund_name]).prod()) ** (12 / len(df_net)) - 1
            impact = net_cagr - gross_cagr
            print(f"{fund_name}: Gross CAGR {gross_cagr:.1%} → Net CAGR {net_cagr:.1%} (impact: {impact:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply management and performance fees to fund returns.")
    parser.add_argument("--csv_in", default=DEFAULT_CSV, help="Input CSV with gross returns.")
    parser.add_argument("--csv_out", default=DEFAULT_OUT, help="Output CSV with net-of-all-fees returns.")
    args = parser.parse_args()
    
    main(args.csv_in, args.csv_out)

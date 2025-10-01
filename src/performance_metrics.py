# src/performance_metrics.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

# ------------------------------
# Helpers
# ------------------------------

_FREQ_TO_PPY = {
    "B": 252, "C": 252, "D": 252, "W": 52,
    "M": 12, "SM": 24, "BM": 12, "CBM": 12,
    "Q": 4, "BQ": 4, "QS": 4, "A": 1, "Y": 1,
}

def _infer_periods_per_year(idx: pd.Index) -> int:
    """
    Infer periods per year from a DatetimeIndex frequency.
    Falls back to 12 (monthly) if unknown.
    """
    if not isinstance(idx, pd.DatetimeIndex):
        return 12
    freqstr = pd.infer_freq(idx)
    if freqstr is None:
        # try to guess from median spacing
        if len(idx) > 1:
            delta_days = np.median(np.diff(idx.values).astype('timedelta64[D]').astype(int))
            if 27 <= delta_days <= 32:
                return 12
            if 6 <= delta_days <= 8:
                return 52
            if 360 <= delta_days <= 370:
                return 1
            if 20 <= delta_days <= 23:
                return 252
        return 12
    # Normalize freq string (e.g., 'M', 'BM', 'B', 'W', 'Q', 'A')
    for k in _FREQ_TO_PPY:
        if freqstr.upper().startswith(k):
            return _FREQ_TO_PPY[k]
    return 12

def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.copy()
    if isinstance(x, (pd.DataFrame,)):
        if x.shape[1] != 1:
            raise ValueError("Expected a Series or a single-column DataFrame.")
        return x.iloc[:, 0].copy()
    return pd.Series(x)

def _clean_returns(r: pd.Series) -> pd.Series:
    r = _to_series(r).astype(float)
    return r.replace([np.inf, -np.inf], np.nan).dropna()

# ------------------------------
# Core metrics
# ------------------------------

def cumulative_returns(r: pd.Series) -> pd.Series:
    """Cumulative wealth curve starting at 1.0."""
    r = _clean_returns(r)
    return (1.0 + r).cumprod()

def cagr(r: pd.Series, periods_per_year: Optional[int] = None) -> float:
    """Compound Annual Growth Rate based on start/end wealth and time span."""
    r = _clean_returns(r)
    if r.empty:
        return np.nan
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(r.index)
    total_return = (1.0 + r).prod()
    years = len(r) / periods_per_year
    if years <= 0:
        return np.nan
    return total_return ** (1 / years) - 1

def annualized_return(r: pd.Series, periods_per_year: Optional[int] = None) -> float:
    """Alias to CAGR for monthly-like series."""
    return cagr(r, periods_per_year)

def annualized_vol(r: pd.Series, periods_per_year: Optional[int] = None) -> float:
    r = _clean_returns(r)
    if r.empty:
        return np.nan
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(r.index)
    return r.std(ddof=1) * np.sqrt(periods_per_year)

def downside_deviation(r: pd.Series, mar: float = 0.0, periods_per_year: Optional[int] = None) -> float:
    """
    Annualized downside deviation using MAR (minimum acceptable return) per period.
    """
    r = _clean_returns(r)
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(r.index)
    dd = r[r < mar] - mar
    if dd.empty:
        return 0.0
    return np.sqrt((dd**2).mean()) * np.sqrt(periods_per_year)

def sharpe(r: pd.Series, rf: float = 0.0, periods_per_year: Optional[int] = None) -> float:
    """
    Annualized Sharpe. `rf` is annual risk-free; converted to per-period internally.
    """
    r = _clean_returns(r)
    if r.empty:
        return np.nan
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(r.index)
    rf_period = (1 + rf) ** (1 / periods_per_year) - 1
    ex = r - rf_period
    vol = annualized_vol(ex, periods_per_year)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return cagr(ex, periods_per_year) / vol

def sortino(r: pd.Series, rf: float = 0.0, periods_per_year: Optional[int] = None, mar: float = 0.0) -> float:
    """
    Annualized Sortino; `mar` is per-period MAR (default 0).
    rf is annual RF and adjusted to per-period for the numerator.
    """
    r = _clean_returns(r)
    if r.empty:
        return np.nan
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(r.index)
    rf_period = (1 + rf) ** (1 / periods_per_year) - 1
    ex = r - rf_period
    an_ret = cagr(ex, periods_per_year)
    dd = downside_deviation(ex, mar=mar, periods_per_year=periods_per_year)
    if dd == 0:
        return np.nan
    return an_ret / dd

def drawdown_series(r: pd.Series) -> pd.Series:
    """Compute drawdown series from the cumulative wealth curve."""
    wealth = cumulative_returns(r)
    peaks = wealth.cummax()
    dd = wealth / peaks - 1.0
    return dd

def max_drawdown(r: pd.Series) -> float:
    dd = drawdown_series(r)
    if dd.empty:
        return np.nan
    return dd.min()

def calmar(r: pd.Series, periods_per_year: Optional[int] = None) -> float:
    """Calmar = CAGR / |Max DD|."""
    mdd = max_drawdown(r)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return cagr(r, periods_per_year) / abs(mdd)

def hit_ratio(r: pd.Series) -> float:
    """Fraction of periods with positive return."""
    r = _clean_returns(r)
    if r.empty:
        return np.nan
    return (r > 0).mean()

def skewness(r: pd.Series) -> float:
    r = _clean_returns(r)
    return r.skew()

def kurtosis_excess(r: pd.Series) -> float:
    r = _clean_returns(r)
    return r.kurt()

# ------------------------------
# Benchmark & relative stats
# ------------------------------

@dataclass
class RegressionResult:
    alpha_ann: float
    beta: float
    r2: float

def alpha_beta(
    r: pd.Series,
    bench: pd.Series,
    periods_per_year: Optional[int] = None,
) -> RegressionResult:
    """
    OLS regression of excess returns r on bench:
        r_t = a + b * bench_t + eps_t
    Alpha annualized using periods_per_year.
    """
    r = _clean_returns(r)
    b = _clean_returns(bench)
    df = pd.concat([r, b], axis=1).dropna()
    if df.shape[0] < 3:
        return RegressionResult(np.nan, np.nan, np.nan)
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    x1 = np.column_stack((np.ones(len(x)), x))
    beta_hat = np.linalg.lstsq(x1, y, rcond=None)[0]
    a, b_ = beta_hat[0], beta_hat[1]
    y_hat = x1 @ beta_hat
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(df.index)
    alpha_ann = (1 + a) ** periods_per_year - 1
    return RegressionResult(alpha_ann=alpha_ann, beta=b_, r2=r2)

def correlation(r: pd.Series, bench: pd.Series) -> float:
    df = pd.concat([_clean_returns(r), _clean_returns(bench)], axis=1).dropna()
    if df.empty:
        return np.nan
    return df.corr().iloc[0, 1]

def tracking_error(r: pd.Series, bench: pd.Series, periods_per_year: Optional[int] = None) -> float:
    diff = pd.concat([_clean_returns(r), _clean_returns(bench)], axis=1).dropna()
    if diff.empty:
        return np.nan
    d = diff.iloc[:, 0] - diff.iloc[:, 1]
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(diff.index)
    return d.std(ddof=1) * np.sqrt(periods_per_year)

def information_ratio(r: pd.Series, bench: pd.Series, periods_per_year: Optional[int] = None) -> float:
    diff = pd.concat([_clean_returns(r), _clean_returns(bench)], axis=1).dropna()
    if diff.empty:
        return np.nan
    d = diff.iloc[:, 0] - diff.iloc[:, 1]
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(diff.index)
    ann_excess = cagr(d, periods_per_year)
    te = tracking_error(r, bench, periods_per_year)
    if te == 0 or np.isnan(te):
        return np.nan
    return ann_excess / te

# ------------------------------
# Tail risk (historical)
# ------------------------------

def var_historical(r: pd.Series, level: float = 0.95) -> float:
    """
    Historical VaR at confidence `level`. Returns negative number (loss).
    Example: level=0.95 -> 5th percentile.
    """
    r = _clean_returns(r)
    if r.empty:
        return np.nan
    q = np.quantile(r, 1 - level)
    return q

def cvar_historical(r: pd.Series, level: float = 0.95) -> float:
    """
    Historical CVaR (Expected Shortfall): average of returns <= VaR.
    """
    r = _clean_returns(r)
    if r.empty:
        return np.nan
    v = var_historical(r, level=level)
    tail = r[r <= v]
    if tail.empty:
        return np.nan
    return tail.mean()

# ------------------------------
# Rolling metrics
# ------------------------------

def rolling_sharpe(r: pd.Series, window: int, rf: float = 0.0) -> pd.Series:
    """
    Rolling *non-annualized* Sharpe over a fixed window (for charting).
    For an annualized version, multiply by sqrt(PPY) outside this function.
    """
    r = _clean_returns(r)
    roll_mean = r.rolling(window).mean()
    roll_std = r.rolling(window).std(ddof=1)
    s = (roll_mean - rf / window) / roll_std
    return s

def rolling_max_drawdown(r: pd.Series, window: int) -> pd.Series:
    r = _clean_returns(r)
    wealth = (1 + r).cumprod()
    roll_max = wealth.rolling(window).max()
    dd = wealth / roll_max - 1.0
    return dd

# ------------------------------
# Summary
# ------------------------------

def summary(
    r: pd.Series,
    bench: Optional[pd.Series] = None,
    rf: float = 0.0,
    periods_per_year: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute a dictionary of common performance stats.
    """
    r = _clean_returns(r)
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(r.index)

    stats = {
        "CAGR": cagr(r, periods_per_year),
        "AnnVol": annualized_vol(r, periods_per_year),
        "Sharpe": sharpe(r, rf, periods_per_year),
        "Sortino": sortino(r, rf, periods_per_year),
        "MaxDrawdown": max_drawdown(r),
        "Calmar": calmar(r, periods_per_year),
        "HitRatio": hit_ratio(r),
        "Skew": skewness(r),
        "KurtosisExcess": kurtosis_excess(r),
        "VaR_95": var_historical(r, 0.95),
        "CVaR_95": cvar_historical(r, 0.95),
    }

    if bench is not None:
        reg = alpha_beta(r, bench, periods_per_year)
        stats.update({
            "Alpha_ann": reg.alpha_ann,
            "Beta": reg.beta,
            "R2": reg.r2,
            "Corr_SPX": correlation(r, bench),
            "TrackingError": tracking_error(r, bench, periods_per_year),
            "InformationRatio": information_ratio(r, bench, periods_per_year),
        })

    return stats

def summary_table(
    returns_df: pd.DataFrame,
    bench_col: Optional[str] = None,
    rf: float = 0.0,
) -> pd.DataFrame:
    """
    Build a nice table of stats for multiple columns in returns_df.
    If bench_col is provided, Alpha/Beta/IR/TE/Corr are relative to that column.
    """
    df = returns_df.copy()
    # attempt to parse date column if present
    if "Date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

    bench = None
    if bench_col is not None:
        if bench_col not in df.columns:
            raise ValueError(f"bench_col '{bench_col}' not found in DataFrame.")
        bench = df[bench_col]

    out = {}
    for col in df.columns:
        if bench_col is not None and col == bench_col:
            # show benchmark stats too (without relative metrics targeting itself)
            out[col] = summary(df[col], bench=None, rf=rf)
        else:
            out[col] = summary(df[col], bench=bench, rf=rf)

    return pd.DataFrame(out).T

# ------------------------------
# Example usage (comment out in production)
# ------------------------------
# if __name__ == "__main__":
#     data = pd.read_csv("../data/fund_returns_clean.csv", parse_dates=["Date"]).set_index("Date")
#     table = summary_table(data[["Fund 1", "Fund 2", "Fund 3", "S&P 500 w/ Div"]], bench_col="S&P 500 w/ Div", rf=0.02)
#     print(table.round(4))

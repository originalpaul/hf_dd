import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(__file__))  # repo root
DATA_CSV = os.path.join(BASE, "data", "fund_returns_clean.csv")
OUT_DIR = os.path.join(BASE, "outputs", "tables")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "summary_stats.csv")

def infer_ppy(idx):
    freq = pd.infer_freq(idx)
    if not freq:
        return 12
    f = freq.upper()
    if f.startswith("M"): return 12
    if f.startswith("W"): return 52
    if f.startswith("B") or f.startswith("D"): return 252
    if f.startswith("Q"): return 4
    if f.startswith("A") or f.startswith("Y"): return 1
    return 12

def cagr(r, ppy=None):
    r = pd.Series(r).dropna()
    if r.empty: return np.nan
    ppy = ppy or infer_ppy(r.index)
    tot = (1+r).prod()
    yrs = len(r)/ppy
    return np.nan if yrs<=0 else tot**(1/yrs)-1

def ann_vol(r, ppy=None):
    r = pd.Series(r).dropna()
    if r.empty: return np.nan
    ppy = ppy or infer_ppy(r.index)
    return r.std(ddof=1)*np.sqrt(ppy)

def sharpe(r, rf=0.02, ppy=None):
    r = pd.Series(r).dropna()
    if r.empty: return np.nan
    ppy = ppy or infer_ppy(r.index)
    rf_per = (1+rf)**(1/ppy)-1
    ex = r - rf_per
    ret = cagr(ex, ppy)
    vol = ann_vol(ex, ppy)
    return np.nan if vol==0 or np.isnan(vol) else ret/vol

def sortino(r, rf=0.02, ppy=None, mar=0.0):
    r = pd.Series(r).dropna()
    ppy = ppy or infer_ppy(r.index)
    rf_per = (1+rf)**(1/ppy)-1
    ex = r - rf_per
    dd = ex[ex < mar] - mar
    if dd.empty: return np.nan
    ret = cagr(ex, ppy)
    downside = np.sqrt((dd**2).mean())*np.sqrt(ppy)
    return np.nan if downside==0 else ret/downside

def drawdown_series(r):
    wealth = (1+r).cumprod()
    peak = wealth.cummax()
    return wealth/peak - 1.0

df = pd.read_csv(DATA_CSV, parse_dates=["Date"]).set_index("Date").sort_index()
cols = ["Fund 1", "Fund 2", "Fund 3", "S&P 500 w/ Div"]
df = df[cols].astype(float)

ppy = infer_ppy(df.index)
rf_annual = 0.02
bench = df["S&P 500 w/ Div"]

rows = []
for c in cols:
    r = df[c]
    mdd = drawdown_series(r).min()
    rows.append({
        "Series": c,
        "CAGR": cagr(r, ppy),
        "AnnVol": ann_vol(r, ppy),
        "Sharpe": sharpe(r, rf_annual, ppy),
        "Sortino": sortino(r, rf_annual, ppy),
        "MaxDrawdown": mdd,
        "Calmar": (cagr(r, ppy)/abs(mdd)) if mdd else np.nan,
        "HitRatio": (r > 0).mean(),
        "Skew": r.dropna().skew(),
        "KurtosisExcess": r.dropna().kurt(),
    })

summary = pd.DataFrame(rows).set_index("Series")
summary.to_csv(OUT_CSV)
print(f"Saved summary stats → {OUT_CSV}")

import os
import re
import argparse
import pandas as pd

from docx import Document


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_CSV = os.path.join(DATA_DIR, "fund_returns_clean.csv")
DEFAULT_OUT = os.path.join(DATA_DIR, "fund_returns_net_mgmt.csv")

DOCX_MAP = {
    "Fund 1": os.path.join(DATA_DIR, "CASESTUDY_FALL25_FUND1.docx"),
    "Fund 2": os.path.join(DATA_DIR, "CASESTUDY_FALL25_FUND2.docx"),
    "Fund 3": os.path.join(DATA_DIR, "CASESTUDY_FALL25_FUND3.docx"),
}


def extract_management_fee_from_docx(docx_path: str) -> float:
    """
    Parse a DOCX file and attempt to extract the annual management fee percentage as a decimal.
    Returns a float like 0.02 for 2%. Raises ValueError if not found.
    """
    doc = Document(docx_path)
    text = "\n".join(p.text for p in doc.paragraphs)
    # Common fee patterns like "management fee 2%", "2% management fee", "mgmt fee: 1.5%"
    patterns = [
        r"management\s*fee\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*%",
        r"(\d+(?:\.\d+)?)\s*%\s*management\s*fee",
        r"mgmt\s*fee\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*%",
        r"management\s*fee\s*of\s*(\d+(?:\.\d+)?)\s*%",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1)) / 100.0
    raise ValueError(f"Could not find management fee in {docx_path}")


def infer_periods_per_year(idx: pd.Index) -> int:
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


def apply_management_fees(df: pd.DataFrame, fund_to_fee: dict) -> pd.DataFrame:
    """
    Given a DataFrame of periodic returns and a mapping of fund name -> annual mgmt fee (decimal),
    subtract the pro-rated management fee each period from those fund return columns.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DateTimeIndex (parsed dates)")
    ppy = infer_periods_per_year(df.index)
    period_fee = {k: v / ppy for k, v in fund_to_fee.items()}
    for fund, fee_ann in fund_to_fee.items():
        if fund not in df.columns:
            continue
        df[fund] = df[fund] - period_fee[fund]
    return df


def load_fees_from_docs() -> dict:
    fees = {}
    for fund, path in DOCX_MAP.items():
        try:
            fees[fund] = extract_management_fee_from_docx(path)
        except Exception as e:
            # Fallback: leave as missing; user can override via CLI
            pass
    return fees


def main(csv_in: str, csv_out: str, override_fees: dict):
    df = pd.read_csv(csv_in, parse_dates=["Date"]).set_index("Date").sort_index()
    fees = load_fees_from_docs()
    fees.update({k: v for k, v in override_fees.items() if v is not None})
    if not fees:
        raise ValueError("No fees found. Provide --fund1, --fund2, --fund3 or ensure DOCX contain fees.")
    df_net = apply_management_fees(df, fees)
    df_net.to_csv(csv_out, index=True)
    print("Wrote net-of-management-fee returns to:", csv_out)
    print("Fees used (annual):", fees)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply annual management fees to periodic fund returns.")
    parser.add_argument("--csv_in", default=DEFAULT_CSV, help="Input CSV with gross returns.")
    parser.add_argument("--csv_out", default=DEFAULT_OUT, help="Output CSV with net-of-management-fee returns.")
    parser.add_argument("--fund1", type=float, default=None, help="Override Fund 1 annual mgmt fee (e.g., 2 for 2%).")
    parser.add_argument("--fund2", type=float, default=None, help="Override Fund 2 annual mgmt fee (e.g., 1.5 for 1.5%).")
    parser.add_argument("--fund3", type=float, default=None, help="Override Fund 3 annual mgmt fee (e.g., 2.5 for 2.5%).")
    args = parser.parse_args()

    overrides = {
        "Fund 1": None if args.fund1 is None else args.fund1 / 100.0,
        "Fund 2": None if args.fund2 is None else args.fund2 / 100.0,
        "Fund 3": None if args.fund3 is None else args.fund3 / 100.0,
    }
    main(args.csv_in, args.csv_out, overrides)



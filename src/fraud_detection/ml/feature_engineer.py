"""
VAT Fraud Feature Engineer
===========================
Transforms raw invoice/transaction DataFrame rows into ML-ready features.

Feature groups:
  - Amount features      : net_amount, vat_amount, roundness, Benford deviation
  - VAT rate features    : declared rate, rate anomaly flag
  - Declaration features : mismatch between seller/buyer declarations
  - Temporal features    : payment speed, filing gaps, registration age
  - Network proxy        : cross-border flag, intra-community type
  - Categorical encodings: invoice_type, hs_risk_group
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Benford constants
# ---------------------------------------------------------------------------
BENFORD_EXPECTED: dict[int, float] = {d: math.log10(1 + 1 / d) for d in range(1, 10)}
HIGH_RISK_HS_PREFIXES = {"8517", "7108", "8471", "8541", "8542"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leading_digit(value: float) -> int:
    if value <= 0:
        return -1
    while value >= 10:
        value /= 10
    while value < 1:
        value *= 10
    return int(value)


def _benford_deviation(value: float) -> float:
    """Deviation of a single amount's leading digit from Benford expectation."""
    d = _leading_digit(value)
    if d < 1:
        return 0.0
    expected = BENFORD_EXPECTED[d]
    return abs(1 / 9 - expected)  # Uniform freq vs Benford expected


def _is_round(value: float, divisor: float = 100.0) -> int:
    return int(value % divisor == 0)


def _hs_risk(hs_code: str) -> int:
    """1 if the HS code belongs to a high-risk commodity group, else 0."""
    prefix = str(hs_code)[:4] if hs_code else ""
    return int(prefix in HIGH_RISK_HS_PREFIXES)


def _invoice_type_code(invoice_type: str) -> int:
    mapping = {
        "standard": 0,
        "export": 1,
        "import": 2,
        "intra_community": 3,
        "credit_note": 4,
        "proforma": 5,
    }
    return mapping.get(str(invoice_type).lower(), 0)


# ---------------------------------------------------------------------------
# Per-seller aggregate features (requires full dataset)
# ---------------------------------------------------------------------------

def build_seller_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute seller-level aggregate statistics across all their invoices.
    Returns a DataFrame indexed by seller_tin with aggregate columns.
    """
    agg = df.groupby("seller_tin").agg(
        seller_total_invoices=("invoice_id", "count"),
        seller_mean_net=("net_amount", "mean"),
        seller_std_net=("net_amount", "std"),
        seller_round_ratio=("net_amount", lambda x: (x % 100 == 0).mean()),
        seller_buyer_count=("buyer_tin", "nunique"),
        seller_cross_border_ratio=("is_cross_border", "mean"),
        seller_export_ratio=("is_export", "mean"),
        seller_missing_seller_decl_ratio=("declared_by_seller", lambda x: (x == 0).mean()),
    ).reset_index()

    agg["seller_std_net"] = agg["seller_std_net"].fillna(0)

    # Benford MAD per seller
    def _seller_benford_mad(amounts: pd.Series) -> float:
        vals = amounts[amounts > 0]
        if len(vals) < 5:
            return 0.0
        counts = {d: 0 for d in range(1, 10)}
        for v in vals:
            d = _leading_digit(float(v))
            if 1 <= d <= 9:
                counts[d] += 1
        n = sum(counts.values())
        if n == 0:
            return 0.0
        mad = sum(abs(counts[d] / n - BENFORD_EXPECTED[d]) for d in range(1, 10)) / 9
        return mad

    benford_mad = df.groupby("seller_tin")["net_amount"].apply(_seller_benford_mad).reset_index()
    benford_mad.columns = ["seller_tin", "seller_benford_mad"]
    agg = agg.merge(benford_mad, on="seller_tin", how="left")

    return agg.set_index("seller_tin")


# ---------------------------------------------------------------------------
# Main feature engineering function
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a raw invoice DataFrame into an ML feature matrix.

    Parameters
    ----------
    df : DataFrame with columns produced by data_generator.generate_dataset()

    Returns
    -------
    feature_df : DataFrame with numeric feature columns only (no labels).
                 The original index is preserved.
    """
    feat = pd.DataFrame(index=df.index)

    # --- Amount features ---
    feat["log_net_amount"] = np.log1p(df["net_amount"].clip(lower=0))
    feat["log_vat_amount"] = np.log1p(df["vat_amount"].clip(lower=0))
    feat["vat_rate"] = df["vat_rate"]
    feat["is_round_100"] = df["net_amount"].apply(lambda x: _is_round(x, 100))
    feat["is_round_1000"] = df["net_amount"].apply(lambda x: _is_round(x, 1000))
    feat["benford_deviation"] = df["net_amount"].apply(_benford_deviation)

    # Declared VAT rate vs implied rate
    implied_rate = np.where(
        df["net_amount"] > 0,
        (df["vat_amount"] / df["net_amount"].clip(lower=1e-9)) * 100,
        0.0,
    )
    feat["vat_rate_discrepancy"] = np.abs(implied_rate - df["vat_rate"])

    # --- Declaration features ---
    feat["declared_by_seller"] = df["declared_by_seller"]
    feat["declared_by_buyer"] = df["declared_by_buyer"]
    # Both declared vs only one
    feat["declaration_mismatch"] = ((df["declared_by_seller"] == 0) | (df["declared_by_buyer"] == 0)).astype(int)
    feat["buyer_only"] = ((df["declared_by_seller"] == 0) & (df["declared_by_buyer"] == 1)).astype(int)
    feat["seller_only"] = ((df["declared_by_seller"] == 1) & (df["declared_by_buyer"] == 0)).astype(int)

    # --- Customs / documentation ---
    feat["has_customs_declaration"] = df["has_customs_declaration"]
    feat["export_no_customs"] = (
        (df["is_export"] == 1) & (df["has_customs_declaration"] == 0)
    ).astype(int)

    # --- Temporal / compliance features ---
    feat["payment_days"] = df["payment_days"].clip(lower=0, upper=365)
    feat["amendment_count"] = df["amendment_count"]
    feat["log_seller_age_days"] = np.log1p(df["seller_days_since_registration"].clip(lower=0))
    feat["seller_consecutive_late_filings"] = df["seller_consecutive_late_filings"].clip(lower=0)
    feat["seller_filing_gap_days"] = df["seller_filing_gap_days"].clip(lower=0)
    feat["new_trader"] = (df["seller_days_since_registration"] < 90).astype(int)

    # --- Cross-border & transaction type ---
    feat["is_export"] = df["is_export"]
    feat["is_cross_border"] = df["is_cross_border"]
    feat["buyer_vat_registered"] = df["buyer_vat_registered"]
    feat["invoice_type_code"] = df["invoice_type"].apply(_invoice_type_code)
    feat["hs_risk"] = df["hs_code"].apply(_hs_risk)

    # --- Seller-level aggregates ---
    seller_agg = build_seller_aggregates(df)
    # Map seller_tin → aggregate row index
    tins = df["seller_tin"].values
    agg_aligned = seller_agg.reindex(tins).reset_index(drop=True)
    agg_aligned.index = feat.index
    feat = pd.concat([feat, agg_aligned], axis=1)

    # Fill NaN from aggregates (sellers appearing only once)
    agg_cols = list(seller_agg.columns)
    feat[agg_cols] = feat[agg_cols].fillna(0)

    # Keep only numeric columns (drop any TIN strings that leaked in)
    feat = feat.select_dtypes(include=[np.number])

    return feat.astype(np.float32)


def get_feature_names() -> list[str]:
    """Return ordered list of feature column names (matches engineer_features output)."""
    return [
        "log_net_amount", "log_vat_amount", "vat_rate",
        "is_round_100", "is_round_1000", "benford_deviation",
        "vat_rate_discrepancy",
        "declared_by_seller", "declared_by_buyer",
        "declaration_mismatch", "buyer_only", "seller_only",
        "has_customs_declaration", "export_no_customs",
        "payment_days", "amendment_count",
        "log_seller_age_days", "seller_consecutive_late_filings", "seller_filing_gap_days",
        "new_trader",
        "is_export", "is_cross_border", "buyer_vat_registered",
        "invoice_type_code", "hs_risk",
        # seller aggregates
        "seller_total_invoices", "seller_mean_net", "seller_std_net",
        "seller_round_ratio", "seller_buyer_count",
        "seller_cross_border_ratio", "seller_export_ratio",
        "seller_missing_seller_decl_ratio", "seller_benford_mad",
    ]

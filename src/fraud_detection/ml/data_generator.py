"""
Synthetic VAT Fraud Data Generator
====================================
Generates labelled electronic invoice datasets covering all five primary
VAT fraud archetypes identified in IMF WP/07/31 (Keen & Smith 2007) and
IMF HTN 2023/001 (Andrew & Baer 2023):

  1. Missing Trader Intra-Community (MTIC) / Carousel fraud
  2. Fake Invoice / Invoice Mill fraud
  3. VAT Refund / Phantom Exporter fraud
  4. Contra-Trading fraud
  5. Suppressed Sales / Under-declaration fraud

Each row in the output DataFrame represents a single invoice transaction
with engineered features and a binary `is_fraud` label plus a `fraud_type`
label for multi-class evaluation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Seeds for reproducibility
# ---------------------------------------------------------------------------
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FRAUD_TYPES = [
    "missing_trader",
    "carousel",
    "invoice_mill",
    "refund_fraud",
    "contra_trading",
    "suppressed_sales",
    "legitimate",
]

HIGH_RISK_HS_CODES = ["8517", "7108", "8471", "8541", "8542"]
STANDARD_HS_CODES = ["6204", "8418", "9401", "3004", "4901", "8432", "2710"]
EU_COUNTRIES = ["DE", "FR", "IT", "ES", "NL", "BE", "PL", "SE", "AT", "PT"]
SECTORS = ["technology", "manufacturing", "retail", "wholesale", "services", "agriculture", "construction"]
VAT_RATES = [0.0, 5.0, 10.0, 20.0, 23.0, 25.0]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FraudScenario:
    fraud_type: str
    n_invoices: int
    tins: list[str]
    invoices: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _tin(prefix: str, n: int) -> str:
    return f"{prefix}{n:06d}"


def _random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def _benford_amount(magnitude: float = 1.0) -> float:
    """Generate an amount whose leading digit follows Benford's Law."""
    digits = list(range(1, 10))
    weights = [np.log10(1 + 1 / d) for d in digits]
    leading = random.choices(digits, weights=weights)[0]
    rest = random.uniform(0, 1)
    return round((leading + rest) * magnitude, 2)


def _round_amount(low: float = 500.0, high: float = 50_000.0) -> float:
    """Generate suspiciously round amounts (multiples of 100 or 1000)."""
    base = random.choice([100, 500, 1000, 5000, 10000])
    n = random.randint(int(low / base), int(high / base))
    return float(base * n)


def _uniform_leading_digit_amount(low: float = 1000.0, high: float = 9999.0) -> float:
    """Amounts with artificially uniform leading digits (Benford violation)."""
    leading = random.randint(5, 6)  # Overrepresent 5 and 6
    rest = random.uniform(0, 1)
    magnitude = 10 ** random.randint(2, 4)
    return round((leading + rest) * magnitude, 2)


def _vat_rate_for_hs(hs_code: str) -> float:
    if hs_code in HIGH_RISK_HS_CODES:
        return 20.0
    return random.choice([5.0, 10.0, 20.0, 23.0])


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------

def _generate_legitimate(
    n: int,
    start: datetime,
    end: datetime,
    tin_offset: int = 0,
) -> list[dict]:
    rows = []
    for i in range(n):
        s_tin = _tin("LEGIT_S", tin_offset + i % 200)
        b_tin = _tin("LEGIT_B", tin_offset + (i + 50) % 200)
        hs = random.choice(STANDARD_HS_CODES)
        vat_rate = _vat_rate_for_hs(hs)
        net = _benford_amount(random.choice([100, 500, 1000, 5000, 10000]))
        vat = round(net * vat_rate / 100, 2)
        invoice_date = _random_date(start, end)
        rows.append({
            "invoice_id": f"INV-LEG-{tin_offset + i:07d}",
            "invoice_date": invoice_date,
            "seller_tin": s_tin,
            "buyer_tin": b_tin,
            "seller_country": "HOME",
            "buyer_country": "HOME",
            "invoice_type": "standard",
            "hs_code": hs,
            "net_amount": net,
            "vat_rate": vat_rate,
            "vat_amount": vat,
            "gross_amount": round(net + vat, 2),
            "declared_by_seller": True,
            "declared_by_buyer": True,
            "customs_declaration": None,
            "payment_days": random.randint(10, 60),
            "amendment_count": 0,
            "is_export": False,
            "is_cross_border": False,
            "buyer_vat_registered": True,
            "seller_days_since_registration": random.randint(365, 3650),
            "seller_consecutive_late_filings": 0,
            "seller_filing_gap_days": random.randint(0, 30),
            "fraud_type": "legitimate",
            "is_fraud": 0,
        })
    return rows


def _generate_missing_trader(
    n: int,
    start: datetime,
    end: datetime,
    tin_offset: int = 0,
) -> list[dict]:
    """
    Missing Trader: newly registered importer acquires goods VAT-free,
    sells with VAT, collects VAT from buyer but disappears without remitting.
    Buyer then claims the input VAT credit.
    """
    rows = []
    for i in range(n):
        # The missing trader is newly registered (<90 days)
        missing_tin = _tin("MT", tin_offset + i % 30)
        buyer_tin = _tin("MT_BUYER", tin_offset + i % 50)
        hs = random.choice(HIGH_RISK_HS_CODES)
        vat_rate = 20.0
        net = _benford_amount(random.choice([5000, 10000, 50000]))
        vat = round(net * vat_rate / 100, 2)
        invoice_date = _random_date(start, end)
        rows.append({
            "invoice_id": f"INV-MT-{tin_offset + i:07d}",
            "invoice_date": invoice_date,
            "seller_tin": missing_tin,
            "buyer_tin": buyer_tin,
            "seller_country": "HOME",
            "buyer_country": "HOME",
            "invoice_type": "standard",
            "hs_code": hs,
            "net_amount": net,
            "vat_rate": vat_rate,
            "vat_amount": vat,
            "gross_amount": round(net + vat, 2),
            "declared_by_seller": False,  # Seller vanishes — never declares
            "declared_by_buyer": True,    # Buyer claims input VAT
            "customs_declaration": None,
            "payment_days": random.randint(0, 5),  # Fast payment
            "amendment_count": 0,
            "is_export": False,
            "is_cross_border": True,      # Often involves intra-EU acquisition
            "buyer_vat_registered": True,
            "seller_days_since_registration": random.randint(1, 89),  # Newly registered
            "seller_consecutive_late_filings": random.randint(3, 12),
            "seller_filing_gap_days": random.randint(90, 365),
            "fraud_type": "missing_trader",
            "is_fraud": 1,
        })
    return rows


def _generate_carousel(
    n: int,
    start: datetime,
    end: datetime,
    tin_offset: int = 0,
) -> list[dict]:
    """
    Carousel fraud: goods circulate in a chain A→B→C→A.
    Each node reclaims input VAT while the 'missing trader' node never remits.
    Characterised by rapid resale and circular transaction graphs.
    """
    rows = []
    chain_size = random.randint(3, 6)
    chain_tins = [_tin("CAR", tin_offset + j) for j in range(chain_size)]

    for i in range(n):
        hop = i % chain_size
        s_tin = chain_tins[hop]
        b_tin = chain_tins[(hop + 1) % chain_size]
        hs = random.choice(HIGH_RISK_HS_CODES)
        vat_rate = 20.0
        net = _benford_amount(random.choice([10000, 50000, 100000]))
        vat = round(net * vat_rate / 100, 2)
        # Rapid resale: all invoices within a 24-hour window
        base_date = _random_date(start, end)
        invoice_date = base_date + timedelta(hours=hop * random.uniform(1, 6))
        rows.append({
            "invoice_id": f"INV-CAR-{tin_offset + i:07d}",
            "invoice_date": invoice_date,
            "seller_tin": s_tin,
            "buyer_tin": b_tin,
            "seller_country": random.choice(EU_COUNTRIES),
            "buyer_country": random.choice(EU_COUNTRIES),
            "invoice_type": "intra_community" if hop == 0 else "standard",
            "hs_code": hs,
            "net_amount": net,
            "vat_rate": vat_rate,
            "vat_amount": vat,
            "gross_amount": round(net + vat, 2),
            "declared_by_seller": hop != 0,  # Missing trader at hop 0
            "declared_by_buyer": True,
            "customs_declaration": f"CUST-{random.randint(100000, 999999)}" if hop == 0 else None,
            "payment_days": random.randint(0, 2),  # Very fast
            "amendment_count": 0,
            "is_export": hop == chain_size - 1,
            "is_cross_border": True,
            "buyer_vat_registered": True,
            "seller_days_since_registration": random.randint(1, 180) if hop == 0 else random.randint(180, 2000),
            "seller_consecutive_late_filings": random.randint(3, 8) if hop == 0 else 0,
            "seller_filing_gap_days": random.randint(90, 365) if hop == 0 else random.randint(0, 30),
            "fraud_type": "carousel",
            "is_fraud": 1,
        })
    return rows


def _generate_invoice_mill(
    n: int,
    start: datetime,
    end: datetime,
    tin_offset: int = 0,
) -> list[dict]:
    """
    Invoice Mill: fabricated invoices for fictitious transactions.
    Signals: round amounts, Benford violation, concentrated buyers,
    no purchase invoices to support, high daily volume.
    """
    rows = []
    mill_tin = _tin("MILL", tin_offset)
    # Concentrated buyers (≤3)
    buyer_tins = [_tin("MILL_B", tin_offset + j) for j in range(3)]

    daily_volume = random.randint(30, 80)  # High daily volume
    current_day = _random_date(start, end)

    for i in range(n):
        if i % daily_volume == 0 and i > 0:
            current_day += timedelta(days=1)
        hs = random.choice(STANDARD_HS_CODES)
        vat_rate = random.choice([10.0, 20.0])
        net = _round_amount(500, 20000)  # Suspiciously round
        if random.random() < 0.3:
            net = _uniform_leading_digit_amount()  # Benford violation
        vat = round(net * vat_rate / 100, 2)
        rows.append({
            "invoice_id": f"INV-MILL-{tin_offset + i:07d}",
            "invoice_date": current_day + timedelta(hours=random.uniform(0, 8)),
            "seller_tin": mill_tin,
            "buyer_tin": random.choice(buyer_tins),
            "seller_country": "HOME",
            "buyer_country": "HOME",
            "invoice_type": "standard",
            "hs_code": hs,
            "net_amount": net,
            "vat_rate": vat_rate,
            "vat_amount": vat,
            "gross_amount": round(net + vat, 2),
            "declared_by_seller": True,
            "declared_by_buyer": True,
            "customs_declaration": None,
            "payment_days": random.randint(0, 3),
            "amendment_count": 0,
            "is_export": False,
            "is_cross_border": False,
            "buyer_vat_registered": True,
            "seller_days_since_registration": random.randint(30, 365),
            "seller_consecutive_late_filings": random.randint(0, 2),
            "seller_filing_gap_days": random.randint(0, 15),
            "fraud_type": "invoice_mill",
            "is_fraud": 1,
        })
    return rows


def _generate_refund_fraud(
    n: int,
    start: datetime,
    end: datetime,
    tin_offset: int = 0,
) -> list[dict]:
    """
    VAT Refund / Phantom Exporter fraud:
    - Exports declared but no matching customs documentation.
    - Refund-to-output ratio > 80%.
    - Large sudden spike in refund claims.
    """
    rows = []
    for i in range(n):
        exporter_tin = _tin("EXP", tin_offset + i % 20)
        hs = random.choice(STANDARD_HS_CODES)
        vat_rate = 0.0  # Exports are zero-rated
        net = _benford_amount(random.choice([50000, 100000, 500000]))
        # Phantom: no real export, so no customs doc
        has_customs = random.random() < 0.1
        invoice_date = _random_date(start, end)
        rows.append({
            "invoice_id": f"INV-REF-{tin_offset + i:07d}",
            "invoice_date": invoice_date,
            "seller_tin": exporter_tin,
            "buyer_tin": _tin("FOREIGN_B", i % 30),
            "seller_country": "HOME",
            "buyer_country": random.choice(EU_COUNTRIES),
            "invoice_type": "export",
            "hs_code": hs,
            "net_amount": net,
            "vat_rate": vat_rate,
            "vat_amount": 0.0,
            "gross_amount": net,
            "declared_by_seller": True,
            "declared_by_buyer": False,  # Foreign buyer — no domestic declaration
            "customs_declaration": f"CUST-{random.randint(100000, 999999)}" if has_customs else None,
            "payment_days": random.randint(0, 180),
            "amendment_count": random.randint(0, 4),
            "is_export": True,
            "is_cross_border": True,
            "buyer_vat_registered": False,
            "seller_days_since_registration": random.randint(30, 730),
            "seller_consecutive_late_filings": random.randint(0, 3),
            "seller_filing_gap_days": random.randint(0, 60),
            "fraud_type": "refund_fraud",
            "is_fraud": 1,
        })
    return rows


def _generate_contra_trading(
    n: int,
    start: datetime,
    end: datetime,
    tin_offset: int = 0,
) -> list[dict]:
    """
    Contra-Trading: orchestrator maintains near-zero net VAT by offsetting
    fraudulent input tax claims against output tax — essentially laundering
    a missing trader's VAT liability across a parallel supply chain.
    Signals: near-zero net VAT, diverse unrelated HS codes, mirrored chains.
    """
    rows = []
    orchestrator_tin = _tin("CONTR", tin_offset)
    sectors_used = random.sample(STANDARD_HS_CODES, min(5, len(STANDARD_HS_CODES)))

    for i in range(n):
        # Alternating buy/sell to net out VAT
        is_purchase = (i % 2 == 0)
        hs = sectors_used[i % len(sectors_used)]
        vat_rate = 20.0
        net = _benford_amount(random.choice([10000, 50000]))
        vat = round(net * vat_rate / 100, 2)
        invoice_date = _random_date(start, end)
        rows.append({
            "invoice_id": f"INV-CON-{tin_offset + i:07d}",
            "invoice_date": invoice_date,
            "seller_tin": _tin("CONTR_S", i % 10) if is_purchase else orchestrator_tin,
            "buyer_tin": orchestrator_tin if is_purchase else _tin("CONTR_B", i % 10),
            "seller_country": "HOME",
            "buyer_country": "HOME",
            "invoice_type": "standard",
            "hs_code": hs,
            "net_amount": net,
            "vat_rate": vat_rate,
            "vat_amount": vat,
            "gross_amount": round(net + vat, 2),
            "declared_by_seller": True,
            "declared_by_buyer": True,
            "customs_declaration": None,
            "payment_days": random.randint(0, 7),
            "amendment_count": 0,
            "is_export": False,
            "is_cross_border": False,
            "buyer_vat_registered": True,
            "seller_days_since_registration": random.randint(180, 2000),
            "seller_consecutive_late_filings": 0,
            "seller_filing_gap_days": random.randint(0, 20),
            "fraud_type": "contra_trading",
            "is_fraud": 1,
        })
    return rows


def _generate_suppressed_sales(
    n: int,
    start: datetime,
    end: datetime,
    tin_offset: int = 0,
) -> list[dict]:
    """
    Suppressed Sales / Under-declaration:
    - Seller omits cash or B2C sales from VAT return.
    - Buyer invoice exists but no corresponding seller declaration.
    - Often seen in retail / hospitality.
    """
    rows = []
    for i in range(n):
        seller_tin = _tin("SUPP_S", tin_offset + i % 40)
        hs = random.choice(STANDARD_HS_CODES)
        vat_rate = random.choice([5.0, 10.0, 20.0])
        net = _benford_amount(random.choice([50, 100, 500, 1000]))
        vat = round(net * vat_rate / 100, 2)
        invoice_date = _random_date(start, end)
        rows.append({
            "invoice_id": f"INV-SUPP-{tin_offset + i:07d}",
            "invoice_date": invoice_date,
            "seller_tin": seller_tin,
            "buyer_tin": _tin("SUPP_B", tin_offset + i % 100),
            "seller_country": "HOME",
            "buyer_country": "HOME",
            "invoice_type": "standard",
            "hs_code": hs,
            "net_amount": net,
            "vat_rate": vat_rate,
            "vat_amount": vat,
            "gross_amount": round(net + vat, 2),
            "declared_by_seller": False,  # Seller suppresses the sale
            "declared_by_buyer": random.random() < 0.4,
            "customs_declaration": None,
            "payment_days": random.randint(0, 5),
            "amendment_count": 0,
            "is_export": False,
            "is_cross_border": False,
            "buyer_vat_registered": random.random() < 0.5,
            "seller_days_since_registration": random.randint(365, 5000),
            "seller_consecutive_late_filings": random.randint(0, 5),
            "seller_filing_gap_days": random.randint(0, 90),
            "fraud_type": "suppressed_sales",
            "is_fraud": 1,
        })
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dataset(
    n_legitimate: int = 5000,
    n_missing_trader: int = 600,
    n_carousel: int = 700,
    n_invoice_mill: int = 800,
    n_refund_fraud: int = 600,
    n_contra_trading: int = 500,
    n_suppressed_sales: int = 800,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic labelled VAT fraud dataset.

    Returns a DataFrame with one row per invoice and columns:
        invoice_id, invoice_date, seller_tin, buyer_tin, seller_country,
        buyer_country, invoice_type, hs_code, net_amount, vat_rate,
        vat_amount, gross_amount, declared_by_seller, declared_by_buyer,
        customs_declaration, payment_days, amendment_count, is_export,
        is_cross_border, buyer_vat_registered, seller_days_since_registration,
        seller_consecutive_late_filings, seller_filing_gap_days,
        fraud_type, is_fraud
    """
    if start_date is None:
        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    if end_date is None:
        end_date = datetime(2024, 12, 31, tzinfo=timezone.utc)

    all_rows: list[dict] = []
    offset = 0

    all_rows += _generate_legitimate(n_legitimate, start_date, end_date, offset)
    offset += n_legitimate

    all_rows += _generate_missing_trader(n_missing_trader, start_date, end_date, offset)
    offset += n_missing_trader

    all_rows += _generate_carousel(n_carousel, start_date, end_date, offset)
    offset += n_carousel

    all_rows += _generate_invoice_mill(n_invoice_mill, start_date, end_date, offset)
    offset += n_invoice_mill

    all_rows += _generate_refund_fraud(n_refund_fraud, start_date, end_date, offset)
    offset += n_refund_fraud

    all_rows += _generate_contra_trading(n_contra_trading, start_date, end_date, offset)
    offset += n_contra_trading

    all_rows += _generate_suppressed_sales(n_suppressed_sales, start_date, end_date, offset)

    df = pd.DataFrame(all_rows)

    # Shuffle rows
    df = df.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)

    # Boolean → int for declared columns
    df["declared_by_seller"] = df["declared_by_seller"].astype(int)
    df["declared_by_buyer"] = df["declared_by_buyer"].astype(int)
    df["is_export"] = df["is_export"].astype(int)
    df["is_cross_border"] = df["is_cross_border"].astype(int)
    df["buyer_vat_registered"] = df["buyer_vat_registered"].astype(int)
    df["has_customs_declaration"] = (df["customs_declaration"].notna()).astype(int)

    return df

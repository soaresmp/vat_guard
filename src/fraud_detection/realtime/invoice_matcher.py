"""
Real-Time Invoice Matcher
==========================
Matches seller-declared invoices against buyer-declared invoices to detect:
  1. Invoices claimed as input VAT by buyer but never declared by seller
  2. Invoices declared by seller but not claimed by any buyer (suppressed sales)
  3. Discrepancies in amounts between seller and buyer declarations
  4. Duplicate invoice submissions

This mirrors the approach used by South Korea, Portugal, and other countries
with successful real-time e-invoicing systems (IMF 2023 How-To Note).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MatchStatus(str, Enum):
    MATCHED = "matched"
    SELLER_ONLY = "seller_only"      # Declared by seller, not by buyer
    BUYER_ONLY = "buyer_only"        # Claimed by buyer, missing from seller
    AMOUNT_MISMATCH = "amount_mismatch"
    DUPLICATE = "duplicate"
    PARTIAL_MATCH = "partial_match"


@dataclass
class MatchResult:
    status: MatchStatus
    seller_tin: str
    buyer_tin: str | None
    invoice_number: str
    seller_vat_amount: float
    buyer_vat_amount: float | None
    discrepancy: float              # Absolute difference
    discrepancy_pct: float          # Relative difference
    risk_score: float
    flags: list[str] = field(default_factory=list)


class InvoiceMatcher:
    """
    Matches invoices from seller and buyer declarations and assigns risk scores.
    Designed for high-throughput real-time processing of e-invoice streams.
    """

    AMOUNT_TOLERANCE = 0.01          # 1% tolerance for rounding differences
    HIGH_VALUE_THRESHOLD = 10_000    # Invoices above this get extra scrutiny

    def match_period(
        self,
        seller_invoices: list[Any],
        buyer_invoices: list[Any],
    ) -> list[MatchResult]:
        """
        Match all invoices for a given filing period.

        Args:
            seller_invoices: Invoices declared in seller's output VAT return.
            buyer_invoices: Invoices claimed in buyer's input VAT return.

        Returns:
            List of MatchResult for all invoice pairs and unmatched invoices.
        """
        # Index seller invoices by (invoice_number, seller_vat_number)
        seller_index: dict[str, Any] = {}
        for inv in seller_invoices:
            key = self._make_key(inv)
            if key in seller_index:
                # Duplicate from seller
                existing = seller_index[key]
                existing._duplicate = True
            else:
                inv._duplicate = False
                seller_index[key] = inv

        # Index buyer invoices by (invoice_number, seller_vat_number)
        buyer_index: dict[str, Any] = {}
        for inv in buyer_invoices:
            key = self._make_key(inv)
            if key in buyer_index:
                inv._duplicate = True
            else:
                inv._duplicate = False
                buyer_index[key] = inv

        results: list[MatchResult] = []
        processed_keys: set[str] = set()

        # ── Match seller invoices ──────────────────────────────────────────
        for key, s_inv in seller_index.items():
            processed_keys.add(key)
            b_inv = buyer_index.get(key)
            result = self._compare(s_inv, b_inv)
            results.append(result)

        # ── Unmatched buyer invoices (phantom input VAT claims) ────────────
        for key, b_inv in buyer_index.items():
            if key not in processed_keys:
                result = self._compare(None, b_inv)
                results.append(result)

        return results

    def match_single(self, seller_inv: Any | None, buyer_inv: Any | None) -> MatchResult:
        """Match a single invoice pair in real time."""
        return self._compare(seller_inv, buyer_inv)

    def _compare(self, seller_inv: Any | None, buyer_inv: Any | None) -> MatchResult:
        """Core matching logic for a single invoice pair."""
        flags: list[str] = []

        # Extract values
        s_vat = float(getattr(seller_inv, "vat_amount", 0) or 0) if seller_inv else 0.0
        b_vat = float(getattr(buyer_inv, "vat_amount", 0) or 0) if buyer_inv else 0.0
        s_tin = getattr(seller_inv, "seller_tin", "") if seller_inv else ""
        b_tin = getattr(buyer_inv, "buyer_tin", None) if buyer_inv else None
        inv_num = (
            getattr(seller_inv, "invoice_number", "")
            or getattr(buyer_inv, "invoice_number", "")
        )
        s_vat_num = (
            getattr(seller_inv, "seller_vat_number", "") if seller_inv else
            getattr(buyer_inv, "seller_vat_number", "") if buyer_inv else ""
        )

        # ── Status determination ───────────────────────────────────────────
        if seller_inv and buyer_inv:
            discrepancy = abs(s_vat - b_vat)
            discrepancy_pct = discrepancy / max(s_vat, 1.0)
            if discrepancy_pct > self.AMOUNT_TOLERANCE:
                status = MatchStatus.AMOUNT_MISMATCH
                flags.append(f"VAT discrepancy: seller={s_vat:.2f}, buyer={b_vat:.2f}")
            elif getattr(seller_inv, "_duplicate", False) is True or getattr(buyer_inv, "_duplicate", False) is True:
                status = MatchStatus.DUPLICATE
                flags.append("Duplicate invoice submission detected")
            else:
                status = MatchStatus.MATCHED
        elif seller_inv and not buyer_inv:
            discrepancy = s_vat
            discrepancy_pct = 1.0
            status = MatchStatus.SELLER_ONLY
            flags.append("Invoice declared by seller but not claimed by any buyer")
        elif buyer_inv and not seller_inv:
            discrepancy = b_vat
            discrepancy_pct = 1.0
            status = MatchStatus.BUYER_ONLY
            flags.append("Input VAT claimed by buyer but seller has no matching declaration")
        else:
            discrepancy = 0.0
            discrepancy_pct = 0.0
            status = MatchStatus.MATCHED

        # ── High-value flag ────────────────────────────────────────────────
        max_vat = max(s_vat, b_vat)
        if max_vat >= self.HIGH_VALUE_THRESHOLD:
            flags.append(f"High-value invoice: VAT={max_vat:.2f}")

        risk_score = self._compute_risk(status, discrepancy, discrepancy_pct, max_vat)

        return MatchResult(
            status=status,
            seller_tin=s_tin,
            buyer_tin=b_tin,
            invoice_number=inv_num,
            seller_vat_amount=s_vat,
            buyer_vat_amount=b_vat if buyer_inv else None,
            discrepancy=round(discrepancy, 2),
            discrepancy_pct=round(discrepancy_pct, 4),
            risk_score=round(risk_score, 2),
            flags=flags,
        )

    def _compute_risk(
        self,
        status: MatchStatus,
        discrepancy: float,
        discrepancy_pct: float,
        vat_amount: float,
    ) -> float:
        if status == MatchStatus.MATCHED:
            return 0.0
        base = {
            MatchStatus.MATCHED: 0.0,
            MatchStatus.SELLER_ONLY: 20.0,
            MatchStatus.BUYER_ONLY: 70.0,   # Most suspicious — phantom input VAT
            MatchStatus.AMOUNT_MISMATCH: 50.0,
            MatchStatus.DUPLICATE: 60.0,
            MatchStatus.PARTIAL_MATCH: 35.0,
        }.get(status, 0.0)

        # Scale by value
        value_addon = min(20.0, vat_amount / 5_000)
        discrepancy_addon = min(10.0, discrepancy_pct * 20)

        return min(100.0, base + value_addon + discrepancy_addon)

    @staticmethod
    def _make_key(invoice: Any) -> str:
        """Create a unique match key from invoice number and seller VAT number."""
        inv_num = getattr(invoice, "invoice_number", "")
        seller_vat = getattr(invoice, "seller_vat_number", "")
        return f"{seller_vat}:{inv_num}"

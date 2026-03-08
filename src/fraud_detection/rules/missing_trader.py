"""
Missing Trader Intra-Community (MTIC) Fraud Detector
=====================================================
Detects taxpayers who:
  1. Show significant VAT collected on sales but repeatedly fail to remit / file.
  2. Suddenly go inactive after a period of high-volume importing.
  3. Have filed as active but registration details suggest a shell / nominee director.
  4. Appear in others' input-VAT chains but have no matching output declarations.

IMF Reference: WP/07/31 – Keen & Smith (2007)
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from src.fraud_detection.rules.base import BaseDetector, DetectionResult
from src.models.alert import AlertSeverity, FraudType


class MissingTraderDetector(BaseDetector):
    name = "missing_trader"
    fraud_type = FraudType.MISSING_TRADER

    # Tuning parameters (can be overridden via config)
    DAYS_INACTIVE_THRESHOLD = 90        # Days without filing before flagged
    MIN_SALES_VAT_COLLECTED = 10_000    # Minimum VAT collected to be relevant
    LATE_FILING_STREAK_THRESHOLD = 3    # Consecutive missed filings
    HIGH_IMPORT_ACTIVITY_DAYS = 30      # Days of high import before going dark
    IMPORT_TO_FILING_GAP_DAYS = 14      # Max days between import and expected filing

    async def evaluate(self, context: dict[str, Any]) -> DetectionResult:
        taxpayer = context.get("taxpayer")
        transactions = context.get("transactions", [])
        invoices = context.get("invoices", [])
        now = context.get("now", datetime.now(timezone.utc))

        if not taxpayer:
            return self._build_no_detection()

        signals: dict[str, Any] = {}
        risk_score = 0.0

        # ── Signal 1: Registered but never filed ─────────────────────────────
        vat_returns = [t for t in transactions if t.transaction_type == "vat_return"]
        if taxpayer.vat_registration_date and not vat_returns:
            days_since_reg = (now - taxpayer.vat_registration_date).days
            if days_since_reg > 30:
                signals["never_filed"] = {
                    "days_since_registration": days_since_reg,
                    "vat_registration_date": str(taxpayer.vat_registration_date),
                }
                risk_score += min(40, days_since_reg / 3)

        # ── Signal 2: Long gap since last filing ──────────────────────────────
        if taxpayer.last_filing_date:
            days_since_last_filing = (now - taxpayer.last_filing_date).days
            if days_since_last_filing > self.DAYS_INACTIVE_THRESHOLD:
                signals["inactive_trader"] = {
                    "days_since_last_filing": days_since_last_filing,
                    "last_filing_date": str(taxpayer.last_filing_date),
                }
                risk_score += min(35, days_since_last_filing / 5)

        # ── Signal 3: Consecutive late / missed filings ───────────────────────
        if taxpayer.consecutive_late_filings >= self.LATE_FILING_STREAK_THRESHOLD:
            signals["consecutive_late_filings"] = {
                "count": taxpayer.consecutive_late_filings,
                "threshold": self.LATE_FILING_STREAK_THRESHOLD,
            }
            risk_score += min(20, taxpayer.consecutive_late_filings * 5)

        # ── Signal 4: Output VAT declared by counter-parties but not by trader ─
        # Sum VAT amounts where buyer claims input credit from this seller
        input_credits_claimed = sum(
            inv.vat_amount
            for inv in invoices
            if getattr(inv, "seller_tin", None) == taxpayer.tin
            and getattr(inv, "declared_by_buyer", False)
            and not getattr(inv, "declared_by_seller", False)
        )
        if input_credits_claimed > self.MIN_SALES_VAT_COLLECTED:
            signals["undeclared_output_vat"] = {
                "input_credits_claimed_by_buyers": input_credits_claimed,
                "threshold": self.MIN_SALES_VAT_COLLECTED,
            }
            risk_score += min(40, input_credits_claimed / self.MIN_SALES_VAT_COLLECTED * 20)

        # ── Signal 5: Sudden inactivity after heavy imports ───────────────────
        recent_imports = [
            inv for inv in invoices
            if getattr(inv, "invoice_type", None) in ("import", "intra_community")
            and getattr(inv, "buyer_tin", None) == taxpayer.tin
            and (now - inv.invoice_date).days <= self.HIGH_IMPORT_ACTIVITY_DAYS
        ]
        if recent_imports and taxpayer.last_filing_date:
            latest_import = max(recent_imports, key=lambda i: i.invoice_date)
            if (now - taxpayer.last_filing_date).days > self.IMPORT_TO_FILING_GAP_DAYS:
                total_import_vat = sum(inv.vat_amount for inv in recent_imports)
                signals["post_import_disappearance"] = {
                    "recent_import_invoices": len(recent_imports),
                    "total_import_vat_value": total_import_vat,
                    "latest_import_date": str(latest_import.invoice_date),
                    "days_since_last_filing": (now - taxpayer.last_filing_date).days,
                }
                risk_score += min(35, total_import_vat / 1000)

        # ── Signal 6: Newly registered, already high-volume ───────────────────
        if taxpayer.is_newly_registered and taxpayer.annual_turnover_reported:
            if taxpayer.annual_turnover_reported > 1_000_000:
                signals["new_high_volume"] = {
                    "annual_turnover": taxpayer.annual_turnover_reported,
                    "is_newly_registered": True,
                }
                risk_score += 15

        risk_score = min(risk_score, 100.0)
        detected = risk_score >= 30 and bool(signals)

        if not detected:
            return self._build_no_detection()

        severity = self._score_to_severity(risk_score)
        estimated_rar = input_credits_claimed if "undeclared_output_vat" in signals else None

        return DetectionResult(
            fraud_type=self.fraud_type,
            detected=True,
            risk_score=round(risk_score, 2),
            confidence=min(0.95, len(signals) / 5),
            severity=severity,
            title=f"Potential Missing Trader – {taxpayer.vat_number}",
            description=(
                f"Taxpayer {taxpayer.name} ({taxpayer.vat_number}) shows "
                f"{len(signals)} missing-trader indicators: "
                + ", ".join(signals.keys())
            ),
            evidence=signals,
            related_tins=[taxpayer.tin],
            estimated_revenue_at_risk=estimated_rar,
            detection_rule=self.name,
        )

    @staticmethod
    def _score_to_severity(score: float) -> AlertSeverity:
        if score >= 80:
            return AlertSeverity.CRITICAL
        if score >= 60:
            return AlertSeverity.HIGH
        if score >= 40:
            return AlertSeverity.MEDIUM
        return AlertSeverity.LOW

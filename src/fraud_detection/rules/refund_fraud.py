"""
VAT Refund Fraud Detector
==========================
Detects false or inflated VAT refund claims, a major fraud vector highlighted
by the IMF 2023 How-To Note (Cedric Andrew & Katherine Baer).

Patterns detected:
  1. Refund-to-turnover ratio far exceeding industry norms
  2. Persistent large refund claimants with minimal domestic output VAT
  3. Sudden spike in refund claims correlated with zero-rated export invoices
     that lack matching customs declarations
  4. Refund claims against invoices from known/suspected invoice mills
  5. Repeated amendments to returns to increase refund amount
  6. Phantom exporter: claims export zero-rating without customs evidence

IMF Reference: "How to Combat Value-Added Tax Refund Fraud" (2023/001)
"""

from typing import Any

from src.fraud_detection.rules.base import BaseDetector, DetectionResult
from src.models.alert import AlertSeverity, FraudType


class RefundFraudDetector(BaseDetector):
    name = "refund_fraud"
    fraud_type = FraudType.REFUND_FRAUD

    REFUND_TO_OUTPUT_RATIO_THRESHOLD = 0.8    # Refund > 80% of output VAT
    REFUND_TO_TURNOVER_THRESHOLD = 0.25       # Refund > 25% of reported turnover
    EXPORT_WITHOUT_CUSTOMS_THRESHOLD = 0.5   # 50% of exports lack customs docs
    AMENDMENT_FREQUENCY_THRESHOLD = 3        # > 3 amendments increasing refund
    SPIKE_MULTIPLIER = 3.0                    # Refund claim is 3× the period average

    async def evaluate(self, context: dict[str, Any]) -> DetectionResult:
        taxpayer = context.get("taxpayer")
        transactions = context.get("transactions", [])
        invoices = context.get("invoices", [])

        if not taxpayer:
            return self._build_no_detection()

        signals: dict[str, Any] = {}
        risk_score = 0.0

        refund_transactions = [
            t for t in transactions
            if t.transaction_type in ("refund_claim", "amended_return")
        ]
        vat_returns = [t for t in transactions if t.transaction_type == "vat_return"]

        # ── Signal 1: Refund-to-output VAT ratio ──────────────────────────────
        total_output_vat = sum(
            getattr(t, "output_vat", 0) or 0 for t in vat_returns
        )
        total_refund_claimed = sum(
            getattr(t, "refund_claimed", 0) or 0 for t in refund_transactions
        )
        if total_output_vat > 0:
            ratio = total_refund_claimed / total_output_vat
            if ratio > self.REFUND_TO_OUTPUT_RATIO_THRESHOLD:
                signals["high_refund_to_output_ratio"] = {
                    "ratio": round(ratio, 3),
                    "total_refund_claimed": total_refund_claimed,
                    "total_output_vat": total_output_vat,
                    "threshold": self.REFUND_TO_OUTPUT_RATIO_THRESHOLD,
                }
                risk_score += min(40, ratio * 40)

        # ── Signal 2: Refund spike vs. historical average ──────────────────────
        if len(refund_transactions) >= 2:
            amounts = [
                getattr(t, "refund_claimed", 0) or 0
                for t in refund_transactions[:-1]
            ]
            if amounts:
                avg_refund = sum(amounts) / len(amounts)
                latest_refund = getattr(refund_transactions[-1], "refund_claimed", 0) or 0
                if avg_refund > 0 and latest_refund > avg_refund * self.SPIKE_MULTIPLIER:
                    signals["refund_spike"] = {
                        "latest_refund": latest_refund,
                        "historical_average": round(avg_refund, 2),
                        "spike_ratio": round(latest_refund / avg_refund, 2),
                        "threshold_multiplier": self.SPIKE_MULTIPLIER,
                    }
                    risk_score += min(30, (latest_refund / avg_refund) * 5)

        # ── Signal 3: Phantom exporter – exports without customs declarations ──
        export_invoices = [
            inv for inv in invoices
            if getattr(inv, "invoice_type", None) in ("export", "intra_community")
            and getattr(inv, "seller_tin", None) == taxpayer.tin
        ]
        if export_invoices:
            no_customs = [
                inv for inv in export_invoices
                if not getattr(inv, "customs_declaration_number", None)
            ]
            no_customs_ratio = len(no_customs) / len(export_invoices)
            if no_customs_ratio > self.EXPORT_WITHOUT_CUSTOMS_THRESHOLD:
                total_zero_rated_vat = sum(inv.net_amount for inv in no_customs)
                signals["phantom_exporter"] = {
                    "export_invoices_total": len(export_invoices),
                    "missing_customs_docs": len(no_customs),
                    "no_customs_ratio": round(no_customs_ratio, 3),
                    "zero_rated_net_amount_at_risk": total_zero_rated_vat,
                }
                risk_score += min(35, no_customs_ratio * 40)

        # ── Signal 4: Refund against invoices from flagged suppliers ──────────
        flagged_supplier_invoices = [
            inv for inv in invoices
            if getattr(inv, "buyer_tin", None) == taxpayer.tin
            and getattr(inv, "is_suspicious", False)
        ]
        if flagged_supplier_invoices:
            suspicious_input_vat = sum(inv.vat_amount for inv in flagged_supplier_invoices)
            signals["refund_from_flagged_suppliers"] = {
                "flagged_invoice_count": len(flagged_supplier_invoices),
                "suspicious_input_vat": suspicious_input_vat,
            }
            risk_score += min(30, len(flagged_supplier_invoices) * 5)

        # ── Signal 5: Frequent return amendments increasing refund ─────────────
        amendments = [
            t for t in transactions
            if t.transaction_type == "amended_return"
            and (getattr(t, "refund_claimed", 0) or 0) > 0
        ]
        if len(amendments) >= self.AMENDMENT_FREQUENCY_THRESHOLD:
            signals["frequent_amendments"] = {
                "amendment_count": len(amendments),
                "threshold": self.AMENDMENT_FREQUENCY_THRESHOLD,
                "total_additional_refund": sum(
                    getattr(a, "refund_claimed", 0) or 0 for a in amendments
                ),
            }
            risk_score += min(35, len(amendments) * 10)

        # ── Signal 6: Refund-to-turnover ratio ────────────────────────────────
        if taxpayer.annual_turnover_reported and taxpayer.annual_turnover_reported > 0:
            refund_turnover_ratio = total_refund_claimed / taxpayer.annual_turnover_reported
            if refund_turnover_ratio > self.REFUND_TO_TURNOVER_THRESHOLD:
                signals["high_refund_to_turnover"] = {
                    "ratio": round(refund_turnover_ratio, 3),
                    "total_refund_claimed": total_refund_claimed,
                    "annual_turnover": taxpayer.annual_turnover_reported,
                    "threshold": self.REFUND_TO_TURNOVER_THRESHOLD,
                }
                risk_score += min(20, refund_turnover_ratio * 40)

        risk_score = min(risk_score, 100.0)
        detected = risk_score >= 30 and bool(signals)

        if not detected:
            return self._build_no_detection()

        severity = self._score_to_severity(risk_score)

        return DetectionResult(
            fraud_type=self.fraud_type,
            detected=True,
            risk_score=round(risk_score, 2),
            confidence=min(0.9, len(signals) / 5),
            severity=severity,
            title=f"VAT Refund Fraud Indicators – {taxpayer.vat_number}",
            description=(
                f"Taxpayer {taxpayer.name} ({taxpayer.vat_number}) shows "
                f"{len(signals)} refund fraud indicators: {', '.join(signals.keys())}. "
                f"Total refund at risk: {total_refund_claimed:,.2f}"
            ),
            evidence=signals,
            related_tins=[taxpayer.tin],
            estimated_revenue_at_risk=total_refund_claimed,
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

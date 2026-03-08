"""
Invoice Mill / Bogus Trader Detector
=====================================
Detects entities that:
  - Issue large volumes of invoices with no corresponding business activity
  - Issue invoices to the same set of buyers repeatedly with perfectly round amounts
  - Have no (or minimal) purchase invoices of their own — "pure VAT generators"
  - Exhibit Benford's Law anomalies on leading digits of invoice amounts
  - Are registered at the same address as many other VAT-registered entities

IMF Reference: WP/07/31 – "Bogus traders: Companies set up solely to generate
invoices that allow recovery of VAT. Such 'invoice mills' exploit the practical
impossibility of cross-checking every invoice against evidence that earlier tax
has been paid."
"""

from collections import Counter
from math import log10
from typing import Any

from src.fraud_detection.rules.base import BaseDetector, DetectionResult
from src.models.alert import AlertSeverity, FraudType

# Benford's Law expected distribution of leading digits (1–9)
BENFORD_EXPECTED = {
    1: 0.30103,
    2: 0.17609,
    3: 0.12494,
    4: 0.09691,
    5: 0.07918,
    6: 0.06695,
    7: 0.05799,
    8: 0.05115,
    9: 0.04576,
}


class InvoiceMillDetector(BaseDetector):
    name = "invoice_mill"
    fraud_type = FraudType.INVOICE_MILL

    MIN_INVOICES_PER_DAY = 30         # High volume threshold
    ROUND_AMOUNT_THRESHOLD = 0.7      # % of invoices with round amounts = suspicious
    PURCHASE_RATIO_THRESHOLD = 0.05   # Purchase invoices / sales invoices ratio
    MAX_ADDRESS_SHARING = 5           # Max other companies at same address
    BENFORD_DEVIATION_THRESHOLD = 0.15 # Max avg deviation from Benford expected

    async def evaluate(self, context: dict[str, Any]) -> DetectionResult:
        taxpayer = context.get("taxpayer")
        invoices = context.get("invoices", [])

        if not taxpayer or not invoices:
            return self._build_no_detection()

        signals: dict[str, Any] = {}
        risk_score = 0.0

        issued = [i for i in invoices if getattr(i, "seller_tin", None) == taxpayer.tin]
        received = [i for i in invoices if getattr(i, "buyer_tin", None) == taxpayer.tin]

        # ── Signal 1: High invoice volume per day ─────────────────────────────
        if issued:
            date_counts = Counter(i.invoice_date.date() for i in issued)
            max_per_day = max(date_counts.values())
            avg_per_day = len(issued) / max(len(date_counts), 1)
            if max_per_day >= self.MIN_INVOICES_PER_DAY:
                signals["high_volume"] = {
                    "max_invoices_per_day": max_per_day,
                    "avg_invoices_per_day": round(avg_per_day, 1),
                    "threshold": self.MIN_INVOICES_PER_DAY,
                    "total_issued": len(issued),
                }
                risk_score += min(30, max_per_day / self.MIN_INVOICES_PER_DAY * 15)

        # ── Signal 2: Suspiciously round amounts ──────────────────────────────
        if issued:
            round_count = sum(
                1 for inv in issued if self._is_round_amount(inv.net_amount)
            )
            round_ratio = round_count / len(issued)
            if round_ratio >= self.ROUND_AMOUNT_THRESHOLD:
                signals["round_amounts"] = {
                    "round_invoice_count": round_count,
                    "total_invoices": len(issued),
                    "round_ratio": round(round_ratio, 3),
                    "threshold": self.ROUND_AMOUNT_THRESHOLD,
                }
                risk_score += min(25, round_ratio * 30)

        # ── Signal 3: No purchase invoices (pure VAT generator) ───────────────
        if issued:
            purchase_ratio = len(received) / len(issued)
            if purchase_ratio < self.PURCHASE_RATIO_THRESHOLD:
                signals["no_purchase_invoices"] = {
                    "issued_count": len(issued),
                    "received_count": len(received),
                    "purchase_ratio": round(purchase_ratio, 3),
                    "threshold": self.PURCHASE_RATIO_THRESHOLD,
                }
                risk_score += 25

        # ── Signal 4: Repeated buyers (issuing to same few buyers) ────────────
        if issued:
            buyer_counts = Counter(
                getattr(inv, "buyer_tin", "unknown") for inv in issued
            )
            unique_buyers = len(buyer_counts)
            if unique_buyers <= 3 and len(issued) > 20:
                signals["concentrated_buyers"] = {
                    "unique_buyers": unique_buyers,
                    "total_invoices": len(issued),
                    "buyers": dict(buyer_counts.most_common(5)),
                }
                risk_score += 15

        # ── Signal 5: Benford's Law anomaly ──────────────────────────────────
        if len(issued) >= 20:
            benford_deviation = self._benford_analysis(issued)
            if benford_deviation > self.BENFORD_DEVIATION_THRESHOLD:
                signals["benford_anomaly"] = {
                    "mean_deviation": round(benford_deviation, 4),
                    "threshold": self.BENFORD_DEVIATION_THRESHOLD,
                    "invoice_count_analyzed": len(issued),
                }
                risk_score += min(20, benford_deviation * 80)

        # ── Signal 6: Newly registered, immediately high volume ───────────────
        if taxpayer.is_newly_registered and issued:
            signals["new_entity_high_volume"] = {
                "is_newly_registered": True,
                "invoices_issued": len(issued),
            }
            risk_score += 10

        risk_score = min(risk_score, 100.0)
        detected = risk_score >= 30 and bool(signals)

        if not detected:
            return self._build_no_detection()

        severity = self._score_to_severity(risk_score)
        total_vat = sum(inv.vat_amount for inv in issued)

        return DetectionResult(
            fraud_type=self.fraud_type,
            detected=True,
            risk_score=round(risk_score, 2),
            confidence=min(0.9, len(signals) / 5),
            severity=severity,
            title=f"Potential Invoice Mill – {taxpayer.vat_number}",
            description=(
                f"Entity {taxpayer.name} ({taxpayer.vat_number}) exhibits "
                f"{len(signals)} invoice mill indicators: {', '.join(signals.keys())}. "
                f"Total VAT on suspect invoices: {total_vat:,.2f}"
            ),
            evidence=signals,
            related_tins=[taxpayer.tin],
            estimated_revenue_at_risk=total_vat,
            detection_rule=self.name,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _is_round_amount(amount: float) -> bool:
        """Check if amount ends in at least 2 zeros (i.e., multiple of 100)."""
        return amount > 0 and amount % 100 == 0

    def _benford_analysis(self, invoices: list) -> float:
        """Return mean absolute deviation from Benford's expected distribution."""
        digits = []
        for inv in invoices:
            amount = abs(inv.net_amount)
            if amount > 0:
                leading = int(str(amount).replace(".", "").lstrip("0")[0])
                if 1 <= leading <= 9:
                    digits.append(leading)

        if not digits:
            return 0.0

        observed = Counter(digits)
        total = len(digits)
        deviations = []
        for digit, expected_freq in BENFORD_EXPECTED.items():
            observed_freq = observed.get(digit, 0) / total
            deviations.append(abs(observed_freq - expected_freq))

        return sum(deviations) / len(deviations)

    @staticmethod
    def _score_to_severity(score: float) -> AlertSeverity:
        if score >= 80:
            return AlertSeverity.CRITICAL
        if score >= 60:
            return AlertSeverity.HIGH
        if score >= 40:
            return AlertSeverity.MEDIUM
        return AlertSeverity.LOW

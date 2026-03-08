"""
Contra-Trading Fraud Detector
==============================
Detects the advanced carousel variant where fraudsters use two parallel transaction
chains — one legitimate, one fraudulent — whose VAT liabilities deliberately
cancel each other out, making individual companies appear compliant.

Key indicators (per IMF WP/07/31):
  1. A single entity simultaneously claims large input VAT and owes large output VAT
     that happen to be almost exactly equal period after period.
  2. The entity trades in completely unrelated commodity groups simultaneously
     (e.g., buying mobile phones and selling potatoes — designed to obscure the pattern).
  3. Net VAT liability is always near zero despite high gross turnover.
  4. The entity appears in multiple unrelated supply chain networks.
  5. The timing of purchases in one chain mirrors sales in another chain.
"""

from typing import Any

from src.fraud_detection.rules.base import BaseDetector, DetectionResult
from src.models.alert import AlertSeverity, FraudType


class ContraTradingDetector(BaseDetector):
    name = "contra_trading"
    fraud_type = FraudType.CONTRA_TRADING

    NET_LIABILITY_TOLERANCE = 0.02      # Net VAT / gross VAT < 2% = near-zero
    MIN_GROSS_VAT = 50_000              # Only flag if significant amounts involved
    SECTOR_DIVERSITY_THRESHOLD = 3      # Trading in 3+ unrelated sectors
    PERIODS_REQUIRED = 3                # Must observe pattern over N periods

    async def evaluate(self, context: dict[str, Any]) -> DetectionResult:
        taxpayer = context.get("taxpayer")
        transactions = context.get("transactions", [])
        invoices = context.get("invoices", [])

        if not taxpayer or len(transactions) < self.PERIODS_REQUIRED:
            return self._build_no_detection()

        signals: dict[str, Any] = {}
        risk_score = 0.0

        vat_returns = [t for t in transactions if t.transaction_type == "vat_return"]

        # ── Signal 1: Persistent near-zero net VAT on high gross turnover ─────
        near_zero_periods = []
        for ret in vat_returns:
            output = getattr(ret, "output_vat", 0) or 0
            input_ = getattr(ret, "input_vat", 0) or 0
            gross = output + input_
            net = abs(output - input_)
            if gross >= self.MIN_GROSS_VAT:
                net_ratio = net / gross if gross > 0 else 1.0
                if net_ratio < self.NET_LIABILITY_TOLERANCE:
                    near_zero_periods.append({
                        "period": ret.period,
                        "output_vat": output,
                        "input_vat": input_,
                        "gross_vat": gross,
                        "net_ratio": round(net_ratio, 4),
                    })

        if len(near_zero_periods) >= self.PERIODS_REQUIRED:
            signals["persistent_near_zero_net_vat"] = {
                "near_zero_period_count": len(near_zero_periods),
                "required_periods": self.PERIODS_REQUIRED,
                "periods": near_zero_periods,
            }
            risk_score += min(40, len(near_zero_periods) * 10)

        # ── Signal 2: Diverse unrelated commodity trading ─────────────────────
        commodity_codes = {
            getattr(inv, "commodity_code", None)[:2]  # 2-digit HS chapter
            for inv in invoices
            if getattr(inv, "commodity_code", None)
        }
        if len(commodity_codes) >= self.SECTOR_DIVERSITY_THRESHOLD:
            signals["diverse_unrelated_sectors"] = {
                "hs_chapters": list(commodity_codes),
                "sector_count": len(commodity_codes),
                "threshold": self.SECTOR_DIVERSITY_THRESHOLD,
            }
            risk_score += min(20, len(commodity_codes) * 5)

        # ── Signal 3: Input and output chains show timing mirror ──────────────
        issued = [i for i in invoices if getattr(i, "seller_tin", None) == taxpayer.tin]
        received = [i for i in invoices if getattr(i, "buyer_tin", None) == taxpayer.tin]
        if issued and received:
            mirror_signals = self._detect_mirrored_timing(issued, received)
            if mirror_signals:
                signals["mirrored_chain_timing"] = mirror_signals
                risk_score += min(25, len(mirror_signals) * 8)

        # ── Signal 4: High gross VAT with consistently low net payments ────────
        total_paid = sum(
            getattr(t, "amount_paid", 0) or 0 for t in vat_returns
        )
        total_output = sum(
            getattr(t, "output_vat", 0) or 0 for t in vat_returns
        )
        if total_output > self.MIN_GROSS_VAT and total_paid < total_output * 0.05:
            signals["minimal_actual_payments"] = {
                "total_output_vat_declared": total_output,
                "total_vat_paid": total_paid,
                "payment_ratio": round(total_paid / total_output, 4) if total_output else 0,
            }
            risk_score += 20

        risk_score = min(risk_score, 100.0)
        detected = risk_score >= 35 and bool(signals)

        if not detected:
            return self._build_no_detection()

        severity = self._score_to_severity(risk_score)
        total_gross_vat = sum(
            (getattr(t, "output_vat", 0) or 0) + (getattr(t, "input_vat", 0) or 0)
            for t in vat_returns
        )

        return DetectionResult(
            fraud_type=self.fraud_type,
            detected=True,
            risk_score=round(risk_score, 2),
            confidence=min(0.85, len(signals) / 4),
            severity=severity,
            title=f"Potential Contra-Trading – {taxpayer.vat_number}",
            description=(
                f"Entity {taxpayer.name} ({taxpayer.vat_number}) exhibits "
                f"contra-trading patterns: {', '.join(signals.keys())}. "
                f"Gross VAT involved: {total_gross_vat:,.2f}"
            ),
            evidence=signals,
            related_tins=[taxpayer.tin],
            estimated_revenue_at_risk=total_gross_vat * 0.5,
            detection_rule=self.name,
        )

    def _detect_mirrored_timing(self, issued: list, received: list) -> list[dict]:
        """Detect if purchase dates in one chain mirror sale dates in another."""
        mirrors = []
        tolerance_hours = 48
        for buy_inv in received:
            buy_date = getattr(buy_inv, "invoice_date", None)
            if not buy_date:
                continue
            for sell_inv in issued:
                sell_date = getattr(sell_inv, "invoice_date", None)
                if not sell_date:
                    continue
                diff_hours = abs((sell_date - buy_date).total_seconds()) / 3600
                if diff_hours <= tolerance_hours:
                    # Check different commodity codes = contra-trading indicator
                    buy_code = getattr(buy_inv, "commodity_code", "")
                    sell_code = getattr(sell_inv, "commodity_code", "")
                    if buy_code and sell_code and buy_code[:2] != sell_code[:2]:
                        mirrors.append({
                            "buy_date": str(buy_date),
                            "sell_date": str(sell_date),
                            "hours_apart": round(diff_hours, 1),
                            "buy_commodity": buy_code,
                            "sell_commodity": sell_code,
                        })
        return mirrors[:10]

    @staticmethod
    def _score_to_severity(score: float) -> AlertSeverity:
        if score >= 80:
            return AlertSeverity.CRITICAL
        if score >= 60:
            return AlertSeverity.HIGH
        if score >= 40:
            return AlertSeverity.MEDIUM
        return AlertSeverity.LOW

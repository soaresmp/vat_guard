"""
Carousel Fraud Detector
=======================
Detects circular VAT chains where goods (often high-value, easy to transport)
are traded around a ring of companies — typically across borders — with each
company claiming VAT refunds but one "missing trader" never remitting.

Pattern: A → B → C → D → ... → A  (goods/VAT circle)

IMF Reference: WP/07/31 – Keen & Smith (2007)
Detection strategy: graph cycle analysis + price trajectory + commodity screening
"""

from typing import Any

import networkx as nx

from src.fraud_detection.rules.base import BaseDetector, DetectionResult
from src.models.alert import AlertSeverity, FraudType

# High-risk commodities frequently used in carousel fraud
# (mobile phones, CPUs, precious metals, carbon credits, etc.)
HIGH_RISK_COMMODITY_CODES = {
    "8517",   # Mobile phones / telephones
    "8471",   # Computers
    "8542",   # Integrated circuits
    "7108",   # Gold
    "7110",   # Platinum
    "2716",   # Electrical energy
    "3825",   # Residual chemicals
    "6301",   # Carbon credits (generic proxy)
}


class CarouselFraudDetector(BaseDetector):
    name = "carousel_fraud"
    fraud_type = FraudType.CAROUSEL_FRAUD

    MIN_CHAIN_DEPTH = 3          # Minimum nodes to form a suspicious cycle
    PRICE_DROP_THRESHOLD = 0.05  # 5% price drop per hop is suspicious
    RAPID_RESALE_HOURS = 24      # Resale within 24 hours flags rapid trading

    async def evaluate(self, context: dict[str, Any]) -> DetectionResult:
        invoices = context.get("invoices", [])
        graph: nx.DiGraph | None = context.get("graph")
        taxpayer = context.get("taxpayer")

        if not invoices and graph is None:
            return self._build_no_detection()

        signals: dict[str, Any] = {}
        risk_score = 0.0

        # ── Build transaction graph if not provided ───────────────────────────
        if graph is None:
            graph = self._build_graph(invoices)

        # ── Signal 1: Circular transaction chains (cycles in directed graph) ──
        cycles = self._find_cycles(graph)
        if cycles:
            max_cycle_len = max(len(c) for c in cycles)
            signals["circular_chains"] = {
                "cycle_count": len(cycles),
                "max_cycle_length": max_cycle_len,
                "cycles": [list(c) for c in cycles[:5]],  # First 5 for evidence
            }
            risk_score += min(50, len(cycles) * 35 + (max_cycle_len - self.MIN_CHAIN_DEPTH) * 5)

        # ── Signal 2: High-risk commodity codes ───────────────────────────────
        high_risk_invoices = [
            inv for inv in invoices
            if self._is_high_risk_commodity(getattr(inv, "commodity_code", None))
        ]
        if high_risk_invoices:
            signals["high_risk_commodities"] = {
                "count": len(high_risk_invoices),
                "commodity_codes": list({
                    inv.commodity_code for inv in high_risk_invoices
                    if inv.commodity_code
                }),
                "total_vat_value": sum(inv.vat_amount for inv in high_risk_invoices),
            }
            risk_score += min(25, len(high_risk_invoices) * 3)

        # ── Signal 3: Price drop across the chain ─────────────────────────────
        price_drops = self._detect_price_drops(invoices)
        if price_drops:
            signals["price_drops_in_chain"] = price_drops
            risk_score += min(20, len(price_drops) * 8)

        # ── Signal 4: Rapid resale (goods flipped within hours) ───────────────
        rapid_resales = self._detect_rapid_resales(invoices)
        if rapid_resales:
            signals["rapid_resales"] = {
                "count": len(rapid_resales),
                "min_hours_between_trades": min(r["hours"] for r in rapid_resales),
                "details": rapid_resales[:5],
            }
            risk_score += min(20, len(rapid_resales) * 5)

        # ── Signal 5: Cross-border chain involvement ──────────────────────────
        cross_border = [
            inv for inv in invoices
            if getattr(inv, "invoice_type", None) in ("export", "intra_community")
        ]
        if cross_border and "circular_chains" in signals:
            countries = {inv.seller_country for inv in cross_border} | {
                inv.buyer_country for inv in cross_border if inv.buyer_country
            }
            signals["cross_border_carousel"] = {
                "countries_involved": list(countries),
                "cross_border_invoice_count": len(cross_border),
            }
            risk_score += min(20, len(countries) * 5)

        risk_score = min(risk_score, 100.0)
        detected = risk_score >= 35 and bool(signals)

        if not detected:
            return self._build_no_detection()

        severity = self._score_to_severity(risk_score)
        involved_tins = list(graph.nodes()) if graph else []
        total_vat = sum(inv.vat_amount for inv in invoices)

        return DetectionResult(
            fraud_type=self.fraud_type,
            detected=True,
            risk_score=round(risk_score, 2),
            confidence=min(0.95, len(signals) / 4),
            severity=severity,
            title=f"Potential Carousel Fraud – {len(involved_tins)} entities",
            description=(
                f"Carousel fraud pattern detected across {len(involved_tins)} entities. "
                f"Indicators: {', '.join(signals.keys())}. "
                f"Total VAT at risk: {total_vat:,.2f}"
            ),
            evidence=signals,
            related_tins=involved_tins[:20],
            estimated_revenue_at_risk=total_vat,
            detection_rule=self.name,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_graph(self, invoices: list) -> nx.DiGraph:
        """Build a directed graph: edge seller → buyer, weight = vat_amount."""
        G = nx.DiGraph()
        for inv in invoices:
            seller = getattr(inv, "seller_tin", None)
            buyer = getattr(inv, "buyer_tin", None)
            if seller and buyer:
                if G.has_edge(seller, buyer):
                    G[seller][buyer]["weight"] += inv.vat_amount
                    G[seller][buyer]["invoice_count"] += 1
                else:
                    G.add_edge(
                        seller, buyer,
                        weight=inv.vat_amount,
                        invoice_count=1,
                    )
        return G

    def _find_cycles(self, graph: nx.DiGraph) -> list[list[str]]:
        """Find simple cycles of length >= MIN_CHAIN_DEPTH."""
        try:
            cycles = [
                c for c in nx.simple_cycles(graph)
                if len(c) >= self.MIN_CHAIN_DEPTH
            ]
            return cycles
        except Exception:
            return []

    def _is_high_risk_commodity(self, code: str | None) -> bool:
        if not code:
            return False
        return any(code.startswith(hrc) for hrc in HIGH_RISK_COMMODITY_CODES)

    def _detect_price_drops(self, invoices: list) -> list[dict]:
        """Detect where unit price drops between seller and buyer in a chain."""
        drops = []
        price_map: dict[str, float] = {}  # seller_tin → last unit price
        for inv in sorted(invoices, key=lambda i: i.invoice_date):
            seller = getattr(inv, "seller_tin", None)
            buyer = getattr(inv, "buyer_tin", None)
            net = getattr(inv, "net_amount", 0)
            if not (seller and buyer and net):
                continue
            if seller in price_map:
                prev_price = price_map[seller]
                if prev_price > 0 and (prev_price - net) / prev_price > self.PRICE_DROP_THRESHOLD:
                    drops.append({
                        "seller_tin": seller,
                        "buyer_tin": buyer,
                        "previous_price": prev_price,
                        "current_price": net,
                        "drop_pct": round((prev_price - net) / prev_price * 100, 2),
                    })
            price_map[buyer] = net
        return drops

    def _detect_rapid_resales(self, invoices: list) -> list[dict]:
        """Detect goods sold onward within RAPID_RESALE_HOURS hours."""
        rapid = []
        # Build: for each buyer, find their corresponding seller date
        buy_dates: dict[str, list] = {}
        sell_dates: dict[str, list] = {}
        for inv in invoices:
            seller = getattr(inv, "seller_tin", None)
            buyer = getattr(inv, "buyer_tin", None)
            dt = getattr(inv, "invoice_date", None)
            if seller and dt:
                sell_dates.setdefault(seller, []).append(dt)
            if buyer and dt:
                buy_dates.setdefault(buyer, []).append(dt)
        for tin, buy_times in buy_dates.items():
            if tin in sell_dates:
                for bt in buy_times:
                    for st in sell_dates[tin]:
                        diff_hours = abs((st - bt).total_seconds()) / 3600
                        if diff_hours <= self.RAPID_RESALE_HOURS:
                            rapid.append({
                                "tin": tin,
                                "buy_date": str(bt),
                                "sell_date": str(st),
                                "hours": round(diff_hours, 1),
                            })
        return rapid

    @staticmethod
    def _score_to_severity(score: float) -> AlertSeverity:
        if score >= 80:
            return AlertSeverity.CRITICAL
        if score >= 60:
            return AlertSeverity.HIGH
        if score >= 40:
            return AlertSeverity.MEDIUM
        return AlertSeverity.LOW

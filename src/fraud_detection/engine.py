"""
Fraud Detection Engine
=======================
Central orchestrator that runs all detection rules, ML models, and graph
analytics against a taxpayer / invoice context, then aggregates results
into actionable alerts.

Architecture:
  1. Context assembly     – load taxpayer, invoices, transactions
  2. Rule evaluation      – run all rule-based detectors in parallel
  3. ML / anomaly check   – Isolation Forest + Benford
  4. Graph analysis       – build network, detect cycles
  5. Risk aggregation     – composite score via TaxpayerRiskScorer
  6. Alert generation     – emit Alert objects for each significant finding
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from src.config import get_settings
from src.fraud_detection.graph.circular_detector import CircularChainDetector
from src.fraud_detection.graph.network_analyzer import NetworkAnalyzer
from src.fraud_detection.ml.anomaly_detector import AnomalyDetector
from src.fraud_detection.ml.benford_analysis import BenfordAnalyzer
from src.fraud_detection.ml.risk_scorer import TaxpayerRiskScorer
from src.fraud_detection.realtime.invoice_matcher import InvoiceMatcher
from src.fraud_detection.rules.carousel import CarouselFraudDetector
from src.fraud_detection.rules.contra_trading import ContraTradingDetector
from src.fraud_detection.rules.invoice_mill import InvoiceMillDetector
from src.fraud_detection.rules.missing_trader import MissingTraderDetector
from src.fraud_detection.rules.refund_fraud import RefundFraudDetector
from src.models.alert import Alert, AlertSeverity, AlertStatus, FraudType

logger = logging.getLogger(__name__)
settings = get_settings()


class FraudDetectionEngine:
    """
    Main entry-point for all VAT fraud analysis.

    Usage:
        engine = FraudDetectionEngine()
        alerts = await engine.analyze_taxpayer(taxpayer, invoices, transactions)
    """

    def __init__(self) -> None:
        # Rule-based detectors
        self._missing_trader = MissingTraderDetector()
        self._carousel = CarouselFraudDetector()
        self._invoice_mill = InvoiceMillDetector()
        self._refund_fraud = RefundFraudDetector()
        self._contra_trading = ContraTradingDetector()

        # ML / analytics
        self._anomaly_detector = AnomalyDetector(contamination=0.05)
        self._benford = BenfordAnalyzer()
        self._risk_scorer = TaxpayerRiskScorer(
            high_threshold=settings.risk_score_high_threshold,
            critical_threshold=settings.risk_score_critical_threshold,
        )

        # Graph analytics
        self._network_analyzer = NetworkAnalyzer()
        self._circular_detector = CircularChainDetector()

        # Invoice matcher
        self._matcher = InvoiceMatcher()

    # ── Public API ────────────────────────────────────────────────────────────

    async def analyze_taxpayer(
        self,
        taxpayer: Any,
        invoices: list[Any],
        transactions: list[Any],
        third_party_flags: dict | None = None,
        now: datetime | None = None,
    ) -> tuple[list[Alert], float]:
        """
        Run full fraud analysis for a taxpayer.

        Returns:
            (list of Alert ORM objects ready for persistence, composite risk score)
        """
        now = now or datetime.now(timezone.utc)
        ctx = {
            "taxpayer": taxpayer,
            "invoices": invoices,
            "transactions": transactions,
            "now": now,
        }

        logger.info("Starting fraud analysis for TIN=%s", getattr(taxpayer, "tin", "?"))

        # ── Step 1: Run all rule detectors concurrently ───────────────────────
        detection_results = await asyncio.gather(
            self._missing_trader.evaluate(ctx),
            self._carousel.evaluate(ctx),
            self._invoice_mill.evaluate(ctx),
            self._refund_fraud.evaluate(ctx),
            self._contra_trading.evaluate(ctx),
            return_exceptions=True,
        )

        valid_results = [
            r for r in detection_results
            if not isinstance(r, Exception) and r.detected
        ]
        for exc in detection_results:
            if isinstance(exc, Exception):
                logger.error("Detection rule failed: %s", exc)

        # ── Step 2: Anomaly detection ─────────────────────────────────────────
        taxpayer_features = AnomalyDetector.extract_taxpayer_features(taxpayer, transactions)
        anomaly_result = self._anomaly_detector.detect(taxpayer_features)

        # ── Step 3: Benford analysis on issued invoices ───────────────────────
        issued_invoices = [
            inv for inv in invoices
            if getattr(inv, "seller_tin", None) == getattr(taxpayer, "tin", None)
        ]
        benford_result = self._benford.analyze_invoices(issued_invoices) if issued_invoices else None

        if benford_result and benford_result.is_suspicious:
            # Create a synthetic detection result for Benford anomaly
            from src.fraud_detection.rules.base import DetectionResult
            benford_detection = DetectionResult(
                fraud_type=FraudType.BENFORD_ANOMALY,
                detected=True,
                risk_score=min(80.0, (1 - benford_result.p_value) * 80),
                confidence=min(0.9, 1 - benford_result.p_value),
                severity=AlertSeverity.HIGH if benford_result.p_value < 0.01 else AlertSeverity.MEDIUM,
                title=f"Benford's Law Anomaly – {getattr(taxpayer, 'vat_number', '')}",
                description=benford_result.interpretation,
                evidence={
                    "chi2": benford_result.chi2_statistic,
                    "p_value": benford_result.p_value,
                    "mad": benford_result.mean_absolute_deviation,
                    "sample_size": benford_result.sample_size,
                    "observed_frequencies": benford_result.observed_frequencies,
                },
                detection_rule="benford_analysis",
            )
            valid_results.append(benford_detection)

        # ── Step 4: Graph / network analysis ─────────────────────────────────
        network_result = self._network_analyzer.analyze(invoices)
        circular_chains = self._circular_detector.detect(invoices)
        network_metrics = self._network_analyzer.get_node_metrics(
            self._network_analyzer.build_graph(invoices),
            getattr(taxpayer, "tin", ""),
        )

        if network_result.risk_score >= 40 and circular_chains:
            from src.fraud_detection.rules.base import DetectionResult
            network_detection = DetectionResult(
                fraud_type=FraudType.NETWORK_ANOMALY,
                detected=True,
                risk_score=network_result.risk_score,
                confidence=0.75,
                severity=self._network_severity(network_result.risk_score),
                title=f"Suspicious Transaction Network – {getattr(taxpayer, 'vat_number', '')}",
                description=network_result.summary,
                evidence={
                    "cycle_count": network_result.cycle_count,
                    "suspicious_nodes": network_result.suspicious_nodes[:10],
                    "strongly_connected_components": network_result.strongly_connected_components,
                    "circular_chains": [
                        {
                            "chain": c.chain,
                            "total_vat": c.total_vat,
                            "cross_border": c.cross_border,
                            "potential_missing_trader": c.potential_missing_trader,
                            "risk_score": c.risk_score,
                        }
                        for c in circular_chains[:5]
                    ],
                },
                related_tins=network_result.suspicious_nodes[:10],
                estimated_revenue_at_risk=network_result.total_vat_in_network,
                detection_rule="network_analyzer",
            )
            valid_results.append(network_detection)

        # ── Step 5: Composite risk score ──────────────────────────────────────
        risk_result = self._risk_scorer.score(
            taxpayer=taxpayer,
            detection_results=valid_results,
            anomaly_result=anomaly_result,
            network_metrics=network_metrics,
            third_party_flags=third_party_flags or {},
            transactions=transactions,
        )

        logger.info(
            "Analysis complete for TIN=%s: score=%.1f level=%s detections=%d",
            getattr(taxpayer, "tin", "?"),
            risk_result.total_score,
            risk_result.risk_level,
            len(valid_results),
        )

        # ── Step 6: Build Alert objects ───────────────────────────────────────
        alerts = [
            self._result_to_alert(r, taxpayer)
            for r in valid_results
            if r.is_alert_worthy
        ]

        return alerts, risk_result.total_score

    async def analyze_invoice_match(
        self,
        seller_invoices: list[Any],
        buyer_invoices: list[Any],
    ) -> list[dict]:
        """
        Real-time invoice matching: detect unmatched / mismatched invoices.
        Returns list of match result dicts for suspicious invoices only.
        """
        results = self._matcher.match_period(seller_invoices, buyer_invoices)
        from src.fraud_detection.realtime.invoice_matcher import MatchStatus
        suspicious = [
            {
                "status": r.status,
                "seller_tin": r.seller_tin,
                "buyer_tin": r.buyer_tin,
                "invoice_number": r.invoice_number,
                "seller_vat_amount": r.seller_vat_amount,
                "buyer_vat_amount": r.buyer_vat_amount,
                "discrepancy": r.discrepancy,
                "discrepancy_pct": r.discrepancy_pct,
                "risk_score": r.risk_score,
                "flags": r.flags,
            }
            for r in results
            if r.status != MatchStatus.MATCHED
        ]
        return suspicious

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _result_to_alert(self, result: Any, taxpayer: Any) -> Alert:
        """Convert a DetectionResult into an Alert ORM object."""
        return Alert(
            taxpayer_tin=getattr(taxpayer, "tin", None),
            fraud_type=result.fraud_type,
            severity=result.severity,
            status=AlertStatus.OPEN,
            title=result.title,
            description=result.description,
            detection_rule=result.detection_rule,
            risk_score=result.risk_score,
            confidence=result.confidence,
            evidence=result.evidence,
            related_tins=result.related_tins,
            related_invoice_ids=[str(i) for i in result.related_invoice_ids],
            estimated_revenue_at_risk=result.estimated_revenue_at_risk,
        )

    @staticmethod
    def _network_severity(score: float) -> AlertSeverity:
        if score >= 80:
            return AlertSeverity.CRITICAL
        if score >= 60:
            return AlertSeverity.HIGH
        return AlertSeverity.MEDIUM

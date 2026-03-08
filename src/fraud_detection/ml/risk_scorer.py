"""
Taxpayer Risk Scorer
=====================
Produces a composite 0–100 risk score for a taxpayer by aggregating:
  - Rule-based detection results
  - Anomaly detection output
  - Compliance history
  - Network centrality (from graph analytics)
  - Third-party data signals (customs, FIU)

Higher scores indicate higher likelihood of fraud.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Weight configuration for each signal category
WEIGHTS = {
    "rule_detections": 0.40,      # Fraud rule triggers
    "anomaly_score": 0.20,        # ML anomaly detection
    "compliance_history": 0.20,   # Filing / payment history
    "network_centrality": 0.10,   # Suspicious network position
    "third_party_signals": 0.10,  # Customs / FIU flags
}


@dataclass
class RiskScoreResult:
    total_score: float              # 0–100
    component_scores: dict[str, float]
    breakdown: dict[str, Any]
    risk_level: str                 # low | medium | high | critical
    top_risk_factors: list[str]


class TaxpayerRiskScorer:
    """Composite risk scorer combining multiple signal sources."""

    def __init__(
        self,
        high_threshold: float = 75.0,
        critical_threshold: float = 90.0,
    ):
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold

    def score(
        self,
        taxpayer: Any,
        detection_results: list,          # List of DetectionResult objects
        anomaly_result: Any | None = None,
        network_metrics: dict | None = None,
        third_party_flags: dict | None = None,
        transactions: list | None = None,
    ) -> RiskScoreResult:
        """Compute composite risk score."""
        transactions = transactions or []
        network_metrics = network_metrics or {}
        third_party_flags = third_party_flags or {}

        components: dict[str, float] = {}
        breakdown: dict[str, Any] = {}

        # ── 1. Rule detection score ───────────────────────────────────────────
        triggered = [r for r in detection_results if getattr(r, "detected", False)]
        if triggered:
            weighted_avg = sum(
                r.risk_score * r.confidence for r in triggered
            ) / sum(r.confidence for r in triggered)
            components["rule_detections"] = min(100.0, weighted_avg)
            breakdown["triggered_rules"] = [
                {
                    "rule": r.detection_rule,
                    "fraud_type": r.fraud_type,
                    "score": r.risk_score,
                    "confidence": r.confidence,
                }
                for r in triggered
            ]
        else:
            components["rule_detections"] = 0.0

        # ── 2. Anomaly score ──────────────────────────────────────────────────
        if anomaly_result:
            components["anomaly_score"] = getattr(anomaly_result, "anomaly_score", 0.0)
            breakdown["anomaly"] = {
                "score": components["anomaly_score"],
                "explanation": getattr(anomaly_result, "explanation", ""),
            }
        else:
            components["anomaly_score"] = 0.0

        # ── 3. Compliance history score ───────────────────────────────────────
        compliance_score = self._compute_compliance_score(taxpayer, transactions)
        components["compliance_history"] = compliance_score
        breakdown["compliance"] = {
            "score": compliance_score,
            "consecutive_late_filings": getattr(taxpayer, "consecutive_late_filings", 0),
            "taxpayer_status": getattr(taxpayer, "status", "active"),
        }

        # ── 4. Network centrality ─────────────────────────────────────────────
        if network_metrics:
            centrality = network_metrics.get("betweenness_centrality", 0.0)
            in_degree = network_metrics.get("in_degree", 0)
            # High betweenness in a VAT network is suspicious (buffer trader)
            network_score = min(100.0, centrality * 200 + in_degree * 2)
            components["network_centrality"] = network_score
            breakdown["network"] = network_metrics
        else:
            components["network_centrality"] = 0.0

        # ── 5. Third-party signals ────────────────────────────────────────────
        tp_score = self._compute_third_party_score(third_party_flags)
        components["third_party_signals"] = tp_score
        breakdown["third_party"] = third_party_flags

        # ── Weighted total ────────────────────────────────────────────────────
        total = sum(
            components[key] * WEIGHTS[key]
            for key in WEIGHTS
            if key in components
        )
        total = round(min(100.0, total), 2)

        risk_level = self._classify_risk(total)
        top_factors = self._identify_top_factors(components, breakdown)

        return RiskScoreResult(
            total_score=total,
            component_scores=components,
            breakdown=breakdown,
            risk_level=risk_level,
            top_risk_factors=top_factors,
        )

    def _compute_compliance_score(self, taxpayer: Any, transactions: list) -> float:
        score = 0.0
        late = getattr(taxpayer, "consecutive_late_filings", 0)
        score += min(40, late * 8)

        status = getattr(taxpayer, "status", "active")
        if status in ("suspended", "missing", "under_investigation"):
            score += 40
        elif status == "deregistered":
            score += 20

        # Late filings in transaction history
        vat_returns = [t for t in transactions if t.transaction_type == "vat_return"]
        if vat_returns:
            late_ratio = sum(1 for t in vat_returns if getattr(t, "is_late", False)) / len(vat_returns)
            score += late_ratio * 30

        return min(100.0, score)

    def _compute_third_party_score(self, flags: dict) -> float:
        score = 0.0
        if flags.get("fiu_sar_filed"):
            score += 50
        if flags.get("customs_mismatch"):
            score += 30
        if flags.get("interpol_watchlist"):
            score += 40
        if flags.get("previous_fraud_conviction"):
            score += 60
        if flags.get("linked_to_known_fraudster"):
            score += 35
        return min(100.0, score)

    def _classify_risk(self, score: float) -> str:
        if score >= self.critical_threshold:
            return "critical"
        if score >= self.high_threshold:
            return "high"
        if score >= 40:
            return "medium"
        return "low"

    def _identify_top_factors(
        self, components: dict[str, float], breakdown: dict
    ) -> list[str]:
        factors = []
        sorted_components = sorted(components.items(), key=lambda x: x[1], reverse=True)
        for key, value in sorted_components[:3]:
            if value >= 20:
                factors.append(f"{key.replace('_', ' ').title()}: {value:.1f}/100")
        if breakdown.get("triggered_rules"):
            for rule in breakdown["triggered_rules"][:2]:
                factors.append(f"Rule: {rule['fraud_type']} (score={rule['score']:.1f})")
        return factors[:5]

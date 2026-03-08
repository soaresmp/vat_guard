"""
Base class for all rule-based fraud detectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.models.alert import FraudType, AlertSeverity


@dataclass
class DetectionResult:
    """Outcome of a single fraud detection rule evaluation."""
    fraud_type: FraudType
    detected: bool
    risk_score: float            # 0–100
    confidence: float            # 0–1
    severity: AlertSeverity
    title: str
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)
    related_tins: list[str] = field(default_factory=list)
    related_invoice_ids: list[str] = field(default_factory=list)
    estimated_revenue_at_risk: float | None = None
    detection_rule: str = ""

    @property
    def is_alert_worthy(self) -> bool:
        return self.detected and self.risk_score >= 30


class BaseDetector(ABC):
    """Abstract base for all VAT fraud detection rules."""

    name: str = "base"
    fraud_type: FraudType = FraudType.OTHER

    @abstractmethod
    async def evaluate(self, context: dict[str, Any]) -> DetectionResult:
        """
        Evaluate the detection rule given a context dictionary.
        Context keys depend on the specific rule but typically include:
            - taxpayer: Taxpayer ORM object
            - invoice: Invoice ORM object (optional)
            - invoices: list of Invoice ORM objects (optional)
            - transactions: list of Transaction ORM objects
            - graph: nx.DiGraph of transaction network (optional)
        """
        ...

    def _build_no_detection(self) -> DetectionResult:
        return DetectionResult(
            fraud_type=self.fraud_type,
            detected=False,
            risk_score=0.0,
            confidence=1.0,
            severity=AlertSeverity.LOW,
            title="No fraud detected",
            description="Rule evaluation passed without finding suspicious patterns.",
            detection_rule=self.name,
        )

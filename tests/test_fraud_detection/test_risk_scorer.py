"""Tests for the composite Taxpayer Risk Scorer."""

import pytest
from unittest.mock import MagicMock

from src.fraud_detection.ml.risk_scorer import TaxpayerRiskScorer
from src.fraud_detection.rules.base import DetectionResult
from src.models.alert import AlertSeverity, FraudType
from tests.conftest import make_taxpayer, make_transaction


@pytest.fixture
def scorer():
    return TaxpayerRiskScorer(high_threshold=75, critical_threshold=90)


def make_detection(fraud_type=FraudType.MISSING_TRADER, score=80.0, confidence=0.9):
    return DetectionResult(
        fraud_type=fraud_type,
        detected=True,
        risk_score=score,
        confidence=confidence,
        severity=AlertSeverity.HIGH,
        title="Test Detection",
        description="Test",
        detection_rule="test_rule",
    )


def test_low_risk_compliant_taxpayer(scorer):
    """Compliant taxpayer with no signals should score low."""
    taxpayer = make_taxpayer()
    result = scorer.score(taxpayer, detection_results=[], transactions=[])
    assert result.total_score < 40
    assert result.risk_level == "low"


def test_high_risk_with_detections(scorer):
    """Multiple rule triggers should push score into high range."""
    taxpayer = make_taxpayer(consecutive_late_filings=4)
    detections = [
        make_detection(FraudType.MISSING_TRADER, 85, 0.9),
        make_detection(FraudType.CAROUSEL_FRAUD, 70, 0.8),
    ]
    result = scorer.score(taxpayer, detections, transactions=[])
    assert result.total_score >= 50
    assert result.risk_level in ("high", "critical")


def test_fiu_flags_increase_score(scorer):
    """Third-party FIU signals should significantly increase risk."""
    taxpayer = make_taxpayer()
    result_without = scorer.score(taxpayer, [], transactions=[])
    result_with_fiu = scorer.score(
        taxpayer, [],
        transactions=[],
        third_party_flags={"fiu_sar_filed": True, "linked_to_known_fraudster": True},
    )
    assert result_with_fiu.total_score > result_without.total_score


def test_critical_threshold(scorer):
    """Taxpayer with suspended status and multiple signals should be critical."""
    taxpayer = make_taxpayer(
        status="suspended",
        consecutive_late_filings=6,
    )
    detections = [make_detection(score=95, confidence=0.95)]
    result = scorer.score(
        taxpayer, detections,
        transactions=[],
        third_party_flags={"previous_fraud_conviction": True},
    )
    assert result.risk_level == "critical"


def test_anomaly_score_contribution(scorer):
    """Anomaly detection output should be included in composite score."""
    taxpayer = make_taxpayer()
    anomaly = MagicMock()
    anomaly.anomaly_score = 90.0
    anomaly.explanation = "High isolation forest anomaly"

    result = scorer.score(taxpayer, [], anomaly_result=anomaly, transactions=[])
    assert result.component_scores.get("anomaly_score", 0) == 90.0
    assert result.total_score > 0


def test_top_risk_factors_populated(scorer):
    """Top risk factors should be identified and returned."""
    taxpayer = make_taxpayer(consecutive_late_filings=3, status="under_investigation")
    detections = [make_detection(FraudType.REFUND_FRAUD, 75, 0.85)]
    result = scorer.score(taxpayer, detections, transactions=[])
    assert len(result.top_risk_factors) > 0

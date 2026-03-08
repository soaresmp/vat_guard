"""Tests for Missing Trader (MTIC) fraud detector."""

import pytest
from datetime import datetime, timedelta, timezone

from src.fraud_detection.rules.missing_trader import MissingTraderDetector
from src.models.alert import AlertSeverity, FraudType
from tests.conftest import make_taxpayer, make_invoice, make_transaction


@pytest.fixture
def detector():
    return MissingTraderDetector()


@pytest.mark.asyncio
async def test_no_fraud_active_compliant_taxpayer(detector):
    """An active, compliant taxpayer should not trigger the rule."""
    taxpayer = make_taxpayer(
        consecutive_late_filings=0,
        last_filing_date=datetime.now(timezone.utc) - timedelta(days=10),
        is_newly_registered=False,
    )
    transactions = [make_transaction(transaction_type="vat_return")]
    result = await detector.evaluate({
        "taxpayer": taxpayer,
        "invoices": [],
        "transactions": transactions,
        "now": datetime.now(timezone.utc),
    })
    assert result.detected is False
    assert result.risk_score == 0.0


@pytest.mark.asyncio
async def test_inactive_trader_flagged(detector):
    """Taxpayer who has not filed for 120 days should be flagged."""
    taxpayer = make_taxpayer(
        consecutive_late_filings=0,
        last_filing_date=datetime.now(timezone.utc) - timedelta(days=120),
    )
    result = await detector.evaluate({
        "taxpayer": taxpayer,
        "invoices": [],
        "transactions": [make_transaction(transaction_type="vat_return")],
        "now": datetime.now(timezone.utc),
    })
    assert result.detected is True
    assert "inactive_trader" in result.evidence
    assert result.risk_score > 0


@pytest.mark.asyncio
async def test_never_filed_after_registration(detector):
    """Taxpayer registered for > 30 days with zero filings should be flagged."""
    taxpayer = make_taxpayer(
        vat_registration_date=datetime.now(timezone.utc) - timedelta(days=60),
        last_filing_date=None,
        consecutive_late_filings=0,
    )
    result = await detector.evaluate({
        "taxpayer": taxpayer,
        "invoices": [],
        "transactions": [],   # No filings at all
        "now": datetime.now(timezone.utc),
    })
    assert result.detected is True
    assert "never_filed" in result.evidence


@pytest.mark.asyncio
async def test_buyer_claims_input_vat_seller_never_declared(detector):
    """Buyers claiming input VAT from this seller, but seller never declared."""
    taxpayer = make_taxpayer(tin="SELLER01")
    # Buyer has declared they received this invoice (declared_by_buyer=True)
    # but seller has NOT declared it (declared_by_seller=False)
    invoices = [
        make_invoice(
            seller_tin="SELLER01",
            buyer_tin="BUYER01",
            vat_amount=50_000.0,
            declared_by_buyer=True,
            declared_by_seller=False,
        )
        for _ in range(3)
    ]
    result = await detector.evaluate({
        "taxpayer": taxpayer,
        "invoices": invoices,
        "transactions": [make_transaction(taxpayer_tin="SELLER01", transaction_type="vat_return")],
        "now": datetime.now(timezone.utc),
    })
    assert result.detected is True
    assert "undeclared_output_vat" in result.evidence
    assert result.estimated_revenue_at_risk > 0


@pytest.mark.asyncio
async def test_consecutive_late_filings(detector):
    """Taxpayer with 5 consecutive late filings should be flagged."""
    taxpayer = make_taxpayer(consecutive_late_filings=5)
    result = await detector.evaluate({
        "taxpayer": taxpayer,
        "invoices": [],
        "transactions": [make_transaction(transaction_type="vat_return")],
        "now": datetime.now(timezone.utc),
    })
    assert result.detected is True
    assert "consecutive_late_filings" in result.evidence


@pytest.mark.asyncio
async def test_severity_scales_with_risk(detector):
    """Higher risk scores should produce higher severity alerts."""
    # Maximum risk scenario: never filed + undeclared VAT + recent high imports
    taxpayer = make_taxpayer(
        tin="MTIC01",
        vat_registration_date=datetime.now(timezone.utc) - timedelta(days=90),
        last_filing_date=datetime.now(timezone.utc) - timedelta(days=200),
        consecutive_late_filings=5,
    )
    invoices = [
        make_invoice(
            seller_tin="MTIC01",
            vat_amount=100_000.0,
            declared_by_buyer=True,
            declared_by_seller=False,
        )
        for _ in range(5)
    ]
    result = await detector.evaluate({
        "taxpayer": taxpayer,
        "invoices": invoices,
        "transactions": [],
        "now": datetime.now(timezone.utc),
    })
    assert result.detected is True
    assert result.severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL)
    assert result.risk_score >= 60

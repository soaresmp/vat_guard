"""Tests for VAT Refund Fraud detector."""

import pytest

from src.fraud_detection.rules.refund_fraud import RefundFraudDetector
from src.models.alert import FraudType
from tests.conftest import make_taxpayer, make_invoice, make_transaction


@pytest.fixture
def detector():
    return RefundFraudDetector()


@pytest.mark.asyncio
async def test_no_fraud_legitimate_exporter(detector):
    """A legitimate exporter with matching customs docs should not be flagged."""
    taxpayer = make_taxpayer(annual_turnover_reported=5_000_000.0)
    transactions = [
        make_transaction(
            transaction_type="vat_return",
            output_vat=100_000.0,
            input_vat=150_000.0,
            net_vat_payable=-50_000.0,
        ),
        make_transaction(
            transaction_type="refund_claim",
            refund_claimed=40_000.0,  # Reasonable refund < 80% of output
        ),
    ]
    export_invoices = [
        make_invoice(
            invoice_type="export",
            seller_tin="TIN123456",
            vat_amount=0.0,
            net_amount=100_000.0,
            customs_declaration_number="CUST-001",  # Has customs doc
        )
    ]
    result = await detector.evaluate({
        "taxpayer": taxpayer,
        "transactions": transactions,
        "invoices": export_invoices,
    })
    assert result.detected is False


@pytest.mark.asyncio
async def test_phantom_exporter_detected(detector):
    """Exports without customs declarations should trigger phantom exporter signal."""
    taxpayer = make_taxpayer(annual_turnover_reported=1_000_000.0)
    transactions = [
        make_transaction(transaction_type="refund_claim", refund_claimed=200_000.0),
    ]
    # All export invoices lack customs docs
    export_invoices = [
        make_invoice(
            invoice_type="export",
            seller_tin="TIN123456",
            net_amount=500_000.0,
            vat_amount=0.0,
            customs_declaration_number=None,  # No customs doc!
        )
        for _ in range(5)
    ]
    result = await detector.evaluate({
        "taxpayer": taxpayer,
        "transactions": transactions,
        "invoices": export_invoices,
    })
    assert result.detected is True
    assert "phantom_exporter" in result.evidence


@pytest.mark.asyncio
async def test_high_refund_to_output_ratio(detector):
    """Refund claiming > 80% of output VAT should be flagged."""
    taxpayer = make_taxpayer()
    transactions = [
        make_transaction(transaction_type="vat_return", output_vat=100_000.0, input_vat=50_000.0),
        make_transaction(transaction_type="refund_claim", refund_claimed=90_000.0),  # 90% of output!
    ]
    result = await detector.evaluate({
        "taxpayer": taxpayer,
        "transactions": transactions,
        "invoices": [],
    })
    assert result.detected is True
    assert "high_refund_to_output_ratio" in result.evidence


@pytest.mark.asyncio
async def test_refund_spike_detected(detector):
    """Sudden large increase in refund claim compared to history."""
    taxpayer = make_taxpayer()
    # Historical claims: small amounts, then a spike
    transactions = [
        make_transaction(transaction_type="refund_claim", refund_claimed=5_000.0),
        make_transaction(transaction_type="refund_claim", refund_claimed=6_000.0),
        make_transaction(transaction_type="refund_claim", refund_claimed=5_500.0),
        make_transaction(transaction_type="refund_claim", refund_claimed=100_000.0),  # 18× spike!
    ]
    result = await detector.evaluate({
        "taxpayer": taxpayer,
        "transactions": transactions,
        "invoices": [],
    })
    assert result.detected is True
    assert "refund_spike" in result.evidence


@pytest.mark.asyncio
async def test_frequent_amendments_detected(detector):
    """Multiple return amendments increasing refund amount should be flagged."""
    taxpayer = make_taxpayer()
    transactions = [
        make_transaction(transaction_type="amended_return", refund_claimed=10_000.0),
        make_transaction(transaction_type="amended_return", refund_claimed=15_000.0),
        make_transaction(transaction_type="amended_return", refund_claimed=20_000.0),
        make_transaction(transaction_type="amended_return", refund_claimed=25_000.0),
    ]
    result = await detector.evaluate({
        "taxpayer": taxpayer,
        "transactions": transactions,
        "invoices": [],
    })
    assert result.detected is True
    assert "frequent_amendments" in result.evidence


@pytest.mark.asyncio
async def test_estimated_revenue_at_risk_set(detector):
    """estimated_revenue_at_risk should equal total refund claimed."""
    taxpayer = make_taxpayer()
    transactions = [
        make_transaction(transaction_type="vat_return", output_vat=10_000.0),
        make_transaction(transaction_type="refund_claim", refund_claimed=9_500.0),
    ]
    result = await detector.evaluate({
        "taxpayer": taxpayer,
        "transactions": transactions,
        "invoices": [],
    })
    if result.detected:
        assert result.estimated_revenue_at_risk == 9_500.0

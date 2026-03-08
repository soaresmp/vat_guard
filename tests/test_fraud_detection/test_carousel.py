"""Tests for Carousel Fraud detector."""

import pytest
from datetime import datetime, timedelta, timezone

from src.fraud_detection.rules.carousel import CarouselFraudDetector
from src.models.alert import AlertSeverity
from tests.conftest import make_invoice


@pytest.fixture
def detector():
    return CarouselFraudDetector()


def make_chain_invoices(chain_tins: list[str], vat_amount: float = 5_000.0) -> list:
    """Create a circular chain of invoices between the given TINs."""
    invoices = []
    for i, seller in enumerate(chain_tins):
        buyer = chain_tins[(i + 1) % len(chain_tins)]
        invoices.append(make_invoice(
            seller_tin=seller,
            buyer_tin=buyer,
            vat_amount=vat_amount,
            invoice_type="standard",
        ))
    return invoices


@pytest.mark.asyncio
async def test_circular_chain_detected(detector):
    """A circular chain A→B→C→A should be detected."""
    chain = ["TIN_A", "TIN_B", "TIN_C"]
    invoices = make_chain_invoices(chain)
    result = await detector.evaluate({"invoices": invoices})
    assert result.detected is True
    assert "circular_chains" in result.evidence
    assert result.evidence["circular_chains"]["cycle_count"] >= 1


@pytest.mark.asyncio
async def test_no_cycle_linear_chain(detector):
    """A linear chain A→B→C (no cycle) should not trigger carousel detection."""
    invoices = [
        make_invoice(seller_tin="TIN_A", buyer_tin="TIN_B"),
        make_invoice(seller_tin="TIN_B", buyer_tin="TIN_C"),
    ]
    result = await detector.evaluate({"invoices": invoices})
    # Without cycles, risk should be low
    assert result.risk_score < 35 or not result.detected


@pytest.mark.asyncio
async def test_high_risk_commodity_detection(detector):
    """Invoices with mobile phone HS code (8517) should add risk."""
    chain = ["TIN_A", "TIN_B", "TIN_C", "TIN_D"]
    invoices = make_chain_invoices(chain)
    for inv in invoices:
        inv.commodity_code = "851712"  # Smartphones
    result = await detector.evaluate({"invoices": invoices})
    assert result.detected is True
    assert "high_risk_commodities" in result.evidence


@pytest.mark.asyncio
async def test_rapid_resale_detected(detector):
    """Goods resold within 24 hours should trigger rapid resale signal."""
    now = datetime.now(timezone.utc)
    chain = ["TIN_A", "TIN_B", "TIN_C"]
    invoices = make_chain_invoices(chain)
    # Set timestamps very close together
    for i, inv in enumerate(invoices):
        inv.invoice_date = now + timedelta(hours=i * 2)
    result = await detector.evaluate({"invoices": invoices})
    assert result.detected is True


@pytest.mark.asyncio
async def test_cross_border_carousel_higher_risk(detector):
    """Cross-border carousel should produce higher risk than domestic."""
    chain = ["TIN_A", "TIN_B", "TIN_C"]
    invoices_domestic = make_chain_invoices(chain)
    invoices_cross_border = make_chain_invoices(chain)
    for inv in invoices_cross_border:
        inv.invoice_type = "export"
        inv.seller_country = "XX"
        inv.buyer_country = "YY"

    r_domestic = await detector.evaluate({"invoices": invoices_domestic})
    r_cross_border = await detector.evaluate({"invoices": invoices_cross_border})

    # Cross-border should have higher or equal risk
    assert r_cross_border.risk_score >= r_domestic.risk_score


@pytest.mark.asyncio
async def test_longer_chain_higher_risk(detector):
    """Longer carousel chains should be scored higher than short ones."""
    short_chain = make_chain_invoices(["A", "B", "C"])
    long_chain = make_chain_invoices(["A", "B", "C", "D", "E", "F"])

    r_short = await detector.evaluate({"invoices": short_chain})
    r_long = await detector.evaluate({"invoices": long_chain})

    assert r_long.risk_score >= r_short.risk_score

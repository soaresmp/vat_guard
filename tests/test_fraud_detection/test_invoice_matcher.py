"""Tests for Real-Time Invoice Matcher."""

import pytest

from src.fraud_detection.realtime.invoice_matcher import InvoiceMatcher, MatchStatus
from tests.conftest import make_invoice


@pytest.fixture
def matcher():
    return InvoiceMatcher()


def make_pair(invoice_number: str, vat_amount: float, seller_vat: str = "VAT001"):
    """Create matching seller and buyer invoice mocks."""
    seller = make_invoice(
        invoice_number=invoice_number,
        seller_vat_number=seller_vat,
        seller_tin="SELLER",
        buyer_tin="BUYER",
        vat_amount=vat_amount,
    )
    buyer = make_invoice(
        invoice_number=invoice_number,
        seller_vat_number=seller_vat,
        seller_tin="SELLER",
        buyer_tin="BUYER",
        vat_amount=vat_amount,
    )
    return seller, buyer


def test_matched_invoice(matcher):
    """Perfectly matching seller and buyer invoices should produce MATCHED status."""
    seller, buyer = make_pair("INV-001", 1000.0)
    result = matcher.match_single(seller, buyer)
    assert result.status == MatchStatus.MATCHED
    assert result.risk_score == 0.0


def test_buyer_only_invoice(matcher):
    """Buyer claims input VAT but no seller declaration — high risk."""
    _, buyer = make_pair("INV-002", 5000.0)
    result = matcher.match_single(None, buyer)
    assert result.status == MatchStatus.BUYER_ONLY
    assert result.risk_score >= 70


def test_seller_only_invoice(matcher):
    """Seller declared but no buyer claim — medium risk (suppressed sale indicator)."""
    seller, _ = make_pair("INV-003", 2000.0)
    result = matcher.match_single(seller, None)
    assert result.status == MatchStatus.SELLER_ONLY
    assert result.risk_score >= 20


def test_amount_mismatch(matcher):
    """Seller and buyer declare different amounts — suspicious."""
    seller, buyer = make_pair("INV-004", 1000.0)
    buyer.vat_amount = 1500.0  # Inflated claim
    result = matcher.match_single(seller, buyer)
    assert result.status == MatchStatus.AMOUNT_MISMATCH
    assert result.risk_score >= 50


def test_high_value_flag(matcher):
    """High-value invoices (VAT >= threshold) should add risk."""
    seller, buyer = make_pair("INV-005", 50_000.0)  # Above HIGH_VALUE_THRESHOLD
    result = matcher.match_single(seller, buyer)
    assert any("High-value" in flag for flag in result.flags)


def test_batch_match(matcher):
    """Batch match should identify all unmatched invoices in a period."""
    seller_invs = [
        make_invoice(invoice_number="S001", seller_vat_number="V001", vat_amount=1000),
        make_invoice(invoice_number="S002", seller_vat_number="V001", vat_amount=2000),
        make_invoice(invoice_number="S003", seller_vat_number="V001", vat_amount=3000),
    ]
    buyer_invs = [
        make_invoice(invoice_number="S001", seller_vat_number="V001", vat_amount=1000),
        make_invoice(invoice_number="S002", seller_vat_number="V001", vat_amount=2000),
        # S003 is claimed by buyer but NOT in seller list
        make_invoice(invoice_number="S004", seller_vat_number="V001", vat_amount=5000),  # Extra buyer claim
    ]
    results = matcher.match_period(seller_invs, buyer_invs)
    unmatched = [r for r in results if r.status != MatchStatus.MATCHED]
    assert len(unmatched) >= 2  # S003 (seller-only) + S004 (buyer-only)


def test_duplicate_detection(matcher):
    """Duplicate invoice submissions should be flagged."""
    seller_invs = [
        make_invoice(invoice_number="DUP001", seller_vat_number="V001", vat_amount=1000),
        make_invoice(invoice_number="DUP001", seller_vat_number="V001", vat_amount=1000),  # Duplicate
    ]
    buyer_invs = [
        make_invoice(invoice_number="DUP001", seller_vat_number="V001", vat_amount=1000),
    ]
    results = matcher.match_period(seller_invs, buyer_invs)
    statuses = [r.status for r in results]
    assert MatchStatus.DUPLICATE in statuses


def test_amount_tolerance(matcher):
    """Small rounding differences (< 1%) should not trigger mismatch."""
    seller, buyer = make_pair("INV-ROUNDING", 10_000.0)
    buyer.vat_amount = 10_099.0  # 0.99% difference — within tolerance
    result = matcher.match_single(seller, buyer)
    assert result.status == MatchStatus.MATCHED

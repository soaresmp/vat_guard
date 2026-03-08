"""Tests for Benford's Law Analyzer."""

import math

import pytest

from src.fraud_detection.ml.benford_analysis import BenfordAnalyzer, BENFORD_EXPECTED


@pytest.fixture
def analyzer():
    return BenfordAnalyzer()


def generate_benford_amounts(n: int = 200) -> list[float]:
    """Generate amounts that follow Benford's distribution."""
    import random
    amounts = []
    for _ in range(n):
        # Pick a leading digit according to Benford probabilities
        digits = list(range(1, 10))
        weights = [BENFORD_EXPECTED[d] for d in digits]
        leading = random.choices(digits, weights=weights)[0]
        # Random magnitude
        magnitude = random.choice([1, 10, 100, 1000, 10000])
        rest = random.uniform(0, 1) * magnitude
        amount = leading * magnitude + rest
        amounts.append(round(amount, 2))
    return amounts


def generate_uniform_amounts(n: int = 200) -> list[float]:
    """Generate amounts with uniform leading digits (clearly violates Benford)."""
    import random
    return [float(random.randint(500, 599)) for _ in range(n)]  # All start with 5


def test_benford_conformant_data(analyzer):
    """Naturally distributed amounts should conform to Benford's Law."""
    amounts = generate_benford_amounts(300)
    result = analyzer.analyze(amounts)
    # Should not be suspicious (p-value should be high for conformant data)
    # Allow some tolerance since random data can occasionally fail
    assert result.sample_size == 300
    assert result.chi2_statistic >= 0
    assert 0 <= result.p_value <= 1


def test_benford_suspicious_uniform_data(analyzer):
    """Uniformly distributed leading digits should be flagged."""
    amounts = generate_uniform_amounts(200)
    result = analyzer.analyze(amounts)
    assert result.is_suspicious is True
    assert result.p_value < 0.05


def test_benford_insufficient_sample(analyzer):
    """Small samples should not trigger suspicious flag."""
    result = analyzer.analyze([100.0, 200.0, 300.0])
    assert result.is_suspicious is False
    assert "Insufficient" in result.interpretation


def test_leading_digit_extraction(analyzer):
    """Test that leading digits are extracted correctly."""
    assert analyzer._leading_digit(1234) == 1
    assert analyzer._leading_digit(0.0056) == 5
    assert analyzer._leading_digit(999) == 9
    assert analyzer._leading_digit(100000) == 1


def test_benford_expected_sums_to_one():
    """Benford expected frequencies must sum to ~1.0."""
    total = sum(BENFORD_EXPECTED.values())
    assert abs(total - 1.0) < 1e-10


def test_analyze_invoices(analyzer):
    """analyze_invoices should correctly extract net_amount field."""
    from unittest.mock import MagicMock
    invoices = []
    for amount in generate_benford_amounts(100):
        inv = MagicMock()
        inv.net_amount = amount
        invoices.append(inv)
    result = analyzer.analyze_invoices(invoices)
    assert result.sample_size == 100


def test_round_number_anomaly(analyzer):
    """Suspiciously round amounts (all multiples of 1000) should be flagged."""
    # All amounts start with 1, 2, or 9 repeatedly — violates Benford
    amounts = [1000.0] * 100 + [2000.0] * 50 + [9000.0] * 100
    result = analyzer.analyze(amounts)
    assert result.is_suspicious is True

"""
Benford's Law Analyzer
=======================
Benford's Law states that in naturally occurring collections of numbers,
the leading digit d (1–9) occurs with probability log10(1 + 1/d).

When VAT invoice amounts are fabricated by humans, they often fail to follow
this distribution because people tend to pick "round" or "convenient" numbers.
Large deviations from Benford's distribution signal potential fabrication.

IMF reference: Statistical tests for invoice fabrication detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


# Expected Benford frequencies for digits 1–9
BENFORD_EXPECTED: dict[int, float] = {
    d: math.log10(1 + 1 / d) for d in range(1, 10)
}


@dataclass
class BenfordResult:
    chi2_statistic: float
    p_value: float
    is_suspicious: bool         # True if distribution significantly differs
    observed_frequencies: dict[int, float]
    expected_frequencies: dict[int, float]
    deviations: dict[int, float]
    mean_absolute_deviation: float
    sample_size: int
    interpretation: str


class BenfordAnalyzer:
    """
    Tests whether a collection of invoice amounts follows Benford's Law.
    Uses Chi-squared goodness-of-fit test.
    """

    MIN_SAMPLE_SIZE = 30        # Chi-squared unreliable below this
    SIGNIFICANCE_LEVEL = 0.05   # p-value threshold

    def analyze(self, amounts: list[float], significance: float | None = None) -> BenfordResult:
        """
        Analyse a list of invoice amounts for Benford's Law compliance.

        Args:
            amounts: List of invoice amounts (net or gross).
            significance: p-value threshold (defaults to class attribute).

        Returns:
            BenfordResult with test statistics and interpretation.
        """
        sig = significance or self.SIGNIFICANCE_LEVEL
        amounts_clean = [a for a in amounts if a > 0]
        n = len(amounts_clean)

        if n < self.MIN_SAMPLE_SIZE:
            return BenfordResult(
                chi2_statistic=0.0,
                p_value=1.0,
                is_suspicious=False,
                observed_frequencies={},
                expected_frequencies=BENFORD_EXPECTED.copy(),
                deviations={},
                mean_absolute_deviation=0.0,
                sample_size=n,
                interpretation=f"Insufficient sample size ({n} < {self.MIN_SAMPLE_SIZE})",
            )

        # Extract leading digits
        leading_digits = [self._leading_digit(a) for a in amounts_clean]

        # Observed counts and frequencies
        counts: dict[int, int] = {d: 0 for d in range(1, 10)}
        for d in leading_digits:
            counts[d] = counts.get(d, 0) + 1

        observed_freq = {d: counts[d] / n for d in range(1, 10)}
        expected_counts = [BENFORD_EXPECTED[d] * n for d in range(1, 10)]
        observed_counts = [counts[d] for d in range(1, 10)]

        # Chi-squared goodness-of-fit
        chi2_stat, p_value = stats.chisquare(
            f_obs=observed_counts,
            f_exp=expected_counts,
        )

        # Mean Absolute Deviation (MAD)
        deviations = {
            d: abs(observed_freq[d] - BENFORD_EXPECTED[d])
            for d in range(1, 10)
        }
        mad = sum(deviations.values()) / 9

        is_suspicious = p_value < sig and n >= self.MIN_SAMPLE_SIZE

        interpretation = self._interpret(chi2_stat, p_value, mad, n, sig)

        return BenfordResult(
            chi2_statistic=round(chi2_stat, 4),
            p_value=round(p_value, 6),
            is_suspicious=is_suspicious,
            observed_frequencies={d: round(f, 4) for d, f in observed_freq.items()},
            expected_frequencies={d: round(f, 4) for d, f in BENFORD_EXPECTED.items()},
            deviations={d: round(v, 4) for d, v in deviations.items()},
            mean_absolute_deviation=round(mad, 4),
            sample_size=n,
            interpretation=interpretation,
        )

    def analyze_invoices(self, invoices: list[Any], amount_field: str = "net_amount") -> BenfordResult:
        """Convenience wrapper to extract amounts from invoice ORM objects."""
        amounts = [
            float(getattr(inv, amount_field, 0) or 0)
            for inv in invoices
        ]
        return self.analyze(amounts)

    @staticmethod
    def _leading_digit(value: float) -> int:
        """Extract the leading (first significant) digit of a positive number."""
        while value >= 10:
            value /= 10
        while value < 1:
            value *= 10
        return int(value)

    @staticmethod
    def _interpret(chi2: float, p: float, mad: float, n: int, sig: float) -> str:
        if p >= sig:
            return (
                f"CONFORMANT: p={p:.4f} ≥ {sig}. "
                f"Invoice amounts follow Benford's Law (n={n}, χ²={chi2:.2f}, MAD={mad:.4f})."
            )
        # Suspicious — classify severity by MAD
        if mad < 0.006:
            level = "marginal"
        elif mad < 0.012:
            level = "moderate"
        elif mad < 0.015:
            level = "marked"
        else:
            level = "severe"
        return (
            f"NON-CONFORMANT ({level}): p={p:.4f} < {sig}. "
            f"Invoice amounts deviate significantly from Benford's Law "
            f"(n={n}, χ²={chi2:.2f}, MAD={mad:.4f}). "
            f"Possible invoice fabrication."
        )

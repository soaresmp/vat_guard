"""
Anomaly Detector
================
Uses Isolation Forest (unsupervised) + Z-score statistical methods
to identify outlier taxpayer behaviour and transaction patterns.

Statistical approach recommended by IMF 2023 How-To Note.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    is_anomaly: bool
    anomaly_score: float     # 0–100, higher = more anomalous
    z_scores: dict[str, float]
    isolation_score: float   # Raw Isolation Forest score (-1 to 1)
    features_used: list[str]
    explanation: str


class AnomalyDetector:
    """
    Detects anomalous taxpayer or invoice patterns using:
      - Isolation Forest for multi-dimensional outlier detection
      - Z-score analysis for individual feature deviation
    """

    ANOMALY_THRESHOLD = -0.1          # Isolation Forest: < this value = anomaly
    Z_SCORE_THRESHOLD = 3.0           # Standard deviations from mean
    MIN_SAMPLES_FOR_MODEL = 10        # Min records needed to fit model

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self._model: IsolationForest | None = None
        self._scaler: StandardScaler = StandardScaler()
        self._is_fitted = False
        self._feature_names: list[str] = []

    def fit(self, feature_matrix: np.ndarray, feature_names: list[str]) -> None:
        """Train the Isolation Forest model on a population of taxpayer features."""
        if len(feature_matrix) < self.MIN_SAMPLES_FOR_MODEL:
            logger.warning(
                "Insufficient samples to fit anomaly model: %d < %d",
                len(feature_matrix),
                self.MIN_SAMPLES_FOR_MODEL,
            )
            return

        self._feature_names = feature_names
        scaled = self._scaler.fit_transform(feature_matrix)
        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=200,
            random_state=42,
        )
        self._model.fit(scaled)
        self._is_fitted = True
        logger.info("Anomaly model fitted on %d samples", len(feature_matrix))

    def detect(self, features: dict[str, float]) -> AnomalyResult:
        """Evaluate a single entity's features for anomalies."""
        feature_vector = np.array([features.get(f, 0.0) for f in self._feature_names])

        # ── Z-score analysis (always available) ──────────────────────────────
        z_scores: dict[str, float] = {}
        if self._is_fitted:
            # Use scaler mean/std for z-scores
            means = self._scaler.mean_
            stds = self._scaler.scale_
            for i, name in enumerate(self._feature_names):
                z = (feature_vector[i] - means[i]) / (stds[i] + 1e-9)
                z_scores[name] = round(float(z), 3)

        high_z_features = {k: v for k, v in z_scores.items() if abs(v) > self.Z_SCORE_THRESHOLD}

        # ── Isolation Forest score ────────────────────────────────────────────
        iso_score = 0.0
        is_anomaly = bool(high_z_features)

        if self._is_fitted and self._model is not None:
            scaled_vec = self._scaler.transform(feature_vector.reshape(1, -1))
            iso_score = float(self._model.decision_function(scaled_vec)[0])
            if iso_score < self.ANOMALY_THRESHOLD:
                is_anomaly = True

        # ── Combined anomaly score (0–100) ────────────────────────────────────
        # Normalise Isolation Forest score: typical range −0.5 to 0.5
        iso_normalised = max(0.0, min(100.0, (-iso_score + 0.5) * 100))
        z_score_component = min(100.0, sum(abs(v) for v in z_scores.values()) * 5)
        anomaly_score = (iso_normalised * 0.6 + z_score_component * 0.4)
        anomaly_score = round(min(100.0, anomaly_score), 2)

        explanation_parts = []
        if iso_score < self.ANOMALY_THRESHOLD:
            explanation_parts.append(
                f"Isolation Forest score {iso_score:.3f} below threshold {self.ANOMALY_THRESHOLD}"
            )
        if high_z_features:
            top = sorted(high_z_features.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            explanation_parts.append(
                "High z-score features: " + ", ".join(f"{k}={v:.1f}σ" for k, v in top)
            )

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            z_scores=z_scores,
            isolation_score=iso_score,
            features_used=self._feature_names,
            explanation="; ".join(explanation_parts) or "No anomaly detected",
        )

    # ── Feature extraction helpers ────────────────────────────────────────────

    @staticmethod
    def extract_taxpayer_features(taxpayer: Any, transactions: list) -> dict[str, float]:
        """Extract numeric features from a Taxpayer ORM object."""
        vat_returns = [t for t in transactions if t.transaction_type == "vat_return"]
        refund_claims = [t for t in transactions if t.transaction_type == "refund_claim"]

        total_output = sum(getattr(t, "output_vat", 0) or 0 for t in vat_returns)
        total_input = sum(getattr(t, "input_vat", 0) or 0 for t in vat_returns)
        total_refund = sum(getattr(t, "refund_claimed", 0) or 0 for t in refund_claims)
        late_filings = sum(1 for t in vat_returns if getattr(t, "is_late", False))

        return {
            "risk_score": getattr(taxpayer, "risk_score", 0.0) or 0.0,
            "compliance_score": getattr(taxpayer, "compliance_score", 100.0) or 100.0,
            "consecutive_late_filings": float(getattr(taxpayer, "consecutive_late_filings", 0)),
            "total_alerts": float(getattr(taxpayer, "total_alerts", 0)),
            "annual_turnover": float(getattr(taxpayer, "annual_turnover_reported", 0) or 0),
            "annual_vat_declared": float(getattr(taxpayer, "annual_vat_declared", 0) or 0),
            "annual_vat_paid": float(getattr(taxpayer, "annual_vat_paid", 0) or 0),
            "total_output_vat": total_output,
            "total_input_vat": total_input,
            "input_output_ratio": total_input / max(total_output, 1),
            "total_refund_claimed": total_refund,
            "refund_output_ratio": total_refund / max(total_output, 1),
            "late_filing_ratio": late_filings / max(len(vat_returns), 1),
            "is_newly_registered": float(getattr(taxpayer, "is_newly_registered", False)),
            "has_foreign_directors": float(getattr(taxpayer, "has_foreign_directors", False)),
        }

    @staticmethod
    def extract_invoice_features(invoice: Any) -> dict[str, float]:
        """Extract numeric features from an Invoice ORM object."""
        return {
            "net_amount": float(getattr(invoice, "net_amount", 0) or 0),
            "vat_amount": float(getattr(invoice, "vat_amount", 0) or 0),
            "vat_rate": float(getattr(invoice, "vat_rate", 0) or 0),
            "gross_amount": float(getattr(invoice, "gross_amount", 0) or 0),
            "is_cross_border": float(
                getattr(invoice, "invoice_type", "") in ("export", "import", "intra_community")
            ),
            "has_customs_doc": float(
                bool(getattr(invoice, "customs_declaration_number", None))
            ),
            "declared_by_both": float(
                getattr(invoice, "declared_by_seller", False)
                and getattr(invoice, "declared_by_buyer", False)
            ),
        }

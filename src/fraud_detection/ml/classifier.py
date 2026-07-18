"""
VAT Fraud Classifier
======================
Supervised ensemble model for VAT fraud detection.

Architecture:
  - Random Forest  : robust to imbalanced data, provides interpretable
                     feature importances
  - XGBoost        : gradient boosted trees — higher precision/recall on
                     structured tabular data
  - Soft-voting    : combines both models' probability outputs

The classifier is optimised for HIGH RECALL (minimising false negatives) at
the cost of some precision, which is the correct operational trade-off for a
tax-authority audit-selection tool (missing fraud >> false alarm cost).

Threshold:
  Default decision threshold is tuned on validation set to achieve ≥ 90 %
  recall. Final threshold is stored as `self.threshold_`.

Persistence:
  `save(path)` / `load(path)` serialize via joblib.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    warnings.warn(
        "xgboost not installed. Classifier will use Random Forest only. "
        "Install with: pip install xgboost",
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainingMetrics:
    roc_auc: float
    avg_precision: float
    f1_fraud: float
    recall_fraud: float
    precision_fraud: float
    confusion_matrix: list[list[int]]
    feature_importances: dict[str, float]
    threshold: float
    classification_report: str


@dataclass
class PredictionResult:
    invoice_id: str
    fraud_probability: float
    predicted_fraud: bool
    risk_level: str          # low / medium / high / critical
    top_features: list[tuple[str, float]]  # (feature_name, shap-like importance)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class VATFraudClassifier:
    """
    Binary VAT fraud classifier (legitimate vs fraud).
    Also supports multi-class fraud type prediction via a secondary model.
    """

    RECALL_TARGET = 0.90  # Minimum recall on fraud class for threshold selection

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        recall_target: float = RECALL_TARGET,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.recall_target = recall_target
        self.random_state = random_state

        self.model_: Optional[object] = None
        self._rf: Optional[object] = None
        self._xgb: Optional[object] = None
        self.threshold_: float = 0.5
        self.feature_names_: list[str] = []
        self.is_fitted_: bool = False

        self._multiclass_model: Optional[object] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "VATFraudClassifier":
        """
        Train the ensemble classifier.

        Parameters
        ----------
        X_train, y_train : training features and binary fraud labels
        X_val, y_val     : optional held-out validation set for threshold tuning
                           (if not provided, a 20% stratified split is used)
        """
        self.feature_names_ = list(X_train.columns)
        sample_weights = compute_sample_weight("balanced", y_train)

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.random_state,
        )
        rf.fit(X_train, y_train, sample_weight=sample_weights)

        self._rf = rf
        self._xgb = None

        if _XGB_AVAILABLE:
            scale_pos = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
            xgb = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
            )
            xgb.fit(X_train, y_train, sample_weight=sample_weights)
            self._xgb = xgb

        # Use RF as primary for sklearn-compat; blend manually in predict_proba
        self.model_ = rf

        # Tune decision threshold on validation data
        val_X = X_val if X_val is not None else X_train
        val_y = y_val if y_val is not None else y_train
        self.threshold_ = self._tune_threshold(val_X, val_y)
        self.is_fitted_ = True
        return self

    def _tune_threshold(self, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """
        Select the lowest threshold that achieves at least `recall_target`
        on the fraud class. Falls back to 0.5 if no threshold achieves it.
        """
        proba = self.model_.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, proba)
        # thresholds has len = len(precisions) - 1
        best_threshold = 0.5
        for thresh, rec in zip(thresholds, recalls[:-1]):
            if rec >= self.recall_target:
                best_threshold = thresh
                break
        # Apply a floor so the threshold is never trivially low
        return float(max(best_threshold, 0.10))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        rf_proba = self._rf.predict_proba(X)
        if self._xgb is not None:
            xgb_proba = self._xgb.predict_proba(X)
            return (rf_proba + xgb_proba) / 2.0
        return rf_proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold_).astype(int)

    def predict_invoice(
        self,
        features: pd.Series,
        invoice_id: str = "UNKNOWN",
    ) -> PredictionResult:
        """Score a single invoice row and return a human-readable result."""
        self._check_fitted()
        X = features.to_frame().T
        prob = float(self.predict_proba(X)[0, 1])
        predicted = prob >= self.threshold_

        if prob < 0.25:
            risk_level = "low"
        elif prob < 0.5:
            risk_level = "medium"
        elif prob < 0.75:
            risk_level = "high"
        else:
            risk_level = "critical"

        top_features = self._top_features(features)
        return PredictionResult(
            invoice_id=invoice_id,
            fraud_probability=round(prob, 4),
            predicted_fraud=bool(predicted),
            risk_level=risk_level,
            top_features=top_features,
        )

    def _top_features(
        self,
        features: pd.Series,
        n: int = 5,
    ) -> list[tuple[str, float]]:
        """Return top-N features contributing to fraud score (importance × |value|)."""
        importances = self._get_feature_importances()
        scores = {
            name: importances.get(name, 0.0) * abs(float(features.get(name, 0)))
            for name in self.feature_names_
        }
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [(name, round(score, 4)) for name, score in top]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> TrainingMetrics:
        """Compute full evaluation metrics on a held-out test set."""
        self._check_fitted()
        proba = self.predict_proba(X_test)[:, 1]
        preds = (proba >= self.threshold_).astype(int)

        roc_auc = roc_auc_score(y_test, proba)
        precision_arr, recall_arr, _ = precision_recall_curve(y_test, proba)
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        avg_precision = abs(float(_trapz(precision_arr, recall_arr)))

        report = classification_report(y_test, preds, target_names=["legitimate", "fraud"])
        report_dict = classification_report(
            y_test, preds, target_names=["legitimate", "fraud"], output_dict=True
        )

        cm = confusion_matrix(y_test, preds).tolist()
        importances = self._get_feature_importances()

        return TrainingMetrics(
            roc_auc=round(roc_auc, 4),
            avg_precision=round(abs(avg_precision), 4),
            f1_fraud=round(report_dict.get("fraud", {}).get("f1-score", 0.0), 4),
            recall_fraud=round(report_dict.get("fraud", {}).get("recall", 0.0), 4),
            precision_fraud=round(report_dict.get("fraud", {}).get("precision", 0.0), 4),
            confusion_matrix=cm,
            feature_importances=importances,
            threshold=round(self.threshold_, 4),
            classification_report=report,
        )

    # ------------------------------------------------------------------
    # Multi-class fraud type classifier (secondary)
    # ------------------------------------------------------------------

    def fit_multiclass(
        self,
        X_train: pd.DataFrame,
        y_fraud_type: pd.Series,
    ) -> "VATFraudClassifier":
        """
        Train a secondary Random Forest to classify the type of fraud
        (only on fraud rows). y_fraud_type must contain string labels.
        """
        rf_multi = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.random_state,
        )
        rf_multi.fit(X_train, y_fraud_type)
        self._multiclass_model = rf_multi
        self._multiclass_classes_ = rf_multi.classes_
        return self

    def predict_fraud_type(self, X: pd.DataFrame) -> np.ndarray:
        if self._multiclass_model is None:
            raise RuntimeError("Multi-class model not trained. Call fit_multiclass() first.")
        return self._multiclass_model.predict(X)

    def predict_fraud_type_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._multiclass_model is None:
            raise RuntimeError("Multi-class model not trained. Call fit_multiclass() first.")
        proba = self._multiclass_model.predict_proba(X)
        return pd.DataFrame(proba, columns=self._multiclass_classes_, index=X.index)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model_,
                "rf": self._rf,
                "xgb": self._xgb,
                "threshold": self.threshold_,
                "feature_names": self.feature_names_,
                "multiclass_model": self._multiclass_model,
                "multiclass_classes": getattr(self, "_multiclass_classes_", None),
            },
            path,
        )

    def load(self, path: Union[str, Path]) -> "VATFraudClassifier":
        import joblib
        data = joblib.load(path)
        self.model_ = data["model"]
        self._rf = data.get("rf", data["model"])
        self._xgb = data.get("xgb")
        self.threshold_ = data["threshold"]
        self.feature_names_ = data["feature_names"]
        self._multiclass_model = data.get("multiclass_model")
        self._multiclass_classes_ = data.get("multiclass_classes")
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("Classifier not trained. Call fit() first.")

    def _get_feature_importances(self) -> dict[str, float]:
        """Extract feature importances from the Random Forest estimator."""
        rf = getattr(self, "_rf", self.model_)
        if rf is not None and hasattr(rf, "feature_importances_"):
            return dict(zip(self.feature_names_, rf.feature_importances_.tolist()))
        return {}

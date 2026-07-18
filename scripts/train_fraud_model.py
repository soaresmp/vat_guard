#!/usr/bin/env python3
"""
VAT Fraud Detection Model — Training Pipeline
===============================================
End-to-end script that:
  1. Generates synthetic labelled invoice data (IMF fraud taxonomy)
  2. Engineers features from the raw data
  3. Trains a Random Forest + XGBoost ensemble classifier
  4. Trains a secondary multi-class fraud type classifier
  5. Evaluates both models and prints a full report
  6. Saves the trained models to models/

Usage:
  python scripts/train_fraud_model.py [--output-dir models/] [--seed 42]

Requirements:
  pip install scikit-learn xgboost pandas numpy scipy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure src/ is on the path when running from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util as _ilu
import types


def _stub_package(dotted: str, path: str | None = None) -> types.ModuleType:
    """Register a lightweight stub module in sys.modules."""
    if dotted in sys.modules:
        return sys.modules[dotted]
    mod = types.ModuleType(dotted)
    mod.__package__ = dotted
    if path is not None:
        mod.__path__ = [path]
        mod.__file__ = path + "/__init__.py"
    sys.modules[dotted] = mod
    return mod


def _import_file(dotted: str, file_path: Path) -> types.ModuleType:
    """Import a single .py file under a given dotted module name."""
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = _ilu.spec_from_file_location(dotted, file_path)
    mod = _ilu.module_from_spec(spec)
    mod.__package__ = dotted.rsplit(".", 1)[0]
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


import enum

# ---- Stub the heavy platform packages that the ML files don't actually use ---
_src = _stub_package("src", str(PROJECT_ROOT / "src"))
_fd = _stub_package("src.fraud_detection", str(PROJECT_ROOT / "src/fraud_detection"))
_stub_package("src.fraud_detection.engine")
_stub_package("src.fraud_detection.ml", str(PROJECT_ROOT / "src/fraud_detection/ml"))
_stub_package("src.models", str(PROJECT_ROOT / "src/models"))
_stub_package("src.config")

_models_alert = _stub_package("src.models.alert")


class _FraudType(str, enum.Enum):
    OTHER = "other"; MISSING_TRADER = "missing_trader"; CAROUSEL_FRAUD = "carousel_fraud"
    INVOICE_MILL = "invoice_mill"; REFUND_FRAUD = "refund_fraud"
    CONTRA_TRADING = "contra_trading"; SUPPRESSED_SALES = "suppressed_sales"


class _AlertSeverity(str, enum.Enum):
    LOW = "low"; MEDIUM = "medium"; HIGH = "high"; CRITICAL = "critical"


_models_alert.FraudType = _FraudType
_models_alert.AlertSeverity = _AlertSeverity

# ---- Now import the ML files directly ---
_dg = _import_file(
    "src.fraud_detection.ml.data_generator",
    PROJECT_ROOT / "src/fraud_detection/ml/data_generator.py",
)
_fe = _import_file(
    "src.fraud_detection.ml.feature_engineer",
    PROJECT_ROOT / "src/fraud_detection/ml/feature_engineer.py",
)
_cl = _import_file(
    "src.fraud_detection.ml.classifier",
    PROJECT_ROOT / "src/fraud_detection/ml/classifier.py",
)

generate_dataset = _dg.generate_dataset
engineer_features = _fe.engineer_features
VATFraudClassifier = _cl.VATFraudClassifier


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VATGuard fraud detection models")
    p.add_argument("--output-dir", default="models", help="Directory to save trained models")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--n-legitimate", type=int, default=5000)
    p.add_argument("--n-missing-trader", type=int, default=600)
    p.add_argument("--n-carousel", type=int, default=700)
    p.add_argument("--n-invoice-mill", type=int, default=800)
    p.add_argument("--n-refund-fraud", type=int, default=600)
    p.add_argument("--n-contra-trading", type=int, default=500)
    p.add_argument("--n-suppressed-sales", type=int, default=800)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "-", width: int = 70) -> str:
    return char * width


def _print_section(title: str) -> None:
    print(f"\n{_sep('=')}")
    print(f"  {title}")
    print(_sep("="))


def _print_fraud_distribution(df: pd.DataFrame) -> None:
    dist = df["fraud_type"].value_counts()
    total = len(df)
    print(f"\n{'Fraud type':<30} {'Count':>8}  {'%':>6}")
    print(_sep())
    for ft, cnt in dist.items():
        print(f"  {ft:<28} {cnt:>8}  {cnt / total * 100:>5.1f}%")
    print(_sep())
    fraud = (df["is_fraud"] == 1).sum()
    legit = (df["is_fraud"] == 0).sum()
    print(f"  {'Total fraud':<28} {fraud:>8}  {fraud / total * 100:>5.1f}%")
    print(f"  {'Total legitimate':<28} {legit:>8}  {legit / total * 100:>5.1f}%")
    print(f"  {'TOTAL':<28} {total:>8}")


def _print_feature_importances(importances: dict[str, float], top_n: int = 15) -> None:
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    max_val = sorted_imp[0][1] if sorted_imp else 1.0
    print(f"\n{'Feature':<40} {'Importance':>10}  Bar")
    print(_sep())
    for name, val in sorted_imp:
        bar = "█" * int(val / max_val * 30)
        print(f"  {name:<38} {val:>10.4f}  {bar}")


def _print_confusion_matrix(cm: list[list[int]]) -> None:
    print("\nConfusion matrix:")
    print(f"                  Predicted Legit  Predicted Fraud")
    print(f"  Actual Legit    {cm[0][0]:>15}  {cm[0][1]:>15}")
    print(f"  Actual Fraud    {cm[1][0]:>15}  {cm[1][1]:>15}")
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total_fraud = fn + tp
    print(f"\n  True Positives  : {tp:>6}  ({tp / max(total_fraud, 1) * 100:.1f}% of all fraud cases caught)")
    print(f"  False Negatives : {fn:>6}  ({fn / max(total_fraud, 1) * 100:.1f}% of fraud missed)")
    print(f"  False Positives : {fp:>6}  ({fp / max(tn + fp, 1) * 100:.1f}% of legitimate flagged)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Generate data
    # ------------------------------------------------------------------
    _print_section("1. Generating synthetic invoice dataset")
    t0 = time.time()
    df = generate_dataset(
        n_legitimate=args.n_legitimate,
        n_missing_trader=args.n_missing_trader,
        n_carousel=args.n_carousel,
        n_invoice_mill=args.n_invoice_mill,
        n_refund_fraud=args.n_refund_fraud,
        n_contra_trading=args.n_contra_trading,
        n_suppressed_sales=args.n_suppressed_sales,
    )
    print(f"  Generated {len(df):,} invoice records in {time.time() - t0:.1f}s")
    _print_fraud_distribution(df)

    # Save dataset (CSV fallback if parquet engine unavailable)
    try:
        dataset_path = output_dir / "synthetic_invoices.parquet"
        df.to_parquet(dataset_path, index=False)
    except ImportError:
        dataset_path = output_dir / "synthetic_invoices.csv.gz"
        df.to_csv(dataset_path, index=False, compression="gzip")
    print(f"\n  Dataset saved → {dataset_path}")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    _print_section("2. Feature engineering")
    t0 = time.time()
    X = engineer_features(df)
    y = df["is_fraud"].astype(int)
    y_type = df["fraud_type"]

    print(f"  Feature matrix : {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Feature names  : {', '.join(X.columns[:8].tolist())} ...")
    print(f"  Completed in   : {time.time() - t0:.1f}s")

    # Check for NaN / Inf
    nan_count = X.isna().sum().sum()
    inf_count = np.isinf(X.values).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  WARNING: {nan_count} NaN values and {inf_count} Inf values in features")
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

    # ------------------------------------------------------------------
    # 3. Train / val / test split
    # ------------------------------------------------------------------
    _print_section("3. Splitting dataset")
    X_trainval, X_test, y_trainval, y_test, yt_trainval, yt_test = train_test_split(
        X, y, y_type, test_size=0.15, stratify=y, random_state=args.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=args.seed
    )
    print(f"  Train   : {len(X_train):,} ({y_train.mean() * 100:.1f}% fraud)")
    print(f"  Val     : {len(X_val):,} ({y_val.mean() * 100:.1f}% fraud)")
    print(f"  Test    : {len(X_test):,} ({y_test.mean() * 100:.1f}% fraud)")

    # ------------------------------------------------------------------
    # 4. Train binary classifier
    # ------------------------------------------------------------------
    _print_section("4. Training binary fraud classifier (RF + XGBoost ensemble)")
    clf = VATFraudClassifier(n_estimators=300, random_state=args.seed)
    t0 = time.time()
    clf.fit(X_train, y_train, X_val, y_val)
    print(f"  Training completed in {time.time() - t0:.1f}s")
    print(f"  Optimal threshold  : {clf.threshold_:.4f}")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    _print_section("5. Evaluation on hold-out test set")
    metrics = clf.evaluate(X_test, y_test)

    print(f"\n  ROC-AUC         : {metrics.roc_auc:.4f}")
    print(f"  Avg Precision   : {metrics.avg_precision:.4f}")
    print(f"  F1 (fraud)      : {metrics.f1_fraud:.4f}")
    print(f"  Recall (fraud)  : {metrics.recall_fraud:.4f}")
    print(f"  Precision (fraud): {metrics.precision_fraud:.4f}")
    print(f"\n{metrics.classification_report}")
    _print_confusion_matrix(metrics.confusion_matrix)
    _print_feature_importances(metrics.feature_importances)

    # ------------------------------------------------------------------
    # 6. Multi-class fraud type classifier
    # ------------------------------------------------------------------
    _print_section("6. Training multi-class fraud type classifier")
    # Train only on fraud rows
    fraud_mask_train = y_train == 1
    fraud_mask_test = y_test == 1

    # Map legitimate to "legitimate" label for completeness
    yt_train_full = y_type.loc[y_train.index]
    yt_test_full = y_type.loc[y_test.index]

    clf.fit_multiclass(X_train[fraud_mask_train], yt_train_full[fraud_mask_train])
    type_preds = clf.predict_fraud_type(X_test[fraud_mask_test])
    type_true = yt_test_full[fraud_mask_test].values

    from sklearn.metrics import classification_report as cr
    print(cr(type_true, type_preds))

    # ------------------------------------------------------------------
    # 7. Save model
    # ------------------------------------------------------------------
    _print_section("7. Saving models")
    model_path = output_dir / "vat_fraud_classifier.joblib"
    clf.save(model_path)
    print(f"  Binary classifier  → {model_path}")

    # Save metrics as JSON
    metrics_path = output_dir / "training_metrics.json"
    metrics_dict = {
        "roc_auc": metrics.roc_auc,
        "avg_precision": metrics.avg_precision,
        "f1_fraud": metrics.f1_fraud,
        "recall_fraud": metrics.recall_fraud,
        "precision_fraud": metrics.precision_fraud,
        "threshold": metrics.threshold,
        "confusion_matrix": metrics.confusion_matrix,
        "top_10_features": sorted(
            metrics.feature_importances.items(), key=lambda x: x[1], reverse=True
        )[:10],
        "dataset_size": len(df),
        "fraud_ratio": float(y.mean()),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Training metrics   → {metrics_path}")

    _print_section("Done")
    print(f"\n  Model is ready. To use it in production:")
    print(f"    from src.fraud_detection.ml.classifier import VATFraudClassifier")
    print(f"    clf = VATFraudClassifier().load('{model_path}')")
    print()


if __name__ == "__main__":
    main()

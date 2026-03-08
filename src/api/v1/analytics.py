"""
Analytics & Reporting Endpoints
Provides aggregated data for the tax authority dashboard.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.fraud_detection.ml.benford_analysis import BenfordAnalyzer
from src.fraud_detection.graph.network_analyzer import NetworkAnalyzer
from src.models.alert import Alert, AlertSeverity, AlertStatus, FraudType
from src.models.case import Case, CaseStatus
from src.models.invoice import Invoice, InvoiceType
from src.models.taxpayer import Taxpayer, TaxpayerStatus
from src.models.transaction import Transaction

router = APIRouter()
_benford = BenfordAnalyzer()
_network = NetworkAnalyzer()


@router.get("/dashboard")
async def get_dashboard(db: AsyncSession = Depends(get_db)):
    """
    High-level executive dashboard for tax authority management.
    Returns KPIs covering fraud detection, alert status, and revenue protection.
    """
    # Taxpayer counts
    total_tp = await _scalar(db, select(func.count(Taxpayer.id)))
    active_tp = await _scalar(db, select(func.count(Taxpayer.id)).where(Taxpayer.status == TaxpayerStatus.ACTIVE))
    high_risk_tp = await _scalar(db, select(func.count(Taxpayer.id)).where(Taxpayer.risk_score >= 75))
    critical_risk_tp = await _scalar(db, select(func.count(Taxpayer.id)).where(Taxpayer.risk_score >= 90))

    # Alert counts
    open_alerts = await _scalar(db, select(func.count(Alert.id)).where(Alert.status == AlertStatus.OPEN))
    critical_alerts = await _scalar(db, select(func.count(Alert.id)).where(
        and_(Alert.severity == AlertSeverity.CRITICAL, Alert.status == AlertStatus.OPEN)
    ))
    total_rar = await _scalar(db, select(func.sum(Alert.estimated_revenue_at_risk)).where(
        Alert.status.in_([AlertStatus.OPEN, AlertStatus.UNDER_REVIEW])
    )) or 0

    # Case stats
    open_cases = await _scalar(db, select(func.count(Case.id)).where(Case.status.in_([
        CaseStatus.OPEN, CaseStatus.IN_PROGRESS
    ])))
    confirmed_fraud = await _scalar(db, select(func.count(Case.id)).where(Case.status == CaseStatus.CLOSED_FRAUD))
    total_recovered = await _scalar(db, select(func.sum(Case.recovered_amount)).where(
        Case.status == CaseStatus.CLOSED_FRAUD
    )) or 0

    # Invoice stats (last 30 days)
    from datetime import timedelta
    since = datetime.now(timezone.utc) - timedelta(days=30)
    recent_invoices = await _scalar(db, select(func.count(Invoice.id)).where(
        Invoice.submission_date >= since
    ))
    suspicious_invoices = await _scalar(db, select(func.count(Invoice.id)).where(
        and_(Invoice.is_suspicious == True, Invoice.submission_date >= since)
    ))

    # Fraud type breakdown (top 5)
    fraud_type_result = await db.execute(
        select(Alert.fraud_type, func.count(Alert.id).label("count"))
        .where(Alert.status != AlertStatus.FALSE_POSITIVE)
        .group_by(Alert.fraud_type)
        .order_by(func.count(Alert.id).desc())
        .limit(5)
    )
    top_fraud_types = [
        {"fraud_type": row.fraud_type, "count": row.count}
        for row in fraud_type_result
    ]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "taxpayers": {
            "total": total_tp,
            "active": active_tp,
            "high_risk": high_risk_tp,
            "critical_risk": critical_risk_tp,
        },
        "alerts": {
            "open": open_alerts,
            "critical_open": critical_alerts,
            "total_revenue_at_risk": float(total_rar),
        },
        "cases": {
            "open": open_cases,
            "confirmed_fraud_cases": confirmed_fraud,
            "total_recovered": float(total_recovered),
        },
        "invoices_last_30d": {
            "total_received": recent_invoices,
            "suspicious": suspicious_invoices,
            "suspicious_rate": (
                round(suspicious_invoices / recent_invoices, 4)
                if recent_invoices else 0
            ),
        },
        "top_fraud_types": top_fraud_types,
    }


@router.get("/risk-heatmap")
async def get_risk_heatmap(
    country_code: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns risk distribution by industry sector for choropleth / heatmap visualisation.
    """
    stmt = (
        select(
            Taxpayer.industry_code,
            Taxpayer.industry_description,
            Taxpayer.country_code,
            func.count(Taxpayer.id).label("taxpayer_count"),
            func.avg(Taxpayer.risk_score).label("avg_risk"),
            func.max(Taxpayer.risk_score).label("max_risk"),
        )
        .group_by(
            Taxpayer.industry_code,
            Taxpayer.industry_description,
            Taxpayer.country_code,
        )
        .order_by(func.avg(Taxpayer.risk_score).desc())
    )
    if country_code:
        stmt = stmt.where(Taxpayer.country_code == country_code.upper())

    result = await db.execute(stmt)
    return [
        {
            "industry_code": row.industry_code,
            "industry_description": row.industry_description,
            "country_code": row.country_code,
            "taxpayer_count": row.taxpayer_count,
            "avg_risk_score": round(float(row.avg_risk or 0), 2),
            "max_risk_score": round(float(row.max_risk or 0), 2),
        }
        for row in result
    ]


@router.get("/benford/{tin}")
async def benford_analysis(
    tin: str,
    period: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Run Benford's Law analysis on invoices issued by a taxpayer.
    """
    stmt = select(Invoice).where(Invoice.seller_tin == tin)
    if period:
        stmt = stmt.where(Invoice.vat_return_period == period)
    result = await db.execute(stmt.limit(2000))
    invoices = list(result.scalars().all())

    if not invoices:
        return {"error": "No invoices found for this taxpayer"}

    analysis = _benford.analyze_invoices(invoices)
    return {
        "tin": tin,
        "period": period,
        "sample_size": analysis.sample_size,
        "chi2_statistic": analysis.chi2_statistic,
        "p_value": analysis.p_value,
        "is_suspicious": analysis.is_suspicious,
        "mean_absolute_deviation": analysis.mean_absolute_deviation,
        "observed_frequencies": analysis.observed_frequencies,
        "expected_frequencies": analysis.expected_frequencies,
        "deviations": analysis.deviations,
        "interpretation": analysis.interpretation,
    }


@router.get("/vat-gap")
async def vat_gap_analysis(
    period: str = Query(..., description="Period in YYYY-MM format"),
    db: AsyncSession = Depends(get_db),
):
    """
    Estimate the VAT gap for a given period by comparing:
      - VAT declared in returns (from transactions)
      - VAT implied by matched invoice data
    """
    # VAT declared in returns
    declared_result = await db.execute(
        select(func.sum(Transaction.output_vat)).where(
            Transaction.period == period,
            Transaction.transaction_type == "vat_return",
        )
    )
    declared_output = float(declared_result.scalar() or 0)

    # VAT on invoices for the period
    invoice_result = await db.execute(
        select(func.sum(Invoice.vat_amount)).where(
            Invoice.vat_return_period == period,
            Invoice.invoice_type.in_([InvoiceType.STANDARD, InvoiceType.INTRA_COMMUNITY]),
        )
    )
    invoice_vat = float(invoice_result.scalar() or 0)

    # Unmatched input VAT claims (buyer claims without seller declaration)
    unmatched_input = await db.execute(
        select(func.sum(Invoice.vat_amount)).where(
            Invoice.vat_return_period == period,
            Invoice.declared_by_buyer == True,
            Invoice.declared_by_seller == False,
        )
    )
    unmatched_vat = float(unmatched_input.scalar() or 0)

    gap = invoice_vat - declared_output
    gap_pct = (gap / invoice_vat * 100) if invoice_vat > 0 else 0

    return {
        "period": period,
        "declared_output_vat": declared_output,
        "invoice_implied_vat": invoice_vat,
        "unmatched_input_vat_claims": unmatched_vat,
        "estimated_vat_gap": max(0, gap),
        "estimated_gap_pct": round(gap_pct, 2),
        "interpretation": (
            "High VAT gap detected – significant under-declaration or missing trader activity."
            if gap_pct > 10 else
            "VAT gap within acceptable tolerance."
        ),
    }


@router.get("/top-risk-taxpayers")
async def top_risk_taxpayers(
    limit: int = Query(20, le=100),
    fraud_type: FraudType | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Return the highest-risk taxpayers for prioritised audit selection."""
    stmt = (
        select(Taxpayer)
        .where(Taxpayer.risk_score > 0)
        .order_by(Taxpayer.risk_score.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    taxpayers = result.scalars().all()

    return [
        {
            "tin": tp.tin,
            "vat_number": tp.vat_number,
            "name": tp.name,
            "risk_score": tp.risk_score,
            "status": tp.status,
            "country_code": tp.country_code,
            "total_alerts": tp.total_alerts,
            "last_filing_date": tp.last_filing_date.isoformat() if tp.last_filing_date else None,
        }
        for tp in taxpayers
    ]


# ── Utility ───────────────────────────────────────────────────────────────────

async def _scalar(db: AsyncSession, stmt) -> int | float:
    result = await db.execute(stmt)
    return result.scalar() or 0

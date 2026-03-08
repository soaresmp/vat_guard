"""
Taxpayer Management & Risk Assessment Endpoints
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.fraud_detection.engine import FraudDetectionEngine
from src.models.taxpayer import Taxpayer, TaxpayerStatus
from src.models.invoice import Invoice
from src.models.transaction import Transaction
from src.schemas.taxpayer import (
    TaxpayerCreate, TaxpayerUpdate, TaxpayerResponse,
    TaxpayerSummary, TaxpayerRiskSummary,
)

router = APIRouter()
_engine = FraudDetectionEngine()


@router.get("/", response_model=list[TaxpayerSummary])
async def list_taxpayers(
    status: TaxpayerStatus | None = None,
    country_code: str | None = None,
    min_risk_score: float | None = Query(None, ge=0, le=100),
    search: str | None = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List taxpayers with optional filters."""
    stmt = select(Taxpayer)
    if status:
        stmt = stmt.where(Taxpayer.status == status)
    if country_code:
        stmt = stmt.where(Taxpayer.country_code == country_code.upper())
    if min_risk_score is not None:
        stmt = stmt.where(Taxpayer.risk_score >= min_risk_score)
    if search:
        stmt = stmt.where(
            or_(
                Taxpayer.name.ilike(f"%{search}%"),
                Taxpayer.tin.ilike(f"%{search}%"),
                Taxpayer.vat_number.ilike(f"%{search}%"),
            )
        )
    stmt = stmt.order_by(Taxpayer.risk_score.desc()).offset(offset).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.post("/", response_model=TaxpayerResponse, status_code=status.HTTP_201_CREATED)
async def create_taxpayer(
    payload: TaxpayerCreate,
    db: AsyncSession = Depends(get_db),
):
    """Register a new VAT taxpayer."""
    # Check uniqueness
    existing = await db.execute(
        select(Taxpayer).where(
            or_(Taxpayer.tin == payload.tin, Taxpayer.vat_number == payload.vat_number)
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Taxpayer with this TIN or VAT number already exists.",
        )
    taxpayer = Taxpayer(**payload.model_dump())
    db.add(taxpayer)
    await db.flush()
    return taxpayer


@router.get("/{tin}", response_model=TaxpayerResponse)
async def get_taxpayer(tin: str, db: AsyncSession = Depends(get_db)):
    """Get taxpayer details by TIN."""
    result = await db.execute(select(Taxpayer).where(Taxpayer.tin == tin))
    taxpayer = result.scalar_one_or_none()
    if not taxpayer:
        raise HTTPException(status_code=404, detail="Taxpayer not found")
    return taxpayer


@router.patch("/{tin}", response_model=TaxpayerResponse)
async def update_taxpayer(
    tin: str,
    payload: TaxpayerUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update taxpayer details."""
    result = await db.execute(select(Taxpayer).where(Taxpayer.tin == tin))
    taxpayer = result.scalar_one_or_none()
    if not taxpayer:
        raise HTTPException(status_code=404, detail="Taxpayer not found")
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(taxpayer, field, value)
    return taxpayer


@router.post("/{tin}/assess-risk", response_model=TaxpayerRiskSummary)
async def assess_risk(
    tin: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger a full fraud risk assessment for a taxpayer.
    Runs all detection rules, ML models, and graph analysis.
    """
    # Load taxpayer
    tp_result = await db.execute(select(Taxpayer).where(Taxpayer.tin == tin))
    taxpayer = tp_result.scalar_one_or_none()
    if not taxpayer:
        raise HTTPException(status_code=404, detail="Taxpayer not found")

    # Load recent invoices and transactions
    inv_result = await db.execute(
        select(Invoice).where(
            or_(Invoice.seller_tin == tin, Invoice.buyer_tin == tin)
        ).limit(1000)
    )
    invoices = list(inv_result.scalars().all())

    tx_result = await db.execute(
        select(Transaction).where(Transaction.taxpayer_tin == tin).limit(200)
    )
    transactions = list(tx_result.scalars().all())

    # Run detection
    alerts, risk_score = await _engine.analyze_taxpayer(taxpayer, invoices, transactions)

    # Persist generated alerts
    for alert in alerts:
        db.add(alert)

    # Update taxpayer risk score
    taxpayer.risk_score = risk_score
    from datetime import datetime, timezone
    taxpayer.last_risk_assessment = datetime.now(timezone.utc)
    taxpayer.total_alerts = taxpayer.total_alerts + len(alerts)

    await db.flush()

    return TaxpayerRiskSummary(
        tin=taxpayer.tin,
        vat_number=taxpayer.vat_number,
        name=taxpayer.name,
        risk_score=risk_score,
        risk_level=(
            "critical" if risk_score >= 90 else
            "high" if risk_score >= 75 else
            "medium" if risk_score >= 40 else "low"
        ),
        top_risk_factors=[a.title for a in alerts[:5]],
        component_scores={},
        total_alerts=taxpayer.total_alerts,
    )


@router.get("/{tin}/network", response_model=dict)
async def get_transaction_network(
    tin: str,
    depth: int = Query(2, ge=1, le=3),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the transaction network graph around a taxpayer (for visualisation).
    """
    from src.fraud_detection.graph.network_analyzer import NetworkAnalyzer
    tp_result = await db.execute(select(Taxpayer).where(Taxpayer.tin == tin))
    taxpayer = tp_result.scalar_one_or_none()
    if not taxpayer:
        raise HTTPException(status_code=404, detail="Taxpayer not found")

    inv_result = await db.execute(
        select(Invoice).where(
            or_(Invoice.seller_tin == tin, Invoice.buyer_tin == tin)
        ).limit(500)
    )
    invoices = list(inv_result.scalars().all())

    analyzer = NetworkAnalyzer()
    result = analyzer.analyze(invoices)

    return {
        "tin": tin,
        "node_count": result.node_count,
        "edge_count": result.edge_count,
        "cycle_count": result.cycle_count,
        "risk_score": result.risk_score,
        "suspicious_nodes": result.suspicious_nodes,
        "cycles": result.cycles[:10],
        "summary": result.summary,
    }

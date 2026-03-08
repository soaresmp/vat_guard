"""
Alert Management Endpoints
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.models.alert import Alert, AlertSeverity, AlertStatus, FraudType
from src.schemas.alert import AlertResponse, AlertSummary, AlertStats, AlertUpdate

router = APIRouter()


@router.get("/", response_model=list[AlertSummary])
async def list_alerts(
    status: AlertStatus | None = None,
    severity: AlertSeverity | None = None,
    fraud_type: FraudType | None = None,
    taxpayer_tin: str | None = None,
    assigned_to: str | None = None,
    min_risk_score: float | None = Query(None, ge=0, le=100),
    limit: int = Query(50, le=500),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List alerts with filters. Ordered by risk score descending."""
    stmt = select(Alert)
    if status:
        stmt = stmt.where(Alert.status == status)
    if severity:
        stmt = stmt.where(Alert.severity == severity)
    if fraud_type:
        stmt = stmt.where(Alert.fraud_type == fraud_type)
    if taxpayer_tin:
        stmt = stmt.where(Alert.taxpayer_tin == taxpayer_tin)
    if assigned_to:
        stmt = stmt.where(Alert.assigned_to == assigned_to)
    if min_risk_score is not None:
        stmt = stmt.where(Alert.risk_score >= min_risk_score)
    stmt = stmt.order_by(Alert.risk_score.desc(), Alert.created_at.desc())
    stmt = stmt.offset(offset).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/stats", response_model=AlertStats)
async def get_alert_stats(db: AsyncSession = Depends(get_db)):
    """Summary statistics for the alert dashboard."""
    # Total counts by status
    status_counts: dict[str, int] = {}
    for s in AlertStatus:
        count_result = await db.execute(
            select(func.count()).where(Alert.status == s)
        )
        status_counts[s.value] = count_result.scalar() or 0

    # By severity
    severity_counts: dict[str, int] = {}
    for sev in AlertSeverity:
        count_result = await db.execute(
            select(func.count()).where(Alert.severity == sev)
        )
        severity_counts[sev.value] = count_result.scalar() or 0

    # By fraud type
    fraud_counts: dict[str, int] = {}
    for ft in FraudType:
        count_result = await db.execute(
            select(func.count()).where(Alert.fraud_type == ft)
        )
        fraud_counts[ft.value] = count_result.scalar() or 0

    # Total revenue at risk
    rar_result = await db.execute(
        select(func.sum(Alert.estimated_revenue_at_risk)).where(
            Alert.status.in_([AlertStatus.OPEN, AlertStatus.UNDER_REVIEW])
        )
    )
    total_rar = float(rar_result.scalar() or 0)

    total_result = await db.execute(select(func.count(Alert.id)))
    total = total_result.scalar() or 0

    return AlertStats(
        total=total,
        open=status_counts.get("open", 0),
        under_review=status_counts.get("under_review", 0),
        confirmed_fraud=status_counts.get("confirmed_fraud", 0),
        false_positive=status_counts.get("false_positive", 0),
        by_severity=severity_counts,
        by_fraud_type=fraud_counts,
        total_revenue_at_risk=total_rar,
    )


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.patch("/{alert_id}", response_model=AlertResponse)
async def update_alert(
    alert_id: UUID,
    payload: AlertUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update alert status, assignment, or resolution."""
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    update_data = payload.model_dump(exclude_none=True)
    for field, value in update_data.items():
        setattr(alert, field, value)

    # Auto-set timestamps
    from datetime import datetime, timezone
    if payload.status in (AlertStatus.RESOLVED, AlertStatus.FALSE_POSITIVE, AlertStatus.CONFIRMED_FRAUD):
        alert.resolved_at = datetime.now(timezone.utc)
    if payload.assigned_to and not alert.assigned_at:
        alert.assigned_at = datetime.now(timezone.utc)

    return alert


@router.post("/{alert_id}/escalate", response_model=AlertResponse)
async def escalate_alert(
    alert_id: UUID,
    reason: str,
    db: AsyncSession = Depends(get_db),
):
    """Escalate an alert to a formal investigation case."""
    from src.models.case import Case, CaseStatus, CasePriority
    from datetime import datetime, timezone
    import uuid

    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    # Create a case
    case = Case(
        case_number=f"CASE-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}",
        title=f"Escalated: {alert.title}",
        description=reason,
        taxpayer_tin=alert.taxpayer_tin,
        status=CaseStatus.OPEN,
        priority=CasePriority.HIGH if alert.severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL) else CasePriority.MEDIUM,
        fraud_types=[alert.fraud_type.value] if alert.fraud_type else None,
        estimated_revenue_at_risk=alert.estimated_revenue_at_risk,
    )
    db.add(case)
    await db.flush()

    alert.status = AlertStatus.ESCALATED
    alert.case_id = case.id
    return alert

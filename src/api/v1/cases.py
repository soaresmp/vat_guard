"""
Case Management Endpoints
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.models.case import Case, CaseStatus, CasePriority
from src.schemas.case import CaseCreate, CaseUpdate, CaseResponse

router = APIRouter()


@router.get("/", response_model=list[CaseResponse])
async def list_cases(
    status: CaseStatus | None = None,
    priority: CasePriority | None = None,
    taxpayer_tin: str | None = None,
    assigned_to: str | None = None,
    is_cross_border: bool | None = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Case)
    if status:
        stmt = stmt.where(Case.status == status)
    if priority:
        stmt = stmt.where(Case.priority == priority)
    if taxpayer_tin:
        stmt = stmt.where(Case.taxpayer_tin == taxpayer_tin)
    if assigned_to:
        stmt = stmt.where(Case.assigned_to == assigned_to)
    if is_cross_border is not None:
        stmt = stmt.where(Case.is_cross_border == is_cross_border)
    stmt = stmt.order_by(Case.opened_at.desc()).offset(offset).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.post("/", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(
    payload: CaseCreate,
    db: AsyncSession = Depends(get_db),
):
    from datetime import datetime, timezone
    import uuid

    case_num = f"CASE-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
    case = Case(case_number=case_num, **payload.model_dump())
    db.add(case)
    await db.flush()
    return case


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(case_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@router.patch("/{case_id}", response_model=CaseResponse)
async def update_case(
    case_id: UUID,
    payload: CaseUpdate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(case, field, value)
    from datetime import datetime, timezone
    if payload.status in (CaseStatus.CLOSED_FRAUD, CaseStatus.CLOSED_NO_FRAUD, CaseStatus.CLOSED_INSUFFICIENT):
        case.closed_at = datetime.now(timezone.utc)
    return case


@router.get("/{case_id}/alerts", response_model=list[dict])
async def get_case_alerts(case_id: UUID, db: AsyncSession = Depends(get_db)):
    from src.models.alert import Alert
    result = await db.execute(select(Alert).where(Alert.case_id == case_id))
    alerts = result.scalars().all()
    return [
        {
            "id": str(a.id),
            "fraud_type": a.fraud_type,
            "severity": a.severity,
            "title": a.title,
            "risk_score": a.risk_score,
            "created_at": a.created_at.isoformat(),
        }
        for a in alerts
    ]

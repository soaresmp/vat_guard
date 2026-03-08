from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from src.models.case import CasePriority, CaseStatus


class CaseCreate(BaseModel):
    title: str
    description: str | None = None
    taxpayer_tin: str | None = None
    priority: CasePriority = CasePriority.MEDIUM
    fraud_types: list[str] | None = None
    estimated_revenue_at_risk: float | None = None
    assigned_to: str | None = None
    team: str | None = None
    tags: list[str] | None = None


class CaseUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    status: CaseStatus | None = None
    priority: CasePriority | None = None
    assigned_to: str | None = None
    team: str | None = None
    confirmed_fraud_amount: float | None = None
    recovered_amount: float | None = None
    prosecution_reference: str | None = None
    customs_reference: str | None = None
    fiu_reference: str | None = None
    outcome_summary: str | None = None
    tags: list[str] | None = None


class CaseResponse(BaseModel):
    id: UUID
    case_number: str
    title: str
    description: str | None = None
    taxpayer_tin: str | None = None
    status: CaseStatus
    priority: CasePriority
    fraud_types: list | None = None
    estimated_revenue_at_risk: float | None = None
    confirmed_fraud_amount: float | None = None
    recovered_amount: float | None = None
    assigned_to: str | None = None
    team: str | None = None
    opened_at: datetime
    closed_at: datetime | None = None
    is_cross_border: bool
    countries_involved: list | None = None
    tags: list | None = None
    created_at: datetime

    model_config = {"from_attributes": True}

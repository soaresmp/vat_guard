from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from src.models.alert import AlertSeverity, AlertStatus, FraudType


class AlertSummary(BaseModel):
    id: UUID
    fraud_type: FraudType
    severity: AlertSeverity
    status: AlertStatus
    title: str
    risk_score: float
    taxpayer_tin: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AlertResponse(BaseModel):
    id: UUID
    taxpayer_tin: str | None = None
    invoice_id: UUID | None = None
    case_id: UUID | None = None
    fraud_type: FraudType
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    detection_rule: str
    risk_score: float
    confidence: float
    evidence: dict | None = None
    related_tins: list | None = None
    estimated_revenue_at_risk: float | None = None
    assigned_to: str | None = None
    resolved_at: datetime | None = None
    resolution_notes: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AlertUpdate(BaseModel):
    status: AlertStatus | None = None
    assigned_to: str | None = None
    resolution_notes: str | None = None
    case_id: UUID | None = None


class AlertStats(BaseModel):
    total: int
    open: int
    under_review: int
    confirmed_fraud: int
    false_positive: int
    by_severity: dict[str, int]
    by_fraud_type: dict[str, int]
    total_revenue_at_risk: float

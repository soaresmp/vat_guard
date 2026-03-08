"""
Case model – a formal investigation case grouping one or more alerts.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey,
    Index, Integer, String, Text, func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class CaseStatus(str, enum.Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING_EXTERNAL = "pending_external"  # Awaiting customs / FIU data
    ESCALATED = "escalated"               # Sent to prosecution / court
    CLOSED_FRAUD = "closed_fraud"
    CLOSED_NO_FRAUD = "closed_no_fraud"
    CLOSED_INSUFFICIENT = "closed_insufficient"


class CasePriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Reference
    case_number: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    # Primary subject
    taxpayer_tin: Mapped[str | None] = mapped_column(
        String(50), ForeignKey("taxpayers.tin"), index=True
    )

    # Classification
    status: Mapped[CaseStatus] = mapped_column(
        Enum(CaseStatus), nullable=False, default=CaseStatus.OPEN, index=True
    )
    priority: Mapped[CasePriority] = mapped_column(
        Enum(CasePriority), nullable=False, default=CasePriority.MEDIUM, index=True
    )
    fraud_types: Mapped[list | None] = mapped_column(JSONB)      # List of FraudType values

    # Financial exposure
    estimated_revenue_at_risk: Mapped[float | None] = mapped_column(Float)
    confirmed_fraud_amount: Mapped[float | None] = mapped_column(Float)
    recovered_amount: Mapped[float | None] = mapped_column(Float)

    # Assignment
    assigned_to: Mapped[str | None] = mapped_column(String(100))
    assigned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    team: Mapped[str | None] = mapped_column(String(100))

    # Timeline
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    target_close_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # External references
    prosecution_reference: Mapped[str | None] = mapped_column(String(100))
    customs_reference: Mapped[str | None] = mapped_column(String(100))
    fiu_reference: Mapped[str | None] = mapped_column(String(100))

    # Outcome
    outcome_summary: Mapped[str | None] = mapped_column(Text)
    is_cross_border: Mapped[bool] = mapped_column(Boolean, default=False)
    countries_involved: Mapped[list | None] = mapped_column(JSONB)

    # Metadata
    tags: Mapped[list | None] = mapped_column(JSONB)
    attachments: Mapped[list | None] = mapped_column(JSONB)  # File references

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    taxpayer: Mapped["Taxpayer | None"] = relationship("Taxpayer", back_populates="cases")  # noqa: F821
    alerts: Mapped[list["Alert"]] = relationship("Alert", back_populates="case")  # noqa: F821

    __table_args__ = (
        Index("ix_cases_status_priority", "status", "priority"),
        Index("ix_cases_opened", "opened_at"),
    )

    def __repr__(self) -> str:
        return f"<Case {self.case_number} {self.status} tin={self.taxpayer_tin}>"

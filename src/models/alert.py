"""
Alert model – a fraud signal raised by any detection rule or ML model.
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


class FraudType(str, enum.Enum):
    """VAT fraud categories based on IMF taxonomy."""
    MISSING_TRADER = "missing_trader"          # MTIC – trader disappears with VAT
    CAROUSEL_FRAUD = "carousel_fraud"          # Circular goods / VAT chain
    CONTRA_TRADING = "contra_trading"          # Two-carousel offset scheme
    INVOICE_MILL = "invoice_mill"              # Fake invoice factory
    BOGUS_TRADER = "bogus_trader"              # Shell company generating false invoices
    REFUND_FRAUD = "refund_fraud"              # False VAT refund claims
    ACQUISITION_FRAUD = "acquisition_fraud"   # Exploiting zero-rated imports
    UNDER_DECLARATION = "under_declaration"   # Sales under-reported
    SUPPRESSED_SALES = "suppressed_sales"     # Off-book cash sales
    PHANTOM_EXPORTER = "phantom_exporter"     # Fake export to claim zero-rating
    IDENTITY_THEFT = "identity_theft"         # Using another entity's VAT number
    SPLIT_TRANSACTIONS = "split_transactions" # Below-threshold splitting
    BENFORD_ANOMALY = "benford_anomaly"        # Statistical digit anomaly
    NETWORK_ANOMALY = "network_anomaly"        # Suspicious transaction network
    LATE_REGISTRATION = "late_registration"   # Filing after deadline pattern
    OTHER = "other"


class AlertSeverity(str, enum.Enum):
    LOW = "low"           # Informational / watch
    MEDIUM = "medium"     # Requires review
    HIGH = "high"         # Requires action
    CRITICAL = "critical" # Immediate escalation


class AlertStatus(str, enum.Enum):
    OPEN = "open"
    UNDER_REVIEW = "under_review"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    CONFIRMED_FRAUD = "confirmed_fraud"


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Links to entities
    taxpayer_tin: Mapped[str | None] = mapped_column(
        String(50), ForeignKey("taxpayers.tin"), index=True
    )
    invoice_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("invoices.id"), index=True
    )
    case_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cases.id"), index=True
    )

    # Classification
    fraud_type: Mapped[FraudType] = mapped_column(Enum(FraudType), nullable=False, index=True)
    severity: Mapped[AlertSeverity] = mapped_column(
        Enum(AlertSeverity), nullable=False, default=AlertSeverity.MEDIUM, index=True
    )
    status: Mapped[AlertStatus] = mapped_column(
        Enum(AlertStatus), nullable=False, default=AlertStatus.OPEN, index=True
    )

    # Detection details
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    detection_rule: Mapped[str] = mapped_column(String(100))  # Rule or model name
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)  # 0–100
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)  # 0–1

    # Supporting evidence
    evidence: Mapped[dict | None] = mapped_column(JSONB)        # Structured proof
    related_tins: Mapped[list | None] = mapped_column(JSONB)    # Other taxpayers involved
    related_invoice_ids: Mapped[list | None] = mapped_column(JSONB)
    estimated_revenue_at_risk: Mapped[float | None] = mapped_column(Float)

    # Assignment
    assigned_to: Mapped[str | None] = mapped_column(String(100))  # Analyst username
    assigned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Resolution
    resolved_by: Mapped[str | None] = mapped_column(String(100))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolution_notes: Mapped[str | None] = mapped_column(Text)
    is_auto_resolved: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    taxpayer: Mapped["Taxpayer | None"] = relationship("Taxpayer", back_populates="alerts")  # noqa: F821
    invoice: Mapped["Invoice | None"] = relationship("Invoice", back_populates="alerts")  # noqa: F821
    case: Mapped["Case | None"] = relationship("Case", back_populates="alerts")  # noqa: F821

    __table_args__ = (
        Index("ix_alerts_severity_status", "severity", "status"),
        Index("ix_alerts_fraud_type", "fraud_type"),
        Index("ix_alerts_created", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Alert {self.fraud_type} {self.severity} "
            f"tin={self.taxpayer_tin} score={self.risk_score:.1f}>"
        )

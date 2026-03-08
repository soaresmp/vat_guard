"""
Taxpayer model – represents a VAT-registered business entity.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey,
    Index, Integer, String, Text, func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class TaxpayerStatus(str, enum.Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEREGISTERED = "deregistered"
    UNDER_INVESTIGATION = "under_investigation"
    MISSING = "missing"           # Filed for deregistration / gone missing


class BusinessType(str, enum.Enum):
    SOLE_PROPRIETOR = "sole_proprietor"
    PARTNERSHIP = "partnership"
    LIMITED_LIABILITY = "limited_liability"
    PUBLIC_COMPANY = "public_company"
    BRANCH = "branch"
    COOPERATIVE = "cooperative"
    NON_PROFIT = "non_profit"


class Taxpayer(Base):
    __tablename__ = "taxpayers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Official registration identifiers
    tin: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    vat_number: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    trade_name: Mapped[str | None] = mapped_column(String(255))

    # Business details
    business_type: Mapped[BusinessType] = mapped_column(
        Enum(BusinessType), nullable=False, default=BusinessType.LIMITED_LIABILITY
    )
    industry_code: Mapped[str | None] = mapped_column(String(10))   # ISIC / local sector code
    industry_description: Mapped[str | None] = mapped_column(String(255))

    # Registration
    registration_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    vat_registration_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deregistration_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[TaxpayerStatus] = mapped_column(
        Enum(TaxpayerStatus), nullable=False, default=TaxpayerStatus.ACTIVE, index=True
    )

    # Contact & Location
    country_code: Mapped[str] = mapped_column(String(3), nullable=False, default="XX")
    region: Mapped[str | None] = mapped_column(String(100))
    address: Mapped[str | None] = mapped_column(Text)
    email: Mapped[str | None] = mapped_column(String(255))
    phone: Mapped[str | None] = mapped_column(String(50))

    # Financial profile (updated periodically from returns)
    annual_turnover_reported: Mapped[float | None] = mapped_column(Float)
    annual_vat_declared: Mapped[float | None] = mapped_column(Float)
    annual_vat_paid: Mapped[float | None] = mapped_column(Float)
    average_monthly_refund: Mapped[float | None] = mapped_column(Float)

    # Risk & compliance
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)  # 0–100
    compliance_score: Mapped[float] = mapped_column(Float, default=100.0)  # 0–100
    last_risk_assessment: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_filing_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    consecutive_late_filings: Mapped[int] = mapped_column(Integer, default=0)
    total_alerts: Mapped[int] = mapped_column(Integer, default=0)
    total_open_cases: Mapped[int] = mapped_column(Integer, default=0)

    # Beneficial ownership / director flags
    has_foreign_directors: Mapped[bool] = mapped_column(Boolean, default=False)
    is_newly_registered: Mapped[bool] = mapped_column(Boolean, default=False)
    is_high_value_trader: Mapped[bool] = mapped_column(Boolean, default=False)
    is_frequent_refund_claimant: Mapped[bool] = mapped_column(Boolean, default=False)

    # Audit columns
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    issued_invoices: Mapped[list["Invoice"]] = relationship(  # noqa: F821
        "Invoice", foreign_keys="Invoice.seller_tin", back_populates="seller", lazy="select"
    )
    received_invoices: Mapped[list["Invoice"]] = relationship(  # noqa: F821
        "Invoice", foreign_keys="Invoice.buyer_tin", back_populates="buyer", lazy="select"
    )
    alerts: Mapped[list["Alert"]] = relationship(  # noqa: F821
        "Alert", back_populates="taxpayer", lazy="select"
    )
    cases: Mapped[list["Case"]] = relationship(  # noqa: F821
        "Case", back_populates="taxpayer", lazy="select"
    )

    __table_args__ = (
        Index("ix_taxpayers_risk_score", "risk_score"),
        Index("ix_taxpayers_country_status", "country_code", "status"),
    )

    def __repr__(self) -> str:
        return f"<Taxpayer vat={self.vat_number} name={self.name} risk={self.risk_score:.1f}>"

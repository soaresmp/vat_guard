"""
Invoice model – represents an electronic invoice submitted to VATGuard.

Supports both domestic and cross-border (intra-community) invoices.
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


class InvoiceType(str, enum.Enum):
    STANDARD = "standard"          # Regular B2B invoice
    EXPORT = "export"              # Zero-rated export
    IMPORT = "import"              # Import invoice (duty + VAT)
    INTRA_COMMUNITY = "intra_community"  # Within customs/trading union
    CREDIT_NOTE = "credit_note"    # Reversal / correction
    DEBIT_NOTE = "debit_note"
    ADVANCE = "advance"            # Pre-payment invoice
    SELF_BILLED = "self_billed"    # Buyer-generated invoice


class InvoiceStatus(str, enum.Enum):
    RECEIVED = "received"
    VALIDATED = "validated"
    MATCHED = "matched"            # Buyer & seller both declared it
    UNMATCHED = "unmatched"        # Counter-party has NOT declared it
    FLAGGED = "flagged"            # Raised an alert
    CANCELLED = "cancelled"
    AMENDED = "amended"


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # External reference
    invoice_number: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    external_id: Mapped[str | None] = mapped_column(String(100), index=True)  # ID in source system

    # Parties
    seller_tin: Mapped[str] = mapped_column(
        String(50), ForeignKey("taxpayers.tin"), nullable=False, index=True
    )
    buyer_tin: Mapped[str | None] = mapped_column(
        String(50), ForeignKey("taxpayers.tin"), index=True
    )
    seller_vat_number: Mapped[str] = mapped_column(String(50), nullable=False)
    buyer_vat_number: Mapped[str | None] = mapped_column(String(50))
    seller_country: Mapped[str] = mapped_column(String(3), nullable=False)
    buyer_country: Mapped[str | None] = mapped_column(String(3))

    # Invoice classification
    invoice_type: Mapped[InvoiceType] = mapped_column(
        Enum(InvoiceType), nullable=False, default=InvoiceType.STANDARD, index=True
    )
    status: Mapped[InvoiceStatus] = mapped_column(
        Enum(InvoiceStatus), nullable=False, default=InvoiceStatus.RECEIVED, index=True
    )

    # Dates
    invoice_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    supply_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    due_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    submission_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    # Financial amounts (all in base currency)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    exchange_rate: Mapped[float] = mapped_column(Float, default=1.0)
    net_amount: Mapped[float] = mapped_column(Float, nullable=False)
    vat_amount: Mapped[float] = mapped_column(Float, nullable=False)
    gross_amount: Mapped[float] = mapped_column(Float, nullable=False)
    vat_rate: Mapped[float] = mapped_column(Float, nullable=False)   # e.g. 0.20 for 20%

    # Goods / services
    description: Mapped[str | None] = mapped_column(Text)
    commodity_code: Mapped[str | None] = mapped_column(String(20))   # HS code for customs
    line_items: Mapped[dict | None] = mapped_column(JSONB)           # Detailed line items

    # Cross-border / customs
    customs_declaration_number: Mapped[str | None] = mapped_column(String(100))
    port_of_entry: Mapped[str | None] = mapped_column(String(50))
    incoterms: Mapped[str | None] = mapped_column(String(20))

    # Matching / counter-declaration
    matched_invoice_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("invoices.id")
    )
    matching_score: Mapped[float | None] = mapped_column(Float)  # 0–1 confidence

    # Risk signals
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    is_suspicious: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    fraud_indicators: Mapped[dict | None] = mapped_column(JSONB)  # Dict of indicator: score

    # Filing reference
    vat_return_period: Mapped[str | None] = mapped_column(String(10))  # e.g. "2024-Q1"
    declared_by_seller: Mapped[bool] = mapped_column(Boolean, default=False)
    declared_by_buyer: Mapped[bool] = mapped_column(Boolean, default=False)

    # Audit
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    source_system: Mapped[str | None] = mapped_column(String(50))  # Origin system

    # Relationships
    seller: Mapped["Taxpayer"] = relationship(  # noqa: F821
        "Taxpayer", foreign_keys=[seller_tin], back_populates="issued_invoices"
    )
    buyer: Mapped["Taxpayer | None"] = relationship(  # noqa: F821
        "Taxpayer", foreign_keys=[buyer_tin], back_populates="received_invoices"
    )
    alerts: Mapped[list["Alert"]] = relationship(  # noqa: F821
        "Alert", back_populates="invoice", lazy="select"
    )

    __table_args__ = (
        Index("ix_invoices_seller_date", "seller_tin", "invoice_date"),
        Index("ix_invoices_buyer_date", "buyer_tin", "invoice_date"),
        Index("ix_invoices_risk", "risk_score", "is_suspicious"),
        Index("ix_invoices_type_status", "invoice_type", "status"),
        Index("ix_invoices_vat_period", "vat_return_period"),
    )

    def __repr__(self) -> str:
        return (
            f"<Invoice {self.invoice_number} "
            f"{self.seller_tin}→{self.buyer_tin} "
            f"VAT={self.vat_amount:.2f} risk={self.risk_score:.1f}>"
        )

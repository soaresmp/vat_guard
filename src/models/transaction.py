"""
Transaction model – aggregated VAT filing / payment record per taxpayer per period.
Used to track VAT gap, late payments, and suspicious filing patterns.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey,
    Index, Integer, String, func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class TransactionType(str, enum.Enum):
    VAT_RETURN = "vat_return"
    VAT_PAYMENT = "vat_payment"
    REFUND_CLAIM = "refund_claim"
    REFUND_PAYMENT = "refund_payment"
    AMENDED_RETURN = "amended_return"
    AUDIT_ASSESSMENT = "audit_assessment"
    PENALTY = "penalty"


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    taxpayer_tin: Mapped[str] = mapped_column(
        String(50), ForeignKey("taxpayers.tin"), nullable=False, index=True
    )
    transaction_type: Mapped[TransactionType] = mapped_column(
        Enum(TransactionType), nullable=False, index=True
    )

    # Filing period
    period: Mapped[str] = mapped_column(String(10), nullable=False)   # e.g. "2024-03"
    period_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    due_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    filed_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_late: Mapped[bool] = mapped_column(Boolean, default=False)

    # VAT return figures (from the declared return)
    output_vat: Mapped[float | None] = mapped_column(Float)      # Tax on sales
    input_vat: Mapped[float | None] = mapped_column(Float)       # Tax on purchases
    net_vat_payable: Mapped[float | None] = mapped_column(Float) # output – input
    refund_claimed: Mapped[float | None] = mapped_column(Float)
    amount_paid: Mapped[float | None] = mapped_column(Float)

    # Cross-checked figures (from invoice matching)
    matched_output_vat: Mapped[float | None] = mapped_column(Float)
    matched_input_vat: Mapped[float | None] = mapped_column(Float)
    discrepancy_output: Mapped[float | None] = mapped_column(Float)
    discrepancy_input: Mapped[float | None] = mapped_column(Float)

    # Goods / trade (sourced from customs integration)
    import_value: Mapped[float | None] = mapped_column(Float)
    export_value: Mapped[float | None] = mapped_column(Float)

    # Refund tracking
    refund_approved: Mapped[float | None] = mapped_column(Float)
    refund_paid_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    refund_denied_reason: Mapped[str | None] = mapped_column(String(255))

    # Metadata
    channel: Mapped[str | None] = mapped_column(String(50))  # online, paper, amended
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(String(255))
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    taxpayer: Mapped["Taxpayer"] = relationship("Taxpayer", lazy="select")  # noqa: F821

    __table_args__ = (
        Index("ix_transactions_tin_period", "taxpayer_tin", "period"),
        Index("ix_transactions_type_period", "transaction_type", "period"),
    )

    def __repr__(self) -> str:
        return f"<Transaction {self.transaction_type} {self.taxpayer_tin} {self.period}>"

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from src.models.invoice import InvoiceStatus, InvoiceType


class InvoiceBase(BaseModel):
    invoice_number: str
    seller_tin: str
    buyer_tin: str | None = None
    seller_vat_number: str
    buyer_vat_number: str | None = None
    seller_country: str = Field(..., max_length=3)
    buyer_country: str | None = Field(None, max_length=3)
    invoice_type: InvoiceType = InvoiceType.STANDARD
    invoice_date: datetime
    supply_date: datetime | None = None
    due_date: datetime | None = None
    currency: str = Field("USD", max_length=3)
    exchange_rate: float = 1.0
    net_amount: float = Field(..., gt=0)
    vat_amount: float = Field(..., ge=0)
    gross_amount: float = Field(..., gt=0)
    vat_rate: float = Field(..., ge=0, le=1)
    description: str | None = None
    commodity_code: str | None = None
    customs_declaration_number: str | None = None


class InvoiceCreate(InvoiceBase):
    external_id: str | None = None
    source_system: str | None = None
    line_items: dict | None = None

    @model_validator(mode="after")
    def validate_amounts(self) -> "InvoiceCreate":
        expected_gross = self.net_amount + self.vat_amount
        if abs(expected_gross - self.gross_amount) > 0.01:
            raise ValueError(
                f"gross_amount ({self.gross_amount}) must equal "
                f"net_amount + vat_amount ({expected_gross:.2f})"
            )
        return self


class InvoiceBatchSubmit(BaseModel):
    invoices: list[InvoiceCreate] = Field(..., min_length=1, max_length=5000)
    source_system: str | None = None
    submission_reference: str | None = None


class InvoiceUpdate(BaseModel):
    status: InvoiceStatus | None = None
    customs_declaration_number: str | None = None
    declared_by_seller: bool | None = None
    declared_by_buyer: bool | None = None


class InvoiceResponse(InvoiceBase):
    id: UUID
    external_id: str | None = None
    status: InvoiceStatus
    submission_date: datetime
    matched_invoice_id: UUID | None = None
    matching_score: float | None = None
    risk_score: float
    is_suspicious: bool
    fraud_indicators: dict | None = None
    declared_by_seller: bool
    declared_by_buyer: bool
    vat_return_period: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class MatchResultResponse(BaseModel):
    status: str
    seller_tin: str
    buyer_tin: str | None = None
    invoice_number: str
    seller_vat_amount: float
    buyer_vat_amount: float | None = None
    discrepancy: float
    discrepancy_pct: float
    risk_score: float
    flags: list[str]

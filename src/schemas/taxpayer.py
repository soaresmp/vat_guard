from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from src.models.taxpayer import BusinessType, TaxpayerStatus


class TaxpayerBase(BaseModel):
    tin: str = Field(..., description="Tax Identification Number")
    vat_number: str = Field(..., description="VAT registration number")
    name: str
    trade_name: str | None = None
    business_type: BusinessType = BusinessType.LIMITED_LIABILITY
    industry_code: str | None = None
    industry_description: str | None = None
    country_code: str = Field("XX", max_length=3)
    region: str | None = None
    address: str | None = None
    email: str | None = None
    phone: str | None = None


class TaxpayerCreate(TaxpayerBase):
    registration_date: datetime | None = None
    vat_registration_date: datetime | None = None


class TaxpayerUpdate(BaseModel):
    name: str | None = None
    trade_name: str | None = None
    status: TaxpayerStatus | None = None
    industry_code: str | None = None
    address: str | None = None
    email: str | None = None
    phone: str | None = None
    annual_turnover_reported: float | None = None
    annual_vat_declared: float | None = None
    annual_vat_paid: float | None = None
    has_foreign_directors: bool | None = None


class TaxpayerSummary(BaseModel):
    id: UUID
    tin: str
    vat_number: str
    name: str
    status: TaxpayerStatus
    risk_score: float
    compliance_score: float
    country_code: str

    model_config = {"from_attributes": True}


class TaxpayerResponse(TaxpayerBase):
    id: UUID
    status: TaxpayerStatus
    registration_date: datetime | None = None
    vat_registration_date: datetime | None = None
    risk_score: float
    compliance_score: float
    last_risk_assessment: datetime | None = None
    last_filing_date: datetime | None = None
    consecutive_late_filings: int
    total_alerts: int
    total_open_cases: int
    is_newly_registered: bool
    is_high_value_trader: bool
    is_frequent_refund_claimant: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TaxpayerRiskSummary(BaseModel):
    tin: str
    vat_number: str
    name: str
    risk_score: float
    risk_level: str
    top_risk_factors: list[str]
    component_scores: dict[str, float]
    total_alerts: int

    model_config = {"from_attributes": True}

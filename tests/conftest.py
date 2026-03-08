"""
Shared test fixtures for VATGuard test suite.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator
from unittest.mock import MagicMock
import uuid

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from src.main import app


# ── Event loop ────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── HTTP Client ───────────────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ── Domain Object Factories ───────────────────────────────────────────────────

def make_taxpayer(**kwargs) -> MagicMock:
    """Create a mock Taxpayer object."""
    defaults = {
        "id": uuid.uuid4(),
        "tin": "TIN123456",
        "vat_number": "VAT123456",
        "name": "Test Company Ltd",
        "status": "active",
        "country_code": "XX",
        "risk_score": 0.0,
        "compliance_score": 100.0,
        "consecutive_late_filings": 0,
        "total_alerts": 0,
        "total_open_cases": 0,
        "is_newly_registered": False,
        "is_high_value_trader": False,
        "is_frequent_refund_claimant": False,
        "has_foreign_directors": False,
        "annual_turnover_reported": 500_000.0,
        "annual_vat_declared": 100_000.0,
        "annual_vat_paid": 100_000.0,
        "last_filing_date": datetime.now(timezone.utc) - timedelta(days=15),
        "vat_registration_date": datetime.now(timezone.utc) - timedelta(days=365),
    }
    defaults.update(kwargs)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


def make_invoice(**kwargs) -> MagicMock:
    """Create a mock Invoice object."""
    defaults = {
        "id": uuid.uuid4(),
        "invoice_number": f"INV-{uuid.uuid4().hex[:8].upper()}",
        "seller_tin": "TIN123456",
        "buyer_tin": "TIN789012",
        "seller_vat_number": "VAT123456",
        "buyer_vat_number": "VAT789012",
        "seller_country": "XX",
        "buyer_country": "YY",
        "invoice_type": "standard",
        "invoice_date": datetime.now(timezone.utc) - timedelta(days=5),
        "net_amount": 10_000.0,
        "vat_amount": 2_000.0,
        "gross_amount": 12_000.0,
        "vat_rate": 0.20,
        "currency": "USD",
        "status": "received",
        "risk_score": 0.0,
        "is_suspicious": False,
        "declared_by_seller": False,
        "declared_by_buyer": False,
        "customs_declaration_number": None,
        "commodity_code": None,
        "fraud_indicators": None,
    }
    defaults.update(kwargs)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


def make_transaction(**kwargs) -> MagicMock:
    """Create a mock Transaction object."""
    defaults = {
        "id": uuid.uuid4(),
        "taxpayer_tin": "TIN123456",
        "transaction_type": "vat_return",
        "period": "2024-01",
        "output_vat": 20_000.0,
        "input_vat": 15_000.0,
        "net_vat_payable": 5_000.0,
        "refund_claimed": 0.0,
        "amount_paid": 5_000.0,
        "is_late": False,
    }
    defaults.update(kwargs)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock

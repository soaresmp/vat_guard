"""
Tax Management System (TMS) Integration Client
Connects to the core tax administration system to retrieve:
  - Taxpayer registration data
  - VAT return history
  - Payment records
  - Compliance scores
"""

from src.integrations.base import BaseIntegrationClient


class TaxManagementClient(BaseIntegrationClient):
    """Client for the core Tax Management System API."""

    async def get_taxpayer(self, tin: str) -> dict | None:
        """Fetch taxpayer registration details from TMS."""
        return await self._get(f"/taxpayers/{tin}")

    async def get_vat_returns(self, tin: str, from_period: str, to_period: str) -> list[dict]:
        """Fetch VAT return history for a taxpayer."""
        data = await self._get(
            f"/taxpayers/{tin}/vat-returns",
            params={"from_period": from_period, "to_period": to_period},
        )
        return data if isinstance(data, list) else (data.get("returns", []) if data else [])

    async def get_payment_history(self, tin: str) -> list[dict]:
        """Fetch VAT payment history."""
        data = await self._get(f"/taxpayers/{tin}/payments")
        return data if isinstance(data, list) else (data.get("payments", []) if data else [])

    async def get_refund_claims(self, tin: str, period: str | None = None) -> list[dict]:
        """Fetch VAT refund claim history."""
        params = {"period": period} if period else None
        data = await self._get(f"/taxpayers/{tin}/refund-claims", params=params)
        return data if isinstance(data, list) else (data.get("claims", []) if data else [])

    async def get_compliance_score(self, tin: str) -> float | None:
        """Retrieve the TMS compliance score for a taxpayer."""
        data = await self._get(f"/taxpayers/{tin}/compliance-score")
        return data.get("score") if data else None

    async def update_risk_flag(self, tin: str, risk_level: str, reason: str) -> bool:
        """Push a risk flag update back to TMS for audit queue management."""
        result = await self._post(
            f"/taxpayers/{tin}/risk-flags",
            {"risk_level": risk_level, "reason": reason, "source": "VATGuard"},
        )
        return result is not None

    async def get_linked_entities(self, tin: str) -> list[dict]:
        """Retrieve entities linked through directors, addresses, or beneficial ownership."""
        data = await self._get(f"/taxpayers/{tin}/linked-entities")
        return data if isinstance(data, list) else (data.get("entities", []) if data else [])

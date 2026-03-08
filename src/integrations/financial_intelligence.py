"""
Financial Intelligence Unit (FIU) Integration Client
Connects to the national/regional FIU to:
  - Submit Suspicious Activity Reports (SARs)
  - Check watchlists and sanctions lists
  - Retrieve Politically Exposed Person (PEP) data
  - Query cross-border financial flow data
"""

from src.integrations.base import BaseIntegrationClient


class FinancialIntelligenceClient(BaseIntegrationClient):
    """Client for the Financial Intelligence Unit API."""

    async def submit_sar(self, tin: str, name: str, reason: str) -> dict:
        """Submit a Suspicious Activity Report to the FIU."""
        result = await self._post("/sar/submit", {
            "entity_type": "taxpayer",
            "tin": tin,
            "name": name,
            "reason": reason,
            "source_system": "VATGuard",
            "report_type": "vat_fraud",
        })
        return result or {"status": "submission_failed", "tin": tin}

    async def check_watchlist(self, tin: str, name: str) -> dict:
        """
        Check if a taxpayer appears on any AML / sanctions watchlist.
        Returns: {
            "on_watchlist": bool,
            "watchlists": [...],
            "risk_rating": "low|medium|high",
        }
        """
        result = await self._post("/watchlist/check", {"tin": tin, "name": name})
        return result or {"on_watchlist": False, "watchlists": [], "risk_rating": "unknown"}

    async def check_pep(self, director_name: str, country: str) -> dict:
        """Check if a director is a Politically Exposed Person."""
        result = await self._get(
            "/pep/check",
            params={"name": director_name, "country": country},
        )
        return result or {"is_pep": False}

    async def get_previous_convictions(self, tin: str) -> list[dict]:
        """Retrieve previous fraud convictions linked to this TIN or its directors."""
        data = await self._get(f"/convictions/{tin}")
        return data if isinstance(data, list) else (data.get("convictions", []) if data else [])

    async def get_cross_border_flows(self, tin: str, period: str) -> dict | None:
        """Retrieve suspicious cross-border financial flows linked to a TIN."""
        return await self._get(f"/cross-border-flows/{tin}", params={"period": period})

    async def is_linked_to_known_fraudster(self, tin: str) -> bool:
        """Check if a taxpayer is linked (director/address/ownership) to a known fraudster."""
        result = await self._get(f"/fraud-links/{tin}")
        return bool(result and result.get("linked"))

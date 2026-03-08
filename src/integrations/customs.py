"""
Customs Management System Integration Client
Connects to customs authorities to:
  - Verify export / import declarations
  - Cross-check invoice amounts with declared customs values
  - Detect phantom exports (claimed but not physically exported)
  - Retrieve intra-community trade statistics
"""

from src.integrations.base import BaseIntegrationClient


class CustomsClient(BaseIntegrationClient):
    """Client for the Customs Management System API."""

    async def get_declaration(self, declaration_number: str) -> dict | None:
        """Retrieve a specific customs declaration."""
        return await self._get(f"/declarations/{declaration_number}")

    async def verify_export(self, invoice_number: str, seller_vat: str, amount: float) -> dict:
        """
        Verify that an export invoice corresponds to a valid customs export entry.
        Returns verification status and any discrepancies.
        """
        result = await self._post("/verify/export", {
            "invoice_number": invoice_number,
            "seller_vat_number": seller_vat,
            "declared_amount": amount,
        })
        if result is None:
            return {"verified": False, "reason": "customs_service_unavailable"}
        return result

    async def get_trader_imports(
        self,
        vat_number: str,
        from_date: str,
        to_date: str,
    ) -> list[dict]:
        """Get all import declarations for a VAT-registered trader."""
        data = await self._get(
            f"/traders/{vat_number}/imports",
            params={"from": from_date, "to": to_date},
        )
        return data if isinstance(data, list) else (data.get("imports", []) if data else [])

    async def get_trader_exports(
        self,
        vat_number: str,
        from_date: str,
        to_date: str,
    ) -> list[dict]:
        """Get all export declarations for a VAT-registered trader."""
        data = await self._get(
            f"/traders/{vat_number}/exports",
            params={"from": from_date, "to": to_date},
        )
        return data if isinstance(data, list) else (data.get("exports", []) if data else [])

    async def get_intrastat_data(
        self,
        country_code: str,
        period: str,
    ) -> dict | None:
        """
        Retrieve Intrastat (intra-community trade statistics) data.
        Used to cross-check declared intra-community transactions against
        aggregate trade statistics — a key MTIC detection method per IMF WP/07/31.
        """
        return await self._get(
            f"/intrastat/{country_code}/{period}"
        )

    async def check_commodity_risk(self, hs_code: str) -> dict:
        """Check if a commodity HS code is on the high-risk carousel fraud watch list."""
        result = await self._get(f"/commodity-risk/{hs_code}")
        return result or {"hs_code": hs_code, "risk_level": "unknown"}

    async def get_cross_border_discrepancy(
        self,
        from_country: str,
        to_country: str,
        period: str,
    ) -> dict | None:
        """
        Retrieve statistical discrepancy between declared exports from country A
        and declared imports in country B for the same period.
        This mirrors the statistical technique described in IMF WP/07/31 for
        detecting missing trader fraud at the aggregate level.
        """
        return await self._get(
            "/statistical-discrepancy",
            params={
                "from_country": from_country,
                "to_country": to_country,
                "period": period,
            },
        )

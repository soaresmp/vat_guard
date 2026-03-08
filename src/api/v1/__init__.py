from fastapi import APIRouter

from src.api.v1 import invoices, taxpayers, alerts, cases, analytics, integrations

router = APIRouter(prefix="/v1")
router.include_router(taxpayers.router, prefix="/taxpayers", tags=["Taxpayers"])
router.include_router(invoices.router, prefix="/invoices", tags=["Invoices"])
router.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
router.include_router(cases.router, prefix="/cases", tags=["Cases"])
router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
router.include_router(integrations.router, prefix="/integrations", tags=["Integrations"])

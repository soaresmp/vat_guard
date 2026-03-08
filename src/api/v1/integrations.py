"""
External System Integration Endpoints
Provides webhooks and outbound integration controls for TMS, Customs, FIU.
"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.models.taxpayer import Taxpayer

router = APIRouter()


@router.post("/tms/sync-taxpayer/{tin}")
async def sync_taxpayer_from_tms(
    tin: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Pull updated taxpayer data from the Tax Management System."""
    tp_result = await db.execute(select(Taxpayer).where(Taxpayer.tin == tin))
    taxpayer = tp_result.scalar_one_or_none()
    if not taxpayer:
        raise HTTPException(status_code=404, detail="Taxpayer not found locally")

    background_tasks.add_task(_sync_from_tms, tin)
    return {"message": f"TMS sync queued for TIN={tin}"}


@router.post("/customs/verify-export/{invoice_id}")
async def verify_export_with_customs(
    invoice_id: str,
    background_tasks: BackgroundTasks,
):
    """Verify export invoice against customs declaration database."""
    background_tasks.add_task(_verify_customs, invoice_id)
    return {"message": f"Customs verification queued for invoice {invoice_id}"}


@router.post("/fiu/report-suspicious-entity/{tin}")
async def report_to_fiu(
    tin: str,
    reason: str,
    db: AsyncSession = Depends(get_db),
):
    """Submit a Suspicious Activity Report (SAR) to the Financial Intelligence Unit."""
    from src.integrations.financial_intelligence import FinancialIntelligenceClient
    from src.config import get_settings
    settings = get_settings()

    tp_result = await db.execute(select(Taxpayer).where(Taxpayer.tin == tin))
    taxpayer = tp_result.scalar_one_or_none()
    if not taxpayer:
        raise HTTPException(status_code=404, detail="Taxpayer not found")

    client = FinancialIntelligenceClient(settings.fiu_base_url, settings.fiu_api_key)
    result = await client.submit_sar(tin=tin, name=taxpayer.name, reason=reason)
    return result


@router.get("/customs/declaration/{customs_number}")
async def lookup_customs_declaration(customs_number: str):
    """Look up a customs declaration by number."""
    from src.integrations.customs import CustomsClient
    from src.config import get_settings
    settings = get_settings()
    client = CustomsClient(settings.customs_base_url, settings.customs_api_key)
    return await client.get_declaration(customs_number)


@router.post("/webhook/invoice-submitted")
async def receive_invoice_webhook(
    payload: dict,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Webhook endpoint for external systems to push e-invoices in real time.
    Used by ERP systems, accounting platforms, and government e-invoice portals.
    """
    from src.schemas.invoice import InvoiceCreate
    from src.api.v1.invoices import submit_invoice

    try:
        invoice_data = InvoiceCreate(**payload)
        # Route to invoice submission handler
        background_tasks.add_task(_process_webhook_invoice, invoice_data.model_dump())
        return {"status": "accepted", "invoice_number": invoice_data.invoice_number}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/webhook/taxpayer-updated")
async def receive_taxpayer_update_webhook(
    payload: dict,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Webhook for TMS to push taxpayer status changes (deregistration, suspension).
    Triggers immediate risk re-assessment.
    """
    tin = payload.get("tin")
    if not tin:
        raise HTTPException(status_code=422, detail="TIN required")

    tp_result = await db.execute(select(Taxpayer).where(Taxpayer.tin == tin))
    taxpayer = tp_result.scalar_one_or_none()

    if taxpayer:
        if new_status := payload.get("status"):
            from src.models.taxpayer import TaxpayerStatus
            try:
                taxpayer.status = TaxpayerStatus(new_status)
            except ValueError:
                pass
        background_tasks.add_task(_reassess_risk, tin)
        return {"status": "updated", "tin": tin}
    return {"status": "taxpayer_not_found", "tin": tin}


# ── Background helpers ────────────────────────────────────────────────────────

async def _sync_from_tms(tin: str) -> None:
    from src.integrations.tax_management import TaxManagementClient
    from src.config import get_settings
    from src.database import AsyncSessionLocal
    settings = get_settings()
    async with AsyncSessionLocal() as db:
        client = TaxManagementClient(settings.tms_base_url, settings.tms_api_key)
        data = await client.get_taxpayer(tin)
        if data:
            result = await db.execute(select(Taxpayer).where(Taxpayer.tin == tin))
            tp = result.scalar_one_or_none()
            if tp:
                for key, val in data.items():
                    if hasattr(tp, key) and val is not None:
                        setattr(tp, key, val)
                await db.commit()


async def _verify_customs(invoice_id: str) -> None:
    from src.integrations.customs import CustomsClient
    from src.config import get_settings
    from src.database import AsyncSessionLocal
    from src.models.invoice import Invoice
    import uuid
    settings = get_settings()
    async with AsyncSessionLocal() as db:
        try:
            inv_result = await db.execute(
                select(Invoice).where(Invoice.id == uuid.UUID(invoice_id))
            )
            invoice = inv_result.scalar_one_or_none()
            if invoice and invoice.customs_declaration_number:
                client = CustomsClient(settings.customs_base_url, settings.customs_api_key)
                decl = await client.get_declaration(invoice.customs_declaration_number)
                if decl and decl.get("verified"):
                    invoice.fraud_indicators = invoice.fraud_indicators or {}
                    invoice.fraud_indicators["customs_verified"] = True
                    await db.commit()
        except Exception:
            pass


async def _process_webhook_invoice(invoice_data: dict) -> None:
    from src.database import AsyncSessionLocal
    from src.models.invoice import Invoice
    async with AsyncSessionLocal() as db:
        invoice = Invoice(**invoice_data)
        db.add(invoice)
        await db.commit()


async def _reassess_risk(tin: str) -> None:
    from src.database import AsyncSessionLocal
    from src.models.invoice import Invoice
    from src.models.transaction import Transaction
    from src.fraud_detection.engine import FraudDetectionEngine
    from sqlalchemy import or_
    async with AsyncSessionLocal() as db:
        tp_result = await db.execute(select(Taxpayer).where(Taxpayer.tin == tin))
        taxpayer = tp_result.scalar_one_or_none()
        if not taxpayer:
            return
        inv_result = await db.execute(
            select(Invoice).where(or_(Invoice.seller_tin == tin, Invoice.buyer_tin == tin)).limit(500)
        )
        tx_result = await db.execute(
            select(Transaction).where(Transaction.taxpayer_tin == tin).limit(100)
        )
        invoices = list(inv_result.scalars().all())
        transactions = list(tx_result.scalars().all())
        engine = FraudDetectionEngine()
        alerts, score = await engine.analyze_taxpayer(taxpayer, invoices, transactions)
        taxpayer.risk_score = score
        for a in alerts:
            db.add(a)
        await db.commit()

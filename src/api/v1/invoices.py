"""
Invoice Submission and Matching Endpoints
"""

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.fraud_detection.engine import FraudDetectionEngine
from src.models.invoice import Invoice, InvoiceStatus
from src.models.taxpayer import Taxpayer
from src.models.transaction import Transaction
from src.schemas.invoice import (
    InvoiceBatchSubmit, InvoiceCreate, InvoiceResponse, InvoiceUpdate,
    MatchResultResponse,
)

router = APIRouter()
_engine = FraudDetectionEngine()


@router.post("/", response_model=InvoiceResponse, status_code=status.HTTP_201_CREATED)
async def submit_invoice(
    payload: InvoiceCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Submit a single e-invoice for real-time processing and risk assessment."""
    invoice = Invoice(**payload.model_dump())
    db.add(invoice)
    await db.flush()

    # Queue background fraud analysis
    background_tasks.add_task(_analyze_invoice_async, invoice.id, invoice.seller_tin)
    return invoice


@router.post("/batch", status_code=status.HTTP_202_ACCEPTED)
async def submit_invoices_batch(
    payload: InvoiceBatchSubmit,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Bulk submit e-invoices (up to 5,000 per request).
    Processing is asynchronous – returns a submission reference.
    """
    invoices_to_add = []
    for inv_data in payload.invoices:
        inv = Invoice(**inv_data.model_dump())
        if payload.source_system:
            inv.source_system = payload.source_system
        invoices_to_add.append(inv)

    db.add_all(invoices_to_add)
    await db.flush()

    # Queue background matching job
    seller_tins = {inv.seller_tin for inv in invoices_to_add}
    background_tasks.add_task(_batch_match_async, list(seller_tins))

    return {
        "accepted": len(invoices_to_add),
        "submission_reference": payload.submission_reference,
        "message": "Invoices queued for processing",
    }


@router.post("/match", response_model=list[MatchResultResponse])
async def match_invoices(
    seller_tin: str,
    buyer_tin: str,
    period: str = Query(..., description="Period in YYYY-MM format"),
    db: AsyncSession = Depends(get_db),
):
    """
    Real-time match invoices between a seller and buyer for a given period.
    Returns only suspicious / unmatched results.
    """
    seller_stmt = select(Invoice).where(
        Invoice.seller_tin == seller_tin,
        Invoice.vat_return_period == period,
    )
    buyer_stmt = select(Invoice).where(
        Invoice.buyer_tin == buyer_tin,
        Invoice.vat_return_period == period,
    )
    s_result = await db.execute(seller_stmt)
    b_result = await db.execute(buyer_stmt)

    seller_invs = list(s_result.scalars().all())
    buyer_invs = list(b_result.scalars().all())

    suspicious = await _engine.analyze_invoice_match(seller_invs, buyer_invs)
    return [MatchResultResponse(**r) for r in suspicious]


@router.get("/", response_model=list[InvoiceResponse])
async def list_invoices(
    seller_tin: str | None = None,
    buyer_tin: str | None = None,
    is_suspicious: bool | None = None,
    invoice_type: str | None = None,
    period: str | None = None,
    min_risk_score: float | None = Query(None, ge=0, le=100),
    limit: int = Query(50, le=500),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List invoices with filters."""
    stmt = select(Invoice)
    if seller_tin:
        stmt = stmt.where(Invoice.seller_tin == seller_tin)
    if buyer_tin:
        stmt = stmt.where(Invoice.buyer_tin == buyer_tin)
    if is_suspicious is not None:
        stmt = stmt.where(Invoice.is_suspicious == is_suspicious)
    if invoice_type:
        stmt = stmt.where(Invoice.invoice_type == invoice_type)
    if period:
        stmt = stmt.where(Invoice.vat_return_period == period)
    if min_risk_score is not None:
        stmt = stmt.where(Invoice.risk_score >= min_risk_score)
    stmt = stmt.order_by(Invoice.risk_score.desc()).offset(offset).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/{invoice_id}", response_model=InvoiceResponse)
async def get_invoice(invoice_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Invoice).where(Invoice.id == invoice_id))
    invoice = result.scalar_one_or_none()
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return invoice


@router.patch("/{invoice_id}", response_model=InvoiceResponse)
async def update_invoice(
    invoice_id: UUID,
    payload: InvoiceUpdate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Invoice).where(Invoice.id == invoice_id))
    invoice = result.scalar_one_or_none()
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(invoice, field, value)
    return invoice


# ── Background tasks ──────────────────────────────────────────────────────────

async def _analyze_invoice_async(invoice_id: UUID, seller_tin: str) -> None:
    """Background task: run fraud analysis after invoice submission."""
    from src.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        try:
            inv_result = await db.execute(select(Invoice).where(Invoice.id == invoice_id))
            invoice = inv_result.scalar_one_or_none()
            if not invoice:
                return
            tp_result = await db.execute(
                select(Taxpayer).where(Taxpayer.tin == seller_tin)
            )
            taxpayer = tp_result.scalar_one_or_none()
            if not taxpayer:
                return

            tx_result = await db.execute(
                select(Transaction).where(Transaction.taxpayer_tin == seller_tin).limit(50)
            )
            transactions = list(tx_result.scalars().all())

            alerts, risk_score = await _engine.analyze_taxpayer(
                taxpayer, [invoice], transactions
            )
            if alerts:
                invoice.is_suspicious = True
                invoice.risk_score = max(inv.risk_score for inv in [invoice])
                for alert in alerts:
                    alert.invoice_id = invoice_id
                    db.add(alert)
            await db.commit()
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Background invoice analysis failed")


async def _batch_match_async(seller_tins: list[str]) -> None:
    """Background task: run matching for batch submissions."""
    pass  # Delegated to Celery worker in production

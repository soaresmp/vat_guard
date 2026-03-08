"""
Celery application configuration for VATGuard background workers.
"""

from celery import Celery
from src.config import get_settings

settings = get_settings()

celery_app = Celery(
    "vatguard",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["src.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Retry policy
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Rate limiting (protect downstream systems)
    task_annotations={
        "src.workers.tasks.run_full_analysis": {"rate_limit": "100/m"},
        "src.workers.tasks.batch_invoice_match": {"rate_limit": "50/m"},
        "src.workers.tasks.sync_from_tms": {"rate_limit": "200/m"},
    },
    # Periodic tasks (scheduled via Celery Beat)
    beat_schedule={
        "daily-high-risk-reassessment": {
            "task": "src.workers.tasks.reassess_high_risk_taxpayers",
            "schedule": 86400,  # Every 24 hours
        },
        "hourly-unmatched-invoice-scan": {
            "task": "src.workers.tasks.scan_unmatched_invoices",
            "schedule": 3600,  # Every hour
        },
        "daily-vat-gap-calculation": {
            "task": "src.workers.tasks.calculate_vat_gap",
            "schedule": 86400,
        },
        "weekly-benford-scan": {
            "task": "src.workers.tasks.run_benford_scan",
            "schedule": 604800,  # Weekly
        },
    },
)

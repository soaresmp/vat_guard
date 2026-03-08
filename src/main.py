"""
VATGuard – FastAPI Application Entry Point
==========================================
Real-time VAT Fraud Detection & Monitoring Platform
for Tax Authorities

Fraud types detected (per IMF WP/07/31 and IMF HTN 2023/001):
  - Missing Trader Intra-Community (MTIC) Fraud
  - Carousel Fraud
  - Contra-Trading
  - Invoice Mill / Bogus Trader
  - VAT Refund Fraud / Phantom Exporter
  - Benford's Law invoice fabrication detection
  - Network/graph anomaly detection
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.v1 import router as v1_router
from src.config import get_settings
from src.database import init_db
from src.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info(
        "Starting VATGuard",
        extra={"env": settings.app_env, "version": "1.0.0"},
    )
    if settings.app_env == "development":
        await init_db()
        logger.info("Database tables initialised (development mode)")
    yield
    logger.info("VATGuard shutdown complete")


app = FastAPI(
    title="VATGuard – VAT Fraud Detection Platform",
    description=(
        "Real-time VAT monitoring and fraud detection platform for tax authorities. "
        "Processes electronic invoice data and VAT-registered taxpayer data to detect "
        "and prevent VAT fraud including MTIC, carousel, invoice mill, refund fraud, "
        "and contra-trading schemes. Integrates with Tax Management Systems, Customs "
        "Management Systems, and Financial Intelligence Units."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_debug else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception Handlers ────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred. Please contact support."},
    )


# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(v1_router, prefix="/api")


@app.get("/health", tags=["Health"])
async def health_check():
    """Platform health check endpoint."""
    return {
        "status": "healthy",
        "platform": "VATGuard",
        "version": "1.0.0",
        "environment": settings.app_env,
    }


@app.get("/", tags=["Root"])
async def root():
    return {
        "platform": "VATGuard – VAT Fraud Detection Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }

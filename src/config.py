"""
VATGuard – Application Configuration
=====================================
All settings are driven by environment variables (via .env file or shell).
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "VATGuard"
    app_env: Literal["development", "staging", "production"] = "development"
    app_secret_key: str = "change-me-in-production"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://vatguard:vatguard@localhost:5432/vatguard"
    database_sync_url: str = "postgresql://vatguard:vatguard@localhost:5432/vatguard"

    # ── Redis / Celery ────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── JWT ───────────────────────────────────────────────────────────────────
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60
    jwt_refresh_token_expire_days: int = 7

    # ── Risk Score Thresholds ─────────────────────────────────────────────────
    risk_score_high_threshold: int = 75
    risk_score_critical_threshold: int = 90

    # ── Fraud Detection Tuning ────────────────────────────────────────────────
    carousel_min_chain_depth: int = 3
    missing_trader_days_inactive: int = 90
    invoice_mill_min_invoices_per_day: int = 50
    refund_fraud_ratio_threshold: float = 0.8
    benford_chi2_significance: float = 0.05

    # ── External Integrations ─────────────────────────────────────────────────
    tms_base_url: str = "https://tms.authority.gov/api/v1"
    tms_api_key: str = ""
    tms_timeout_seconds: int = 30

    customs_base_url: str = "https://customs.authority.gov/api/v1"
    customs_api_key: str = ""
    customs_timeout_seconds: int = 30

    fiu_base_url: str = "https://fiu.authority.gov/api/v1"
    fiu_api_key: str = ""
    fiu_timeout_seconds: int = 30

    business_register_base_url: str = "https://register.authority.gov/api/v1"
    business_register_api_key: str = ""

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"


@lru_cache
def get_settings() -> Settings:
    return Settings()

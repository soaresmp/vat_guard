# VATGuard – VAT Fraud Detection & Monitoring Platform

Real-time Value Added Tax (VAT) fraud detection platform for tax authorities.
Processes electronic invoice data and VAT-registered taxpayer data to detect
and prevent VAT fraud in real time.

## Fraud Types Detected

Based on IMF Working Paper WP/07/31 (Keen & Smith, 2007) and IMF How-To Note
HTN 2023/001 (Andrew & Baer, 2023):

| Fraud Type | Description | Key Detection Signals |
|---|---|---|
| **Missing Trader (MTIC)** | Trader collects VAT on sales then disappears without remitting | Inactive periods, buyer claims without seller declarations, post-import inactivity |
| **Carousel Fraud** | Circular chain of trades to generate false refund claims | Graph cycle detection, rapid resale, high-risk commodities (phones, CPUs, gold) |
| **Contra-Trading** | Two parallel chains whose VAT liabilities cancel out | Near-zero net VAT on high gross turnover, diverse unrelated commodity trading |
| **Invoice Mill / Bogus Trader** | Shell company generating fraudulent invoices | High volume per day, round amounts, no purchase invoices, Benford's Law anomaly |
| **VAT Refund Fraud** | False or inflated refund claims | High refund-to-output ratio, phantom exporters, refund spikes, frequent amendments |
| **Phantom Exporter** | Claims zero-rating for exports that never occurred | Export invoices without customs declarations |
| **Statistical Anomalies** | Manufactured invoice amounts | Benford's Law chi-squared test, digit frequency analysis |
| **Network Anomalies** | Suspicious transaction network structures | Graph betweenness centrality, strongly connected components |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Tax Authority Users                       │
│           Analysts │ Investigators │ Management │ Auditors       │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST API / Webhooks
┌────────────────────────────▼────────────────────────────────────┐
│                    VATGuard FastAPI Application                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐ │
│  │Taxpayers │ │ Invoices │ │  Alerts  │ │  Cases   │ │Analyt.│ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └───────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  Fraud Detection Engine                           │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │               Rule-Based Detectors                           │ │
│  │  Missing Trader │ Carousel │ Invoice Mill │ Refund │ Contra  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │               ML / Statistical Analytics                     │ │
│  │   Isolation Forest │ Benford's Law │ Z-Score Anomaly         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │               Graph Analytics                                │ │
│  │   Network Analyzer │ Circular Chain Detector │ Centrality    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │               Real-Time Invoice Matcher                      │ │
│  │   Seller↔Buyer matching │ Amount reconciliation │ Duplicates │ │
│  └─────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                   External Integrations                           │
│  Tax Management System │ Customs System │ Financial Intel. Unit  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
vat_guard/
├── src/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Settings (pydantic-settings)
│   ├── database.py                # Async SQLAlchemy setup
│   ├── models/                    # SQLAlchemy ORM models
│   │   ├── taxpayer.py            # VAT-registered taxpayer
│   │   ├── invoice.py             # Electronic invoice
│   │   ├── transaction.py         # VAT return / payment records
│   │   ├── alert.py               # Fraud alert
│   │   ├── case.py                # Investigation case
│   │   └── audit.py               # Immutable audit log
│   ├── schemas/                   # Pydantic request/response schemas
│   ├── api/v1/                    # REST API routes
│   │   ├── taxpayers.py           # Taxpayer CRUD + risk assessment
│   │   ├── invoices.py            # Invoice submission + batch processing
│   │   ├── alerts.py              # Alert management + escalation
│   │   ├── cases.py               # Case management
│   │   ├── analytics.py           # Dashboard + VAT gap + heatmap
│   │   └── integrations.py        # Webhooks + external system sync
│   ├── fraud_detection/
│   │   ├── engine.py              # Orchestrates all detectors
│   │   ├── rules/
│   │   │   ├── missing_trader.py  # MTIC detection
│   │   │   ├── carousel.py        # Carousel fraud detection
│   │   │   ├── invoice_mill.py    # Invoice mill / bogus trader
│   │   │   ├── refund_fraud.py    # VAT refund fraud
│   │   │   └── contra_trading.py  # Contra-trading detection
│   │   ├── ml/
│   │   │   ├── anomaly_detector.py # Isolation Forest
│   │   │   ├── risk_scorer.py      # Composite risk scoring
│   │   │   └── benford_analysis.py # Benford's Law chi-squared test
│   │   ├── graph/
│   │   │   ├── network_analyzer.py # NetworkX graph analytics
│   │   │   └── circular_detector.py # Carousel cycle detection
│   │   └── realtime/
│   │       └── invoice_matcher.py  # Real-time invoice matching
│   ├── integrations/
│   │   ├── tax_management.py      # TMS integration client
│   │   ├── customs.py             # Customs system client
│   │   └── financial_intelligence.py # FIU client
│   └── workers/
│       ├── celery_app.py          # Celery configuration + beat schedule
│       └── tasks.py               # Background fraud detection tasks
├── tests/
│   ├── test_fraud_detection/
│   │   ├── test_missing_trader.py
│   │   ├── test_carousel.py
│   │   ├── test_refund_fraud.py
│   │   ├── test_benford.py
│   │   ├── test_invoice_matcher.py
│   │   └── test_risk_scorer.py
│   └── test_api/
│       └── test_health.py
├── alembic/                       # Database migrations
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Quick Start

### 1. Clone and configure

```bash
cp .env.example .env
# Edit .env with your database and integration credentials
```

### 2. Run with Docker Compose

```bash
docker-compose up -d
```

This starts:
- **PostgreSQL** (port 5432) – primary data store
- **Redis** (port 6379) – cache and Celery broker
- **VATGuard API** (port 8000) – REST API
- **Celery Worker** – background fraud analysis tasks
- **Celery Beat** – scheduled scans (hourly, daily, weekly)
- **Flower** (port 5555) – Celery task monitoring UI

### 3. Access the API

- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc
- **Health check**: http://localhost:8000/health
- **Task monitor**: http://localhost:5555

### 4. Run tests

```bash
pip install -r requirements.txt
pytest
```

## Key API Endpoints

### Taxpayers
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/taxpayers/` | List with risk score filtering |
| `POST` | `/api/v1/taxpayers/` | Register new taxpayer |
| `POST` | `/api/v1/taxpayers/{tin}/assess-risk` | Trigger full fraud analysis |
| `GET` | `/api/v1/taxpayers/{tin}/network` | Transaction network graph |

### Invoices
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/invoices/` | Submit single e-invoice (real-time) |
| `POST` | `/api/v1/invoices/batch` | Bulk submit up to 5,000 invoices |
| `POST` | `/api/v1/invoices/match` | Real-time buyer/seller matching |

### Alerts
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/alerts/` | List with fraud type / severity filters |
| `GET` | `/api/v1/alerts/stats` | Dashboard KPIs |
| `POST` | `/api/v1/alerts/{id}/escalate` | Escalate to formal case |

### Analytics
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/analytics/dashboard` | Executive KPI dashboard |
| `GET` | `/api/v1/analytics/vat-gap` | VAT gap analysis by period |
| `GET` | `/api/v1/analytics/benford/{tin}` | Benford's Law analysis |
| `GET` | `/api/v1/analytics/risk-heatmap` | Risk by industry / region |
| `GET` | `/api/v1/analytics/top-risk-taxpayers` | Prioritised audit list |

### Integrations
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/integrations/tms/sync-taxpayer/{tin}` | Pull from TMS |
| `POST` | `/api/v1/integrations/customs/verify-export/{id}` | Verify with customs |
| `POST` | `/api/v1/integrations/fiu/report-suspicious-entity/{tin}` | File SAR |
| `POST` | `/api/v1/integrations/webhook/invoice-submitted` | Receive e-invoice webhook |

## Scheduled Tasks

| Task | Schedule | Description |
|------|----------|-------------|
| `reassess_high_risk_taxpayers` | Daily | Re-run full analysis on all taxpayers with risk ≥ 60 |
| `scan_unmatched_invoices` | Hourly | Flag buyers claiming input VAT with no matching seller declaration |
| `run_benford_scan` | Weekly | Benford's Law analysis across all high-volume sellers |
| `calculate_vat_gap` | Daily | Estimate VAT gap for the previous month |

## Risk Scoring

The composite risk score (0–100) is computed from five weighted components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Rule detections | 40% | Triggered fraud detection rules |
| Anomaly score | 20% | Isolation Forest + Z-score outlier detection |
| Compliance history | 20% | Filing patterns, late submissions, status |
| Network centrality | 10% | Position in suspicious transaction networks |
| Third-party signals | 10% | FIU watchlists, customs mismatches, convictions |

**Risk levels**: Low (< 40) · Medium (40–74) · High (75–89) · Critical (≥ 90)

## References

- Keen, M. & Smith, S. (2007). *VAT Fraud and Evasion: What Do We Know, and What Can Be Done?* IMF Working Paper WP/07/31.
- Andrew, C. & Baer, K. (2023). *How to Combat Value-Added Tax Refund Fraud.* IMF Fiscal Affairs Department How-To Note 2023/001.

"""
Circular Chain Detector
========================
Specialised detector for finding and scoring circular transaction chains
(the defining structural characteristic of carousel fraud).

Provides:
  - Fast cycle detection using Johnson's algorithm
  - Chain scoring by total VAT value and depth
  - Identification of the "missing trader" (typically the entry-point importer)
  - Cross-border chain flagging
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class CircularChain:
    chain: list[str]                # Ordered list of TINs forming the cycle
    chain_length: int
    total_vat: float
    cross_border: bool
    potential_missing_trader: str | None   # TIN of suspected defaulting trader
    risk_score: float
    evidence: dict[str, Any] = field(default_factory=dict)


class CircularChainDetector:
    """Detects and characterises circular VAT chains."""

    MIN_CHAIN_LENGTH = 3
    MAX_CYCLES_TO_ANALYSE = 100     # Avoid performance issues on large graphs

    def detect(
        self,
        invoices: list[Any],
        cross_border_tins: set[str] | None = None,
    ) -> list[CircularChain]:
        """
        Detect all circular chains in the transaction network.

        Args:
            invoices: List of Invoice ORM objects.
            cross_border_tins: Set of TINs known to engage in cross-border trade.

        Returns:
            List of CircularChain objects, sorted by risk score descending.
        """
        cross_border_tins = cross_border_tins or set()
        G, edge_data = self._build_graph_with_data(invoices)

        if G.number_of_nodes() < self.MIN_CHAIN_LENGTH:
            return []

        try:
            raw_cycles = [
                c for c in nx.simple_cycles(G)
                if len(c) >= self.MIN_CHAIN_LENGTH
            ]
        except Exception as exc:
            logger.error("Cycle detection failed: %s", exc)
            return []

        # Limit to avoid excessive computation
        raw_cycles = raw_cycles[: self.MAX_CYCLES_TO_ANALYSE]

        results: list[CircularChain] = []
        for cycle in raw_cycles:
            chain_info = self._analyze_cycle(
                cycle, G, edge_data, cross_border_tins
            )
            results.append(chain_info)

        return sorted(results, key=lambda c: c.risk_score, reverse=True)

    def _build_graph_with_data(
        self, invoices: list[Any]
    ) -> tuple[nx.DiGraph, dict]:
        """Build graph and capture edge-level data for scoring."""
        G = nx.DiGraph()
        edge_data: dict[tuple, dict] = {}

        for inv in invoices:
            seller = getattr(inv, "seller_tin", None)
            buyer = getattr(inv, "buyer_tin", None)
            if not (seller and buyer):
                continue

            G.add_edge(seller, buyer)
            key = (seller, buyer)
            if key not in edge_data:
                edge_data[key] = {
                    "total_vat": 0.0,
                    "invoice_count": 0,
                    "invoice_types": set(),
                    "countries": set(),
                }
            edge_data[key]["total_vat"] += float(getattr(inv, "vat_amount", 0) or 0)
            edge_data[key]["invoice_count"] += 1
            inv_type = getattr(inv, "invoice_type", "standard")
            edge_data[key]["invoice_types"].add(inv_type)
            for country in (
                getattr(inv, "seller_country", None),
                getattr(inv, "buyer_country", None),
            ):
                if country:
                    edge_data[key]["countries"].add(country)

        return G, edge_data

    def _analyze_cycle(
        self,
        cycle: list[str],
        G: nx.DiGraph,
        edge_data: dict,
        cross_border_tins: set[str],
    ) -> CircularChain:
        """Score a single cycle and identify structural indicators."""
        # Total VAT flowing around the chain
        total_vat = 0.0
        chain_countries: set[str] = set()
        has_cross_border = False

        for i, tin in enumerate(cycle):
            next_tin = cycle[(i + 1) % len(cycle)]
            key = (tin, next_tin)
            data = edge_data.get(key, {})
            total_vat += data.get("total_vat", 0.0)
            chain_countries.update(data.get("countries", set()))
            inv_types = data.get("invoice_types", set())
            if inv_types & {"export", "import", "intra_community"}:
                has_cross_border = True

        # Also check if any node is known cross-border
        if cross_border_tins & set(cycle):
            has_cross_border = True

        # Identify potential missing trader:
        # Heuristic: node with highest in-degree in the cycle
        # (receives most VAT credits but likely doesn't remit)
        in_degrees = {tin: G.in_degree(tin) for tin in cycle}
        potential_mt = max(in_degrees, key=lambda t: in_degrees[t])

        risk_score = self._score_cycle(cycle, total_vat, has_cross_border)

        return CircularChain(
            chain=cycle,
            chain_length=len(cycle),
            total_vat=total_vat,
            cross_border=has_cross_border,
            potential_missing_trader=potential_mt,
            risk_score=round(risk_score, 2),
            evidence={
                "chain_countries": list(chain_countries),
                "chain_in_degrees": in_degrees,
            },
        )

    def _score_cycle(
        self, cycle: list[str], total_vat: float, cross_border: bool
    ) -> float:
        """Score a cycle by length, VAT value, and cross-border involvement."""
        score = 30.0  # Base score for any detected cycle
        score += min(30, (len(cycle) - self.MIN_CHAIN_LENGTH) * 5)
        score += min(25, total_vat / 10_000)
        if cross_border:
            score += 15
        return min(100.0, score)

"""
Transaction Network Analyzer
==============================
Builds and analyses the directed graph of VAT transactions to detect
suspicious network structures characteristic of MTIC and carousel fraud.

Uses NetworkX for graph construction and centrality analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class NetworkAnalysisResult:
    node_count: int
    edge_count: int
    cycle_count: int
    cycles: list[list[str]]
    centrality: dict[str, float]         # tin → betweenness centrality
    suspicious_nodes: list[str]          # High-centrality nodes (buffer traders)
    weakly_connected_components: int
    strongly_connected_components: int
    graph_density: float
    total_vat_in_network: float
    summary: str
    risk_score: float                    # 0–100 network-level risk


class NetworkAnalyzer:
    """
    Builds a directed weighted graph from invoice data and derives
    network-level fraud risk signals.
    """

    SUSPICIOUS_CENTRALITY_THRESHOLD = 0.1   # Betweenness centrality
    MIN_CYCLE_LENGTH = 3
    MIN_VAT_THRESHOLD = 5_000

    def build_graph(self, invoices: list[Any]) -> nx.DiGraph:
        """Build directed graph from invoice list: node = TIN, edge = invoice."""
        G = nx.DiGraph()
        for inv in invoices:
            seller = getattr(inv, "seller_tin", None)
            buyer = getattr(inv, "buyer_tin", None)
            vat = float(getattr(inv, "vat_amount", 0) or 0)
            inv_id = str(getattr(inv, "id", ""))

            if not (seller and buyer):
                continue

            # Add/update nodes
            for tin in (seller, buyer):
                if tin not in G:
                    G.add_node(tin, vat_collected=0.0, vat_paid=0.0, invoice_ids=[])

            # Add edge (accumulate if already exists)
            if G.has_edge(seller, buyer):
                G[seller][buyer]["weight"] += vat
                G[seller][buyer]["invoice_count"] += 1
                G[seller][buyer]["invoice_ids"].append(inv_id)
            else:
                G.add_edge(seller, buyer, weight=vat, invoice_count=1, invoice_ids=[inv_id])

            G.nodes[seller]["vat_collected"] = G.nodes[seller].get("vat_collected", 0) + vat
        return G

    def analyze(self, invoices: list[Any]) -> NetworkAnalysisResult:
        """Run full network analysis and return results."""
        G = self.build_graph(invoices)

        if G.number_of_nodes() == 0:
            return self._empty_result()

        # ── Cycles ────────────────────────────────────────────────────────────
        cycles = [
            c for c in nx.simple_cycles(G)
            if len(c) >= self.MIN_CYCLE_LENGTH
        ]

        # ── Centrality ────────────────────────────────────────────────────────
        try:
            betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
        except Exception:
            betweenness = {n: 0.0 for n in G.nodes()}

        suspicious_nodes = [
            tin for tin, centrality in betweenness.items()
            if centrality >= self.SUSPICIOUS_CENTRALITY_THRESHOLD
        ]

        # ── Components ───────────────────────────────────────────────────────
        wcc = nx.number_weakly_connected_components(G)
        scc = nx.number_strongly_connected_components(G)
        density = nx.density(G)

        total_vat = sum(
            G[u][v]["weight"]
            for u, v in G.edges()
        )

        # ── Risk Score ────────────────────────────────────────────────────────
        risk_score = self._compute_risk_score(G, cycles, suspicious_nodes, total_vat)

        summary = self._build_summary(G, cycles, suspicious_nodes, scc, density, risk_score)

        return NetworkAnalysisResult(
            node_count=G.number_of_nodes(),
            edge_count=G.number_of_edges(),
            cycle_count=len(cycles),
            cycles=[list(c) for c in cycles[:10]],
            centrality={k: round(v, 4) for k, v in betweenness.items()},
            suspicious_nodes=suspicious_nodes,
            weakly_connected_components=wcc,
            strongly_connected_components=scc,
            graph_density=round(density, 4),
            total_vat_in_network=total_vat,
            summary=summary,
            risk_score=round(risk_score, 2),
        )

    def get_node_metrics(self, G: nx.DiGraph, tin: str) -> dict[str, float]:
        """Return per-node centrality and connectivity metrics."""
        if tin not in G:
            return {}
        try:
            bc = nx.betweenness_centrality(G, weight="weight")
            return {
                "betweenness_centrality": round(bc.get(tin, 0.0), 4),
                "in_degree": G.in_degree(tin),
                "out_degree": G.out_degree(tin),
                "total_vat_received": sum(
                    G[u][tin]["weight"] for u in G.predecessors(tin)
                ),
                "total_vat_sent": sum(
                    G[tin][v]["weight"] for v in G.successors(tin)
                ),
            }
        except Exception:
            return {}

    def _compute_risk_score(
        self, G: nx.DiGraph, cycles: list, suspicious_nodes: list, total_vat: float
    ) -> float:
        score = 0.0
        if cycles:
            score += min(50, len(cycles) * 15)
        if suspicious_nodes:
            score += min(25, len(suspicious_nodes) * 8)
        scc = nx.number_strongly_connected_components(G)
        if scc > 1:
            score += min(15, scc * 3)
        if total_vat > self.MIN_VAT_THRESHOLD:
            score += min(10, total_vat / 100_000)
        return min(100.0, score)

    def _build_summary(
        self, G: nx.DiGraph, cycles: list, suspicious: list,
        scc: int, density: float, risk: float
    ) -> str:
        parts = [
            f"Network: {G.number_of_nodes()} entities, {G.number_of_edges()} transaction edges.",
            f"Circular chains detected: {len(cycles)}.",
            f"High-centrality (buffer trader) nodes: {len(suspicious)}.",
            f"Strongly connected components: {scc}.",
            f"Graph density: {density:.4f}.",
            f"Network risk score: {risk:.1f}/100.",
        ]
        return " ".join(parts)

    @staticmethod
    def _empty_result() -> NetworkAnalysisResult:
        return NetworkAnalysisResult(
            node_count=0, edge_count=0, cycle_count=0, cycles=[],
            centrality={}, suspicious_nodes=[], weakly_connected_components=0,
            strongly_connected_components=0, graph_density=0.0,
            total_vat_in_network=0.0, summary="No transaction data.", risk_score=0.0,
        )

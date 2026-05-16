"""
Rule-based Risk Dependency Graph builder.

Queries Silver (kg_canonical_entities) to detect missing or weak clauses,
then writes Risk vertices and INCREASES_RISK_FOR / CAUSES edges into AGE.

No LLM.  Rules are deterministic gap-detection on canonical Silver entities.
"""
from __future__ import annotations

import logging
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

# (risk_name, clause_label, severity, risk_type)
_MISSING_CLAUSE_RULES: list[tuple[str, str, str, str]] = [
    ("Missing Indemnity Clause",       "IndemnityClause",       "HIGH",   "compliance_gap"),
    ("Missing Termination Clause",     "TerminationClause",     "HIGH",   "compliance_gap"),
    ("Uncapped Liability",             "LiabilityClause",       "HIGH",   "financial_risk"),
    ("No Governing Law",               "Jurisdiction",          "MEDIUM", "legal_risk"),
    ("No Confidentiality Obligation",  "ConfidentialityClause", "MEDIUM", "compliance_gap"),
]

# Risk A CAUSES Risk B — both must be triggered for the edge to be created.
_CAUSATION_RULES: list[tuple[str, str]] = [
    ("No Governing Law",           "Missing Indemnity Clause"),
    ("No Governing Law",           "Missing Termination Clause"),
    ("Missing Indemnity Clause",   "Uncapped Liability"),
]


class RiskGraphBuilder:
    """
    Infers Risk vertices and edges from Silver canonical entities.

    Call build() after GoldProjector.project() for the same contract.
    """

    def __init__(self, pool: asyncpg.Pool, age_store: Any) -> None:
        self._pool = pool
        self._age = age_store

    async def build(self, contract_id: str) -> dict[str, int]:
        """Build risk subgraph for one contract. Returns counts of risks and edges."""
        existing_labels = await self._existing_labels(contract_id)
        party_age_ids   = await self._party_age_ids(contract_id)

        if not party_age_ids:
            logger.debug("[risk] contract %s has no Party vertices — skipping", contract_id)
            return {"risks": 0, "risk_edges": 0}

        triggered: dict[str, str] = {}  # risk_name → AGE vertex ID

        for risk_name, clause_label, severity, risk_type in _MISSING_CLAUSE_RULES:
            if clause_label in existing_labels:
                continue

            age_id = await self._age.upsert_entity(
                name=risk_name,
                entity_type="Risk",
                document_id=contract_id,
                metadata={"severity": severity, "risk_type": risk_type},
            )
            triggered[risk_name] = age_id

            for party_age_id in party_age_ids:
                await self._age.add_relationship(
                    source_id=age_id,
                    target_id=party_age_id,
                    relationship_type="INCREASES_RISK_FOR",
                    document_id=contract_id,
                    properties={"severity": severity, "risk_type": risk_type},
                )

        edge_count = len(triggered) * len(party_age_ids)

        for cause_name, effect_name in _CAUSATION_RULES:
            if cause_name in triggered and effect_name in triggered:
                await self._age.add_relationship(
                    source_id=triggered[cause_name],
                    target_id=triggered[effect_name],
                    relationship_type="CAUSES",
                    document_id=contract_id,
                    properties={},
                )
                edge_count += 1

        logger.info(
            "[risk] contract %s -> %d risk vertices, %d edges",
            contract_id, len(triggered), edge_count,
        )
        return {"risks": len(triggered), "risk_edges": edge_count}

    async def _existing_labels(self, contract_id: str) -> frozenset[str]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT label FROM kg_canonical_entities"
                " WHERE contract_id = $1::uuid",
                contract_id,
            )
        return frozenset(r["label"] for r in rows)

    async def _party_age_ids(self, contract_id: str) -> list[str]:
        """Return AGE vertex UUIDs for Party vertices belonging to this contract."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT canonical_name FROM kg_canonical_entities"
                " WHERE contract_id = $1::uuid AND label = 'Party'",
                contract_id,
            )
        if not rows:
            return []

        ids: list[str] = []
        for row in rows:
            age_id = await self._age.upsert_entity(
                name=row["canonical_name"],
                entity_type="Party",
                document_id=contract_id,
            )
            ids.append(age_id)
        return ids

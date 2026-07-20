# Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
# See LICENSE file in the project root for details.

"""
Knowledge Graph module.

Core graph store backed by Apache AGE.

    AgeGraphStore  — Apache AGE / Cypher graph store
    create_kg_store() — factory; returns AgeGraphStore

NOTE: The CUAD legal ingestion and retrieval modules (ExtractionPipeline,
build_cuad_kg, NL2CypherConverter, GraphRouter, IntentParser, etc.) were moved
to misc/kg_legal_cuad/kg_legal/. They are no longer importable from kg.*
The nl_graph_query tool in rag/agent/rag_agent.py will fail at call time
until those imports are updated or the tool is removed.
"""

from kg.age_graph_store import AgeGraphStore
from kg.entity_index import EntityIndex


def create_kg_store() -> AgeGraphStore:
    """Return an AgeGraphStore instance (Apache AGE, port 5433)."""
    return AgeGraphStore()


__all__ = [
    "AgeGraphStore",
    "EntityIndex",
    "create_kg_store",
]

# Copyright 2024 The Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Knowledge Graph module for RAG.

Apache AGE-backed knowledge graph built from CUAD legal contract annotations.

---
EXTRACTION  (populates the graph — uses LLM)
---
    AgeGraphStore      — Apache AGE Cypher graph
    build_cuad_kg()    — fast ingest from cuad_eval.json CUAD annotations
    ExtractionPipeline — Bronze → Silver → Gold medallion pipeline
                         Bronze: immutable JSONB per chunk + JSON to entity_relationships/jsons/
                         Silver: deduplicated canonical tables in PostgreSQL
                         Gold:   distinct vertex labels projected into AGE
    RiskGraphBuilder   — rule-based risk inference (no LLM); runs after Gold

---
RETRIEVAL  (queries the graph — no LLM, deterministic)
---
    GraphType           — enum: ENTITY, HIERARCHY, LINEAGE, RISK
    GraphRouter         — regex router: question → list[GraphType]
    get_schema()        — compact schema string for selected graph types
    IntentParser        — regex parser: question → IntentMatch(intent, params)
    QUERY_CAPABILITIES  — registry: intent name → Cypher builder function
    NL2CypherConverter  — orchestrator: IntentParser + QUERY_CAPABILITIES → Cypher
                          No LLM calls; no prompt injection surface.

---
Ontology constants (single source of truth)
---
    VALID_LABELS, VALID_REL_TYPES, ENTITY_TYPE_MAP, RELATIONSHIP_MAP,
    entity_type_for(), relationship_type_for()

Usage:
    from kg import create_kg_store, NL2CypherConverter

    store     = create_kg_store()
    converter = NL2CypherConverter()
    await store.initialize()

    cypher = await converter.convert("Which parties indemnify each other?")
    result = await store.run_cypher_query(cypher)
    await store.close()
"""

from kg.age_graph_store import AgeGraphStore
from kg.legal.cuad_kg_ingest import build_cuad_kg
from kg.legal.extraction_pipeline import ExtractionPipeline
from kg.legal.cuad_ontology import (
    VALID_LABELS,
    VALID_REL_TYPES,
    ENTITY_TYPE_MAP,
    RELATIONSHIP_MAP,
    entity_type_for,
    relationship_type_for,
)
from kg.legal.schemas import GraphType, get_schema
from kg.legal.graph_router import GraphRouter
from kg.legal.intent_parser import IntentParser, IntentMatch
from kg.legal.query_builder import QUERY_CAPABILITIES
from kg.legal.nl2cypher import NL2CypherConverter
from kg.legal.risk_graph_builder import RiskGraphBuilder


def create_kg_store() -> AgeGraphStore:
    """Return an AgeGraphStore instance (Apache AGE, port 5433)."""
    return AgeGraphStore()


__all__ = [
    "AgeGraphStore",
    "build_cuad_kg",
    "ExtractionPipeline",
    "create_kg_store",
    "GraphType",
    "GraphRouter",
    "get_schema",
    "IntentParser",
    "IntentMatch",
    "QUERY_CAPABILITIES",
    "NL2CypherConverter",
    "RiskGraphBuilder",
    "VALID_LABELS",
    "VALID_REL_TYPES",
    "ENTITY_TYPE_MAP",
    "RELATIONSHIP_MAP",
    "entity_type_for",
    "relationship_type_for",
]

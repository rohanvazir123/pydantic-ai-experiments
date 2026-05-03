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
    AgeGraphStore        — Apache AGE Cypher graph (active backend)
    build_cuad_kg()      — fast ingest from cuad_eval.json CUAD annotations
    LegalEntityExtractor — 5-pass LLM extraction (entity/rel/hierarchy/lineage/validate)
    ExtractionPipeline   — Bronze → Silver → Gold medallion pipeline
                           Bronze: immutable JSONB per chunk
                           Silver: deduplicated canonical tables in PostgreSQL
                           Gold:   distinct vertex labels projected into AGE

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
Legacy (source kept, not wired into active pipeline)
---
    PgGraphStore    — entities/relationships in plain PostgreSQL tables (no Cypher)

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

from kg.pg_graph_store import PgGraphStore
from kg.age_graph_store import AgeGraphStore
from kg.cuad_kg_ingest import build_cuad_kg
from kg.legal_extractor import LegalEntityExtractor
from kg.extraction_pipeline import ExtractionPipeline
from kg.constants import (
    VALID_LABELS,
    VALID_REL_TYPES,
    ENTITY_TYPE_MAP,
    RELATIONSHIP_MAP,
    entity_type_for,
    relationship_type_for,
)
from kg.schemas import GraphType, get_schema
from kg.graph_router import GraphRouter
from kg.intent_parser import IntentParser, IntentMatch
from kg.query_builder import QUERY_CAPABILITIES
from kg.nl2cypher import NL2CypherConverter

from rag.config.settings import load_settings


def create_kg_store() -> AgeGraphStore | PgGraphStore:
    """
    Return the configured knowledge graph store.

    Reads ``KG_BACKEND`` from settings:
    - ``"age"``      (default) → AgeGraphStore  — Apache AGE Cypher graph (docker-compose)
    - ``"postgres"`` (legacy)  → PgGraphStore   — entity/relationship SQL tables (no Cypher)

    Switching backends requires only one line in .env::

        KG_BACKEND=postgres   # opt back into legacy SQL backend
    """
    settings = load_settings()
    if settings.kg_backend == "age":
        return AgeGraphStore()
    return PgGraphStore()  # "postgres" or any unknown value → safe default


__all__ = [
    "PgGraphStore",
    "AgeGraphStore",
    "build_cuad_kg",
    "LegalEntityExtractor",
    "ExtractionPipeline",
    "create_kg_store",
    "GraphType",
    "GraphRouter",
    "get_schema",
    "IntentParser",
    "IntentMatch",
    "QUERY_CAPABILITIES",
    "NL2CypherConverter",
    "VALID_LABELS",
    "VALID_REL_TYPES",
    "ENTITY_TYPE_MAP",
    "RELATIONSHIP_MAP",
    "entity_type_for",
    "relationship_type_for",
]

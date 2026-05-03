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
No Neo4j or Graphiti required.

Primary components:
    AgeGraphStore        — Apache AGE Cypher graph (active backend)
    build_cuad_kg()      — ingest cuad_eval.json annotations into AgeGraphStore
    LegalEntityExtractor — LLM-driven 5-pass entity/relationship extraction
    ExtractionPipeline   — Bronze → Silver → Gold medallion ingestion pipeline

Legacy / reference components (source kept, not wired into main pipeline):
    PgGraphStore    — entities/relationships in PostgreSQL SQL tables (legacy)

Ontology constants (single source of truth):
    VALID_LABELS, VALID_REL_TYPES, ENTITY_TYPE_MAP, RELATIONSHIP_MAP,
    entity_type_for(), relationship_type_for()  — from constants.py

Usage:
    from kg import create_kg_store

    store = create_kg_store()   # returns AgeGraphStore by default
    await store.initialize()
    context = await store.search_as_context("governing law Delaware")
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
    "VALID_LABELS",
    "VALID_REL_TYPES",
    "ENTITY_TYPE_MAP",
    "RELATIONSHIP_MAP",
    "entity_type_for",
    "relationship_type_for",
]

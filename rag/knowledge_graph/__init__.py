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

PostgreSQL-backed knowledge graph (entities + relationships) built from
CUAD legal contract annotations.  No Neo4j or Graphiti required.

Primary components:
    PgGraphStore    — entities/relationships in PostgreSQL (kg_entities, kg_relationships)
    CuadKgBuilder   — populates the graph from cuad_eval.json annotations

Legacy Graphiti/Neo4j components (kept for reference, not wired into main pipeline):
    GraphitiStore, graphiti_config, graphiti_agent, kg_agent

Usage:
    from rag.knowledge_graph import PgGraphStore, CuadKgBuilder

    store = PgGraphStore()
    await store.initialize()
    context = await store.search_as_context("governing law Delaware")
    await store.close()
"""

from rag.knowledge_graph.pg_graph_store import PgGraphStore
from rag.knowledge_graph.age_graph_store import AgeGraphStore
from rag.knowledge_graph.cuad_kg_builder import CuadKgBuilder


from rag.config.settings import load_settings


def create_kg_store() -> PgGraphStore | AgeGraphStore:
    """
    Return the configured knowledge graph store.

    Reads ``KG_BACKEND`` from settings:
    - ``"postgres"`` (default) → PgGraphStore  — entity/relationship tables in Neon
    - ``"age"``               → AgeGraphStore  — Apache AGE Cypher graph (docker-compose)

    Switching backends requires only one line in .env::

        KG_BACKEND=age
        AGE_DATABASE_URL=postgresql://age_user:age_pass@localhost:5433/legal_graph
    """
    settings = load_settings()
    if settings.kg_backend == "age":
        return AgeGraphStore()
    return PgGraphStore()


__all__ = [
    "PgGraphStore",
    "AgeGraphStore",
    "CuadKgBuilder",
    "create_kg_store",
]

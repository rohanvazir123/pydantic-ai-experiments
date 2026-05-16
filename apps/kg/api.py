"""
Knowledge Graph — FastAPI server.

Endpoints
---------
GET  /health           — AGE container connectivity
GET  /v1/stats         — Graph statistics (entity + relationship counts)
POST /v1/search        — Search entities by name substring
POST /v1/context       — Retrieve graph context for a query (LLM-ready)
POST /v1/related       — Get entities related to a UUID
POST /v1/contracts     — Find contracts that mention a named entity
POST /v1/cypher        — Execute a read-only Cypher MATCH query

Usage:
    uvicorn apps.kg.api:app --host 0.0.0.0 --port 8002 --reload

Example:
    curl http://localhost:8002/v1/stats
    curl -X POST http://localhost:8002/v1/search \\
      -H "Content-Type: application/json" \\
      -d '{"query": "Amazon", "entity_type": "Party"}'
"""

import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from kg.age_graph_store import AgeGraphStore
from kg.legal.graph_router import GraphRouter
from kg.legal.nl2cypher import NL2CypherConverter
from kg.legal.schemas import GraphType, get_schema

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Knowledge Graph API",
    description="Apache AGE knowledge graph — entity search, traversal, and Cypher",
    version="1.0.0",
)

# Module-level singletons; store initialized on first request
_store: AgeGraphStore | None = None
_router: GraphRouter = GraphRouter()
_converter: NL2CypherConverter = NL2CypherConverter()


async def _get_store() -> AgeGraphStore:
    global _store
    if _store is None:
        _store = AgeGraphStore()
        await _store.initialize()
    return _store


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., description="Entity name substring to search for")
    entity_type: str | None = Field(default=None, description="Optional label filter (e.g. Party)")
    limit: int = Field(default=20, ge=1, le=100)


class EntityResult(BaseModel):
    id: str
    name: str
    entity_type: str
    document_id: str | None = None


class ContextRequest(BaseModel):
    query: str = Field(..., description="Text query to match against entity names")
    limit: int = Field(default=15, ge=1, le=50)


class RelatedRequest(BaseModel):
    entity_id: str = Field(..., description="UUID of the entity to traverse from")
    relationship_type: str | None = Field(default=None, description="Optional edge type filter")
    limit: int = Field(default=20, ge=1, le=100)


class RelatedResult(BaseModel):
    id: str
    name: str
    entity_type: str
    relationship_type: str


class ContractsRequest(BaseModel):
    entity_name: str = Field(..., description="Entity name to search for")
    entity_type: str | None = None
    limit: int = Field(default=10, ge=1, le=50)


class ContractResult(BaseModel):
    title: str
    document_id: str


class CypherRequest(BaseModel):
    cypher: str = Field(..., description="Read-only Cypher MATCH query")


class NLQueryRequest(BaseModel):
    question: str = Field(..., description="Natural-language question about the knowledge graph")


class NLQueryResponse(BaseModel):
    question: str
    graph_types: list[str]
    cypher: str
    result: str


class StatsResponse(BaseModel):
    total_entities: int
    total_relationships: int
    entities_by_type: dict[str, int]
    relationships_by_type: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    age_connected: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    try:
        store = await _get_store()
        stats = await store.get_graph_stats()
        return HealthResponse(status="ok", age_connected=True)
    except Exception as exc:
        logger.warning("Health: AGE check failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=HealthResponse(status="unhealthy", age_connected=False).model_dump(),
        ) from exc


@app.get("/v1/stats", response_model=StatsResponse, tags=["graph"])
async def stats() -> StatsResponse:
    """Return vertex and edge counts broken down by type."""
    try:
        store = await _get_store()
        data = await store.get_graph_stats()
        return StatsResponse(**data)
    except Exception as exc:
        logger.exception("Stats endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/search", response_model=list[EntityResult], tags=["graph"])
async def search(request: SearchRequest) -> list[EntityResult]:
    """Case-insensitive substring search over entity names."""
    try:
        store = await _get_store()
        results = await store.search_entities(
            query=request.query,
            entity_type=request.entity_type,
            limit=request.limit,
        )
        return [
            EntityResult(
                id=r["id"],
                name=r["name"],
                entity_type=r["entity_type"],
                document_id=r.get("document_id"),
            )
            for r in results
        ]
    except Exception as exc:
        logger.exception("Search endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/context", tags=["graph"])
async def context(request: ContextRequest) -> dict[str, str]:
    """
    Retrieve entity relationships formatted as LLM-ready context.

    Returns a Markdown-formatted string listing matched entity relationships,
    suitable for inclusion in an LLM prompt.
    """
    try:
        store = await _get_store()
        ctx = await store.search_as_context(query=request.query, limit=request.limit)
        return {"context": ctx}
    except Exception as exc:
        logger.exception("Context endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/related", response_model=list[RelatedResult], tags=["graph"])
async def related(request: RelatedRequest) -> list[RelatedResult]:
    """Return entities connected to the given entity UUID (both directions)."""
    try:
        store = await _get_store()
        results = await store.get_related_entities(
            entity_id=request.entity_id,
            relationship_type=request.relationship_type,
            limit=request.limit,
        )
        return [
            RelatedResult(
                id=r["id"],
                name=r["name"],
                entity_type=r["entity_type"],
                relationship_type=r["relationship_type"],
            )
            for r in results
        ]
    except Exception as exc:
        logger.exception("Related endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/contracts", response_model=list[ContractResult], tags=["graph"])
async def contracts(request: ContractsRequest) -> list[ContractResult]:
    """Return contracts (document_id + title) that contain the named entity."""
    try:
        store = await _get_store()
        results = await store.find_contracts_by_entity(
            entity_name=request.entity_name,
            entity_type=request.entity_type,
            limit=request.limit,
        )
        return [ContractResult(title=r["title"], document_id=r["document_id"]) for r in results]
    except Exception as exc:
        logger.exception("Contracts endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/cypher", tags=["graph"])
async def cypher(request: CypherRequest) -> dict[str, str]:
    """
    Execute a read-only Cypher MATCH query and return results as a table string.

    Only MATCH/RETURN queries are permitted. CREATE/MERGE/SET/DELETE/DROP/DETACH
    are blocked by the store's guardrail.
    """
    try:
        store = await _get_store()
        result = await store.run_cypher_query(request.cypher)
        return {"result": result}
    except Exception as exc:
        logger.exception("Cypher endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/nl_query", response_model=NLQueryResponse, tags=["graph"])
async def nl_query(request: NLQueryRequest) -> NLQueryResponse:
    """
    Answer a natural-language question by routing to the right graph schema,
    generating Cypher, and executing it.

    Pipeline:
      1. GraphRouter classifies the question → relevant graph type(s).
      2. get_schema() returns the compact token-bounded schema for those types.
      3. NL2CypherConverter generates a Cypher MATCH query (temperature=0).
      4. AgeGraphStore executes the query and returns results.

    Returns the routed graph types, generated Cypher, and query results.
    """
    try:
        store     = await _get_store()

        graph_types = _router.route(request.question)
        schema      = get_schema(graph_types)
        generated   = await _converter.convert(request.question, schema)
        result      = await store.run_cypher_query(generated)

        return NLQueryResponse(
            question=request.question,
            graph_types=[gt.value for gt in graph_types],
            cypher=generated,
            result=result,
        )
    except Exception as exc:
        logger.exception("NL query endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.kg.api:app", host="0.0.0.0", port=8002, reload=True)

"""Search route — synchronous hybrid search (fast path, < 200ms)."""

import uuid

from fastapi import APIRouter, HTTPException, Request

from knowledge.api.middleware import get_request_id, get_tenant_id
from knowledge.api.schemas import APIResponse, SearchRequest, SearchResponse, SearchResultItem

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=APIResponse[SearchResponse])
async def search(body: SearchRequest, request: Request) -> APIResponse[SearchResponse]:
    """Synchronous hybrid search — skips the agent and judge.

    Returns ranked chunks with confidence scores and corpus citations.
    Does not call an LLM — suitable for search-only (free tier) use.
    """
    retriever  = getattr(request.app.state, "retriever", None)
    request_id = get_request_id() or str(uuid.uuid4())

    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")

    results = await retriever.retrieve(
        query=body.query,
        corpus_ids=body.corpus_ids,
        tenant_id=get_tenant_id() or "default",
        k=body.k,
        include_graph=body.include_graph,
    )

    items = [
        SearchResultItem(
            chunk_id=str(r.chunk_id),
            document_title=r.document_title,
            document_source=r.document_source,
            content=r.content,
            confidence=r.confidence,
            excerpt=r.content[:200],
        )
        for r in results
    ]

    return APIResponse(
        request_id=request_id,
        data=SearchResponse(results=items, query=body.query, k=body.k),
    )

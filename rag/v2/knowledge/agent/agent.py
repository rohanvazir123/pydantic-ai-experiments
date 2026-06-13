"""Pydantic AI RAG agent.

Architecture anchor: mirrors rag/agent/rag_agent.py exactly.
Key patterns carried forward:
  - PydanticAgent module-level singleton
  - RAGState(BaseModel) lazy-init with asyncio.Lock
  - @agent.tool async functions for each retrieval tool
  - contextvars.ContextVar for per-coroutine Langfuse trace
  - agent.run() for blocking; agent.run_stream() for SSE streaming
  - result.usage() for token counts (Pydantic AI built-in — no manual counting)

Structured output: GenerationResult instead of plain str.
Tools scope all retrieval to ctx.deps.corpus_ids + ctx.deps.tenant_id.
"""

import asyncio
import contextvars
import logging
from typing import Any

from pydantic import BaseModel, PrivateAttr
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from knowledge.agent.prompts import LOW_CONFIDENCE_NOTICE, MAIN_SYSTEM_PROMPT
from knowledge.config.settings import Settings, load_settings
from knowledge.ingestion.models import Citation, SearchResult

logger = logging.getLogger(__name__)

# Per-coroutine Langfuse trace reference — safe for concurrent requests
_trace_context: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "rag_trace", default=None
)


# ── Structured output models ──────────────────────────────────────────────────

class CitationCheck(BaseModel):
    is_trustworthy:  bool
    uncited_claims:  list[str] = []


class GenerationResult(BaseModel):
    answer:         str
    citations:      list[Citation] = []
    citation_check: CitationCheck


# ── RAGState — shared dependency injected into every tool ─────────────────────

class RAGState(BaseModel):
    """Lazy-initialised retriever + stores, shared across all tool calls in one run."""

    model_config = {"arbitrary_types_allowed": True}

    user_id:    str       = ""
    tenant_id:  str       = ""
    session_id: str       = ""
    corpus_ids: list[str] = []

    _retriever:    Any | None = PrivateAttr(default=None)
    _age_store:    Any | None = PrivateAttr(default=None)
    _initialized:  bool       = PrivateAttr(default=False)
    _init_lock:    asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def get_retriever(self) -> Any:
        async with self._init_lock:
            if not self._initialized:
                from knowledge.ingestion.embedder import Embedder
                from knowledge.retrieval.retriever import Retriever
                from knowledge.store.vector import PostgresHybridStore
                store = PostgresHybridStore()
                await store.initialize()
                self._retriever   = Retriever(vector_store=store, embedder=Embedder())
                self._initialized = True
        return self._retriever

    async def close(self) -> None:
        if self._retriever and hasattr(self._retriever, "_vs") and self._retriever._vs:
            await self._retriever._vs.close()
            self._initialized = False


# ── Agent factory ─────────────────────────────────────────────────────────────

def _get_llm_model(settings: Settings) -> OpenAIChatModel:
    provider = OpenAIProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
    return OpenAIChatModel(settings.llm_model, provider=provider)


def _build_agent(settings: Settings, low_confidence: bool = False) -> Any:
    system_prompt = MAIN_SYSTEM_PROMPT
    if low_confidence:
        system_prompt += LOW_CONFIDENCE_NOTICE

    model = _get_llm_model(settings)
    _ms: dict = {}
    if settings.llm_provider == "ollama":
        _ms = {"extra_body": {"num_ctx": settings.llm_num_ctx}}

    ag: Any = PydanticAgent(  # type: ignore[call-overload]
        model,
        system_prompt=system_prompt,
        output_type=GenerationResult,
        model_settings=_ms,
        deps_type=RAGState,
    )

    @ag.tool
    async def search_knowledge_base(
        ctx: RunContext[RAGState],
        query: str,
        match_count: int | None = 5,
        search_type: str | None = "hybrid",
    ) -> str:
        """Search the knowledge base for relevant information.

        Returns formatted context with [chunk_id] anchors for citation.
        """
        retriever = await ctx.deps.get_retriever()
        results: list[SearchResult] = await retriever.retrieve(
            query=query,
            corpus_ids=ctx.deps.corpus_ids,
            tenant_id=ctx.deps.tenant_id,
            k=match_count or 5,
        )
        if not results:
            return "No relevant information found."
        lines = []
        for r in results:
            lines.append(
                f"[chunk_id: {r.chunk_id}] {r.document_title} ({r.document_source})\n"
                f"{r.content[:500]}"
            )
        return "\n\n".join(lines)

    @ag.tool
    async def search_knowledge_graph(
        ctx: RunContext[RAGState],
        query: str,
        entity_type: str | None = None,
        limit: int | None = 15,
    ) -> str:
        """Search the knowledge graph for entities and relationships.

        Use when the question asks about parties, jurisdictions, or relationships.
        """
        retriever = await ctx.deps.get_retriever()
        if not retriever._graph:
            return "Knowledge graph is not available for this corpus."
        corpus_id = ctx.deps.corpus_ids[0] if ctx.deps.corpus_ids else ""
        results = await retriever._graph.query(
            query, corpus_id, ctx.deps.tenant_id, limit or 15
        )
        if not results:
            return "No relevant entities found in knowledge graph."
        lines = [
            f"- [{r.metadata.get('entity_type', 'Entity')}] {r.content}"
            for r in results
        ]
        return "## Knowledge Graph — Entities\n" + "\n".join(lines)

    return ag


# Module-level default agent (loaded once per process)
_settings = load_settings()
agent = _build_agent(_settings)


# ── Traced run helper ─────────────────────────────────────────────────────────

async def traced_agent_run(
    query: str,
    state: RAGState,
    message_history: list | None = None,
    low_confidence: bool = False,
) -> Any:
    """Run the RAG agent with optional Langfuse tracing.

    Uses result.usage() for token counts — no manual token counting.
    """
    settings = load_settings()

    # Select agent (rebuild if low_confidence flag changes system prompt)
    ag = _build_agent(settings, low_confidence=low_confidence) if low_confidence else agent

    # Langfuse tracing (no-op when disabled)
    langfuse_ctx = None
    if settings.langfuse_enabled:
        try:
            from langfuse import Langfuse
            lf = Langfuse()
            langfuse_ctx = lf.trace(  # type: ignore[attr-defined]
                name="rag_agent_run",
                input={"query": query},
                user_id=state.user_id or None,
                session_id=state.session_id or None,
            )
            _trace_context.set(langfuse_ctx)
        except Exception:
            pass

    try:
        kwargs: dict = {"deps": state}
        if message_history:
            kwargs["message_history"] = message_history

        result = await ag.run(query, **kwargs)

        # Token usage via Pydantic AI built-in (no manual counting)
        usage = result.usage()
        logger.debug(
            "Agent run: prompt_tokens=%d completion_tokens=%d",
            usage.request_tokens or 0,
            usage.response_tokens or 0,
        )

        if langfuse_ctx:
            langfuse_ctx.update(output={"answer": str(result.output.answer)[:500]})

        return result

    finally:
        if settings.langfuse_enabled:
            try:
                from langfuse import Langfuse
                Langfuse().flush()
            except Exception:
                pass
        _trace_context.set(None)

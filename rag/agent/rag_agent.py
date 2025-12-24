"""Main MongoDB RAG agent implementation."""

import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import RunContext as PydanticRunContext
from pydantic_ai.models.openai import OpenAIModel, OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from rag.agent.prompts import MAIN_SYSTEM_PROMPT
from rag.config.settings import load_settings
from rag.observability import get_langfuse, trace_tool_call
from rag.retrieval.retriever import Retriever
from rag.storage.vector_store.mongo import MongoHybridStore

logger = logging.getLogger(__name__)

# Global trace reference for tool calls (set by traced_agent_run)
_current_trace = None


def get_llm_model(model_choice: str | None = None) -> OpenAIChatModel:
    """
    Get LLM model configuration based on environment variables.
    Supports any OpenAI-compatible API provider.

    Args:
        model_choice: Optional override for model choice

    Returns:
        Configured OpenAI-compatible model
    """
    settings = load_settings()

    llm_choice = model_choice or settings.llm_model
    base_url = settings.llm_base_url
    api_key = settings.llm_api_key

    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIChatModel(llm_choice, provider=provider)


def get_model_info() -> dict:
    """
    Get information about current model configuration.

    Returns:
        Dictionary with model configuration info
    """
    settings = load_settings()

    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_base_url": settings.llm_base_url,
        "embedding_model": settings.embedding_model,
    }


class RAGState(BaseModel):
    """
    Shared state for the RAG agent.

    Holds a store and retriever that are lazily initialized on first use.
    This avoids event loop issues when created in one loop (e.g., Streamlit startup)
    but used in another (agent execution).
    """

    model_config = {"arbitrary_types_allowed": True}

    # Internal storage - lazily initialized
    _store: MongoHybridStore | None = None
    _retriever: Retriever | None = None
    _initialized: bool = False

    async def get_retriever(self) -> Retriever:
        """Get or create the retriever (lazy initialization in current event loop)."""
        if not self._initialized:
            self._store = MongoHybridStore()
            await self._store.initialize()
            self._retriever = Retriever(store=self._store)
            self._initialized = True
            logger.info("[PROFILE] Lazy-initialized store/retriever in current event loop")
        return self._retriever

    async def close(self) -> None:
        """Clean up resources."""
        if self._store:
            await self._store.close()
            self._initialized = False


# Create the RAG agent
agent = PydanticAgent(get_llm_model(), system_prompt=MAIN_SYSTEM_PROMPT)


@agent.tool
async def search_knowledge_base(
    ctx: PydanticRunContext,
    query: str,
    match_count: int | None = 5,
    search_type: str | None = "hybrid",
) -> str:
    """
    Search the knowledge base for relevant information.

    Args:
        ctx: Agent runtime context (NOTE: ctx.deps is available but not currently used)
        query: Search query text
        match_count: Number of results to return (default: 5)
        search_type: Type of search - "semantic", "text", or "hybrid" (default)

    Returns:
        String containing the retrieved information formatted for the LLM
    """
    global _current_trace
    start_time = time.time()

    # Check if we have RAGState from deps for shared/lazy-initialized retriever
    deps = ctx.deps

    # deps could be RAGState directly or wrapped in StateDeps
    state = deps if isinstance(deps, RAGState) else getattr(deps, 'state', None)

    # Track if we created a local store (need to close it later)
    local_store = None

    try:
        if state is not None and isinstance(state, RAGState):
            # Use lazy-initialized retriever from RAGState (same event loop, reused)
            retriever = await state.get_retriever()
            logger.info("[PROFILE] Using shared retriever from RAGState")
        else:
            # Fall back to creating new instances (slower, but works without deps)
            logger.info("[PROFILE] Creating NEW store/retriever (no RAGState)")
            local_store = MongoHybridStore()
            retriever = Retriever(store=local_store)

        # Perform search
        actual_search_type = search_type or "hybrid"
        actual_match_count = match_count or 5

        result = await retriever.retrieve_as_context(
            query=query, match_count=actual_match_count, search_type=actual_search_type
        )

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Trace the tool call if Langfuse is enabled
        if _current_trace is not None:
            trace_tool_call(
                trace=_current_trace,
                tool_name="search_knowledge_base",
                tool_input={
                    "query": query,
                    "match_count": actual_match_count,
                    "search_type": actual_search_type,
                },
                tool_output=result[:500] + "..." if len(result) > 500 else result,
                duration_ms=duration_ms,
            )

        # Only close if we created a local store (don't close shared store)
        if local_store is not None:
            await local_store.close()

        return result

    except Exception as e:
        logger.exception(f"Error searching knowledge base: {e}")
        return f"Error searching knowledge base: {str(e)}"


class RunContext:
    """Minimal RunContext placeholder for editor/type-checker."""

    def __init__(self, **kwargs):
        pass


class Agent:
    def __init__(
        self, model: Any, system_prompt: str | None = None, deps_type: Any | None = None
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.deps_type = deps_type

    def tool(self, func: Callable):
        """Decorator placeholder - returns the function unchanged."""
        return func


__all__ = ["Agent", "RunContext", "traced_agent_run"]


async def traced_agent_run(
    query: str,
    user_id: str | None = None,
    session_id: str | None = None,
    message_history: list | None = None,
) -> Any:
    """
    Run the RAG agent with Langfuse tracing enabled.

    This is a convenience wrapper that sets up tracing for the agent run,
    including all tool calls made during execution.

    Args:
        query: The user's query
        user_id: Optional user identifier for grouping traces
        session_id: Optional session identifier for conversation tracking
        message_history: Optional list of previous messages for context

    Returns:
        The agent's response

    Example:
        result = await traced_agent_run(
            "What is RAG?",
            user_id="user123",
            session_id="session456"
        )
        print(result.output)
    """
    global _current_trace

    langfuse = get_langfuse()

    if langfuse is not None:
        # Create a trace for this run
        _current_trace = langfuse.trace(
            name="rag_agent_run",
            input={"query": query},
            user_id=user_id,
            session_id=session_id,
            metadata={"has_history": message_history is not None},
        )

    try:
        # Run the agent
        if message_history:
            result = await agent.run(query, message_history=message_history)
        else:
            result = await agent.run(query)

        # Update trace with output
        if _current_trace is not None:
            _current_trace.update(
                output={"response": str(result.output)[:1000]},
            )

        return result

    except Exception as e:
        if _current_trace is not None:
            _current_trace.update(
                output={"error": str(e)},
                level="ERROR",
            )
        raise

    finally:
        # Flush trace
        if langfuse is not None:
            langfuse.flush()
        _current_trace = None

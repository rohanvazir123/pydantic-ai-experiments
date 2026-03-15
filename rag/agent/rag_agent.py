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
Main RAG agent (PostgreSQL/pgvector) implementation.

Module: rag.agent.rag_agent
===========================

This module provides the Pydantic AI-based RAG agent that uses the knowledge
base to answer questions. Supports Langfuse tracing and Mem0 personalization.

Classes
-------
RAGState(BaseModel)
    Shared state for lazy-initialized store/retriever/mem0.

    Attributes:
        user_id: str | None     - User identifier for Mem0 personalization

    Methods:
        async get_retriever() -> Retriever
            Get or create retriever (lazy init in current event loop).

        get_mem0() -> Mem0Store
            Get or create Mem0 store for user memories.

        async close() -> None
            Clean up resources.

Agent (placeholder)
    Minimal placeholder class for type hints.

RunContext (placeholder)
    Minimal placeholder class for type hints.

Functions
---------
get_llm_model(model_choice: str | None = None) -> OpenAIChatModel
    Get LLM model configuration from environment settings.

get_model_info() -> dict
    Get information about current model configuration.

search_knowledge_base(ctx, query, match_count, search_type) -> str
    Agent tool: Search knowledge base and return formatted results.
    Combines RAG retrieval with Mem0 user context when available.

traced_agent_run(
    query: str,
    user_id: str | None = None,
    session_id: str | None = None,
    message_history: list | None = None
) -> Any
    Run RAG agent with Langfuse tracing enabled.

Module Attributes
-----------------
agent: PydanticAgent
    Pre-configured Pydantic AI agent with search tool.

_trace_context: contextvars.ContextVar
    Per-coroutine trace reference for tool calls (set by traced_agent_run).
    Safe for concurrent requests — each coroutine has its own value.

Agent Tools
-----------
@agent.tool search_knowledge_base(ctx, query, match_count, search_type)
    Search the knowledge base using hybrid/semantic/text search.
    If user_id is set in RAGState and Mem0 is enabled, prepends
    relevant user context to the search results.

Usage
-----
    from rag.agent.rag_agent import agent, traced_agent_run, RAGState

    # Simple agent run
    result = await agent.run("What does NeuralFlow AI do?")
    print(result.output)

    # With tracing (Langfuse)
    result = await traced_agent_run(
        query="What is the PTO policy?",
        user_id="user123",
        session_id="session456"
    )

    # With shared state and Mem0 personalization
    state = RAGState(user_id="john_doe")
    result = await agent.run("Query here", deps=state)
    await state.close()

    # Add user memory (for personalization)
    from rag.memory import create_mem0_store
    mem0 = create_mem0_store()
    mem0.add("User prefers concise answers", user_id="john_doe")
"""

import asyncio
import contextvars
import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from pydantic import PrivateAttr
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import RunContext as PydanticRunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from rag.agent.prompts import MAIN_SYSTEM_PROMPT
from rag.config.settings import load_settings
from rag.memory.mem0_store import Mem0Store
from rag.observability import get_langfuse, trace_tool_call
from rag.retrieval.retriever import Retriever
from rag.storage.vector_store.postgres import PostgresHybridStore

logger = logging.getLogger(__name__)

# Per-coroutine trace reference — safe for concurrent requests
_trace_context: contextvars.ContextVar = contextvars.ContextVar("rag_trace", default=None)


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

    Holds a store, retriever, and mem0 that are lazily initialized on first use.
    This avoids event loop issues when created in one loop (e.g., Streamlit startup)
    but used in another (agent execution).

    Attributes:
        user_id: Optional user identifier for Mem0 personalization
    """

    model_config = {"arbitrary_types_allowed": True}

    # User identifier for Mem0 personalization
    user_id: str | None = None

    # Internal storage - lazily initialized
    _store: PostgresHybridStore | None = PrivateAttr(default=None)
    _retriever: Retriever | None = PrivateAttr(default=None)
    _mem0: Mem0Store | None = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)
    _init_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def get_retriever(self) -> Retriever:
        """Get or create the retriever (lazy initialization in current event loop)."""
        async with self._init_lock:
            if not self._initialized:
                self._store = PostgresHybridStore()
                await self._store.initialize()
                self._retriever = Retriever(store=self._store)
                self._mem0 = Mem0Store()
                self._initialized = True
                logger.info(
                    "[PROFILE] Lazy-initialized store/retriever/mem0 in current event loop"
                )
        return self._retriever

    def get_mem0(self) -> Mem0Store:
        """Get or create the Mem0 store."""
        if self._mem0 is None:
            self._mem0 = Mem0Store()
        return self._mem0

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

    Combines RAG retrieval with Mem0 user context (if enabled and user_id provided).

    Args:
        ctx: Agent runtime context (NOTE: ctx.deps is available but not currently used)
        query: Search query text
        match_count: Number of results to return (default: 5)
        search_type: Type of search - "semantic", "text", or "hybrid" (default)

    Returns:
        String containing the retrieved information formatted for the LLM,
        optionally prefixed with user context from Mem0
    """
    start_time = time.time()

    # Check if we have RAGState from deps for shared/lazy-initialized retriever
    deps = ctx.deps

    # deps could be RAGState directly or wrapped in StateDeps
    state = deps if isinstance(deps, RAGState) else getattr(deps, "state", None)

    # Track if we created a local store (need to close it later)
    local_store = None
    mem0_store = None

    try:
        if state is not None and isinstance(state, RAGState):
            # Use lazy-initialized retriever from RAGState (same event loop, reused)
            retriever = await state.get_retriever()
            mem0_store = state.get_mem0()
            user_id = state.user_id
            logger.info("[PROFILE] Using shared retriever from RAGState")
        else:
            # Fall back to creating new instances (slower, but works without deps)
            logger.info("[PROFILE] Creating NEW store/retriever (no RAGState)")
            local_store = PostgresHybridStore()
            retriever = Retriever(store=local_store)
            mem0_store = Mem0Store()
            user_id = None

        # Perform search
        actual_search_type = search_type or "hybrid"
        actual_match_count = match_count or 5

        # Get RAG results
        rag_result = await retriever.retrieve_as_context(
            query=query, match_count=actual_match_count, search_type=actual_search_type
        )

        # Get Mem0 user context if available
        user_context = ""
        if mem0_store and user_id and mem0_store.is_enabled():
            user_context = mem0_store.get_context_string(
                query=query, user_id=user_id, limit=3
            )
            if user_context:
                logger.info(f"[PROFILE] Added Mem0 context for user: {user_id}")

        # Combine contexts
        if user_context:
            result = f"{user_context}\n\n{rag_result}"
        else:
            result = rag_result

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Trace the tool call if Langfuse is enabled
        if _trace_context.get() is not None:
            trace_tool_call(
                trace=_trace_context.get(),
                tool_name="search_knowledge_base",
                tool_input={
                    "query": query,
                    "match_count": actual_match_count,
                    "search_type": actual_search_type,
                    "user_id": user_id,
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


__all__ = [
    "Agent",
    "RunContext",
    "traced_agent_run",
    "agent",
    "RAGState",
    "get_model_info",
]


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
    langfuse = get_langfuse()

    if langfuse is not None:
        # Set trace in context var — isolated per coroutine, safe for concurrent calls
        _trace_context.set(
            langfuse.trace(
                name="rag_agent_run",
                input={"query": query},
                user_id=user_id,
                session_id=session_id,
                metadata={"has_history": message_history is not None},
            )
        )

    state = RAGState(user_id=user_id)
    try:
        # Run the agent with shared state (enables lazy-initialized store reuse)
        if message_history:
            result = await agent.run(query, message_history=message_history, deps=state)
        else:
            result = await agent.run(query, deps=state)

        # Update trace with output
        if _trace_context.get() is not None:
            _trace_context.get().update(
                output={"response": str(result.output)[:1000]},
            )

        return result

    except Exception as e:
        if _trace_context.get() is not None:
            _trace_context.get().update(
                output={"error": str(e)},
                level="ERROR",
            )
        raise

    finally:
        await state.close()
        # Flush trace and reset context var
        if langfuse is not None:
            langfuse.flush()
        _trace_context.set(None)


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _logger = logging.getLogger(__name__)

    async def main():
        _logger.info("=" * 60)
        _logger.info("RAG Agent Module Test")
        _logger.info("=" * 60)

        # Display model info
        _logger.info("--- Model Configuration ---")
        info = get_model_info()
        for key, value in info.items():
            _logger.info(f"  {key}: {value}")

        # Test queries
        test_queries = [
            "What does NeuralFlow AI do? Keep your answer brief.",
            "How many employees does the company have?",
            "What is the learning budget for employees?",
        ]

        # Run agent for each query
        for query in test_queries:
            _logger.info("--- Query ---")
            _logger.info(f"  {query}")
            _logger.info("-" * 40)

            try:
                start = time.time()
                result = await agent.run(query)
                elapsed = (time.time() - start) * 1000

                # Display response
                response = result.output
                _logger.info(f"  Response ({elapsed:.0f}ms):")
                # Word wrap at 60 chars
                words = response.split()
                line = "    "
                for word in words:
                    if len(line) + len(word) > 64:
                        _logger.info(line)
                        line = "    "
                    line += word + " "
                if line.strip():
                    _logger.info(line)

            except Exception as e:
                _logger.error(f"  Error: {e}")

        # Test with shared state (RAGState)
        _logger.info("--- RAGState Test ---")
        state = RAGState()
        try:
            result = await agent.run(
                "What is the PTO policy? One sentence.",
                deps=state,
            )
            _logger.info(f"  With RAGState: {result.output[:100]}...")
        except Exception as e:
            _logger.error(f"  Error with RAGState: {e}")
        finally:
            await state.close()

        _logger.info("=" * 60)
        _logger.info("RAG agent test completed!")
        _logger.info("=" * 60)

    asyncio.run(main())

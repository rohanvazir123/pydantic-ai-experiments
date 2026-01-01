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
Langfuse integration for RAG agent observability.

This module provides tracing and observability for the RAG agent using Langfuse.
It instruments LLM calls, tool executions, and retrieval operations.

Usage:
    from rag.observability.langfuse_integration import (
        get_langfuse,
        trace_agent_run,
        trace_retrieval,
    )

    # Initialize Langfuse (call once at startup)
    langfuse = get_langfuse()

    # Trace an agent run
    with trace_agent_run("user_query") as trace:
        result = await agent.run(query)
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from functools import wraps
from typing import Any

from rag.config.settings import load_settings

logger = logging.getLogger(__name__)

# Global Langfuse instance
_langfuse_instance = None


def get_langfuse():
    """
    Get or create the global Langfuse instance.

    Returns:
        Langfuse instance if enabled and configured, None otherwise.
    """
    global _langfuse_instance

    if _langfuse_instance is not None:
        return _langfuse_instance

    settings = load_settings()

    if not settings.langfuse_enabled:
        logger.debug("Langfuse is disabled")
        return None

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning(
            "Langfuse is enabled but missing keys. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env"
        )
        return None

    try:
        from langfuse import Langfuse

        _langfuse_instance = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info(f"Langfuse initialized with host: {settings.langfuse_host}")
        return _langfuse_instance

    except ImportError:
        logger.warning("Langfuse package not installed. Run: pip install langfuse")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")
        return None


def shutdown_langfuse() -> None:
    """Flush and shutdown Langfuse gracefully."""
    global _langfuse_instance
    if _langfuse_instance is not None:
        try:
            _langfuse_instance.flush()
            _langfuse_instance.shutdown()
            logger.info("Langfuse shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down Langfuse: {e}")
        finally:
            _langfuse_instance = None


@contextmanager
def trace_agent_run(
    query: str,
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Generator[Any]:
    """
    Context manager for tracing an agent run.

    Args:
        query: The user's query
        user_id: Optional user identifier for grouping traces
        session_id: Optional session identifier for conversation tracking
        metadata: Additional metadata to attach to the trace

    Yields:
        Langfuse trace object (or None if Langfuse is disabled)

    Example:
        with trace_agent_run("What is RAG?", user_id="user123") as trace:
            result = await agent.run(query)
    """
    langfuse = get_langfuse()

    if langfuse is None:
        yield None
        return

    trace = langfuse.trace(
        name="rag_agent_run",
        input={"query": query},
        user_id=user_id,
        session_id=session_id,
        metadata=metadata or {},
    )

    try:
        yield trace
    except Exception as e:
        trace.update(
            output={"error": str(e)},
            level="ERROR",
        )
        raise
    finally:
        # Trace will be automatically closed
        pass


def trace_retrieval(
    trace: Any,
    query: str,
    search_type: str,
    results_count: int,
    results: list[dict[str, Any]] | None = None,
) -> Any:
    """
    Add a retrieval span to an existing trace.

    Args:
        trace: Parent Langfuse trace object
        query: Search query
        search_type: Type of search (semantic, text, hybrid)
        results_count: Number of results returned
        results: Optional list of result summaries

    Returns:
        Langfuse span object (or None if tracing is disabled)
    """
    if trace is None:
        return None

    span = trace.span(
        name="retrieval",
        input={
            "query": query,
            "search_type": search_type,
        },
        metadata={"results_count": results_count},
    )

    if results:
        # Summarize results to avoid storing too much data
        result_summaries = [
            {
                "title": r.get("document_title", "Unknown"),
                "score": r.get("similarity", 0),
                "content_preview": r.get("content", "")[:200] + "...",
            }
            for r in results[:5]  # Limit to first 5
        ]
        span.update(output={"results": result_summaries})

    span.end()
    return span


def trace_llm_call(
    trace: Any,
    model: str,
    prompt: str,
    response: str,
    tokens_input: int | None = None,
    tokens_output: int | None = None,
) -> Any:
    """
    Add an LLM generation span to an existing trace.

    Args:
        trace: Parent Langfuse trace object
        model: Model name/identifier
        prompt: Input prompt
        response: Model response
        tokens_input: Optional input token count
        tokens_output: Optional output token count

    Returns:
        Langfuse generation object (or None if tracing is disabled)
    """
    if trace is None:
        return None

    generation = trace.generation(
        name="llm_generation",
        model=model,
        input=prompt,
        output=response,
        usage={
            "input": tokens_input,
            "output": tokens_output,
        }
        if tokens_input or tokens_output
        else None,
    )

    generation.end()
    return generation


def trace_tool_call(
    trace: Any,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_output: str | dict[str, Any],
    duration_ms: float | None = None,
) -> Any:
    """
    Add a tool call span to an existing trace.

    Args:
        trace: Parent Langfuse trace object
        tool_name: Name of the tool being called
        tool_input: Input arguments to the tool
        tool_output: Output from the tool
        duration_ms: Optional execution duration in milliseconds

    Returns:
        Langfuse span object (or None if tracing is disabled)
    """
    if trace is None:
        return None

    span = trace.span(
        name=f"tool_{tool_name}",
        input=tool_input,
        output={"result": tool_output} if isinstance(tool_output, str) else tool_output,
        metadata={"duration_ms": duration_ms} if duration_ms else None,
    )

    span.end()
    return span


def observe(name: str | None = None):
    """
    Decorator for observing function execution with Langfuse.

    Args:
        name: Optional name for the span (defaults to function name)

    Example:
        @observe("search_documents")
        async def search(query: str) -> list:
            ...
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            langfuse = get_langfuse()
            if langfuse is None:
                return await func(*args, **kwargs)

            span_name = name or func.__name__

            # Create a span for this function
            trace = langfuse.trace(name=span_name)
            span = trace.span(
                name=span_name,
                input={"args": str(args)[:500], "kwargs": str(kwargs)[:500]},
            )

            try:
                result = await func(*args, **kwargs)
                span.update(output={"result": str(result)[:1000]})
                return result
            except Exception as e:
                span.update(output={"error": str(e)}, level="ERROR")
                raise
            finally:
                span.end()

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            langfuse = get_langfuse()
            if langfuse is None:
                return func(*args, **kwargs)

            span_name = name or func.__name__

            trace = langfuse.trace(name=span_name)
            span = trace.span(
                name=span_name,
                input={"args": str(args)[:500], "kwargs": str(kwargs)[:500]},
            )

            try:
                result = func(*args, **kwargs)
                span.update(output={"result": str(result)[:1000]})
                return result
            except Exception as e:
                span.update(output={"error": str(e)}, level="ERROR")
                raise
            finally:
                span.end()

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Convenience function to check if Langfuse is enabled
def is_langfuse_enabled() -> bool:
    """Check if Langfuse tracing is enabled and configured."""
    settings = load_settings()
    return (
        settings.langfuse_enabled
        and settings.langfuse_public_key is not None
        and settings.langfuse_secret_key is not None
    )

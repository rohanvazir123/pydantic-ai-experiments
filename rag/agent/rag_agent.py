"""Main MongoDB RAG agent implementation."""

import logging
from typing import Any, Callable

from pydantic import BaseModel
from pydantic_ai import Agent as PydanticAgent, RunContext as PydanticRunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from rag.agent.prompts import MAIN_SYSTEM_PROMPT
from rag.config.settings import load_settings
from rag.retrieval.retriever import Retriever
from rag.storage.vector_store.mongo import MongoHybridStore

logger = logging.getLogger(__name__)


def get_llm_model(model_choice: str | None = None) -> OpenAIModel:
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
    return OpenAIModel(llm_choice, provider=provider)


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
    """Minimal shared state for the RAG agent."""

    pass


# Create the RAG agent
rag_agent = PydanticAgent(get_llm_model(), system_prompt=MAIN_SYSTEM_PROMPT)


@rag_agent.tool
async def search_knowledge_base(
    ctx: PydanticRunContext,
    query: str,
    match_count: int | None = 5,
    search_type: str | None = "hybrid",
) -> str:
    """
    Search the knowledge base for relevant information.

    Args:
        ctx: Agent runtime context
        query: Search query text
        match_count: Number of results to return (default: 5)
        search_type: Type of search - "semantic", "text", or "hybrid" (default)

    Returns:
        String containing the retrieved information formatted for the LLM
    """
    try:
        # Initialize components
        store = MongoHybridStore()
        retriever = Retriever(store=store)

        # Perform search
        result = await retriever.retrieve_as_context(
            query=query, match_count=match_count, search_type=search_type or "hybrid"
        )

        # Clean up
        await store.close()

        return result

    except Exception as e:
        logger.exception(f"Error searching knowledge base: {e}")
        return f"Error searching knowledge base: {str(e)}"


# Export for convenience
# The line `agent = rag_agent` is assigning the RAG agent instance `rag_agent` to a variable named
# `agent`. This allows the RAG agent to be accessed and used conveniently through the `agent` variable
# in other parts of the code.
agent = rag_agent


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


__all__ = ["Agent", "RunContext"]

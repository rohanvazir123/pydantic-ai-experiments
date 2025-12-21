"""
Knowledge Graph module for RAG using Graphiti.

This module provides integration with Graphiti for building and querying
knowledge graphs from documents, enabling entity-relationship based retrieval.

Components:
    - GraphitiStore: High-level wrapper for graph operations
    - graphiti_config: Configuration and client creation
    - graphiti_agent: LangGraph agent with Graphiti integration

Usage:
    from rag.knowledge_graph import GraphitiStore, create_graphiti_client

    # Using the store wrapper
    store = GraphitiStore()
    await store.initialize()
    await store.add_episode("content", name="Document")
    results = await store.search("query")

    # Using the raw client
    client = create_graphiti_client()
    await client.add_episode(...)
"""

from rag.knowledge_graph.graphiti_config import (
    GraphitiConfig,
    create_graphiti_client,
    get_graphiti_config,
    initialize_graphiti,
)
from rag.knowledge_graph.graphiti_store import GraphitiStore

__all__ = [
    # Store
    "GraphitiStore",
    # Config
    "GraphitiConfig",
    "create_graphiti_client",
    "get_graphiti_config",
    "initialize_graphiti",
]

# Agent exports (Pydantic AI based)
from rag.knowledge_graph.graphiti_agent import (  # noqa: F401
    GraphitiAgentDeps,
    create_graphiti_session,
    graphiti_agent,
    run_graphiti_agent,
)

__all__.extend(
    [
        "graphiti_agent",
        "GraphitiAgentDeps",
        "run_graphiti_agent",
        "create_graphiti_session",
    ]
)

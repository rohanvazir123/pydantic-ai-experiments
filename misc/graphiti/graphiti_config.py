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
Graphiti configuration for Neo4j and Ollama integration.

This module provides configuration and client creation for Graphiti,
supporting local Ollama LLMs and embeddings.

Usage:
    from rag.knowledge_graph.graphiti_config import create_graphiti_client

    client = create_graphiti_client()
    await client.add_episode(...)
"""

import logging
from dataclasses import dataclass
from typing import Any

from rag.config.settings import load_settings

logger = logging.getLogger(__name__)


@dataclass
class GraphitiConfig:
    """Configuration for Graphiti client."""

    # Neo4j connection
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # LLM configuration
    llm_model: str
    llm_small_model: str
    llm_base_url: str
    llm_api_key: str

    # Embedding configuration
    embedding_model: str
    embedding_dim: int
    embedding_base_url: str
    embedding_api_key: str


def get_graphiti_config() -> GraphitiConfig:
    """
    Get Graphiti configuration from environment settings.

    Returns:
        GraphitiConfig with all necessary settings for Graphiti client.

    Raises:
        ValueError: If required Neo4j settings are missing.
    """
    settings = load_settings()

    # Neo4j settings (with defaults for local development)
    import os

    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "")

    if not neo4j_password:
        logger.warning(
            "NEO4J_PASSWORD not set. Set it in your .env file for production use."
        )

    return GraphitiConfig(
        # Neo4j
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        # LLM (use existing settings)
        llm_model=settings.llm_model,
        llm_small_model=os.environ.get("LLM_SMALL_MODEL", "llama3.2:3b"),
        llm_base_url=settings.llm_base_url or "http://localhost:11434/v1",
        llm_api_key=settings.llm_api_key,
        # Embeddings (use existing settings)
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dimension,
        embedding_base_url=settings.embedding_base_url or "http://localhost:11434/v1",
        embedding_api_key=settings.embedding_api_key,
    )


def create_graphiti_client(config: GraphitiConfig | None = None) -> Any:
    """
    Create a configured Graphiti client for Neo4j with Ollama.

    This function creates a Graphiti client configured to use:
    - Neo4j as the graph database
    - Ollama for LLM inference (OpenAI-compatible API)
    - Ollama for embeddings (nomic-embed-text recommended)
    - Optional cross-encoder reranking

    Args:
        config: Optional GraphitiConfig. If None, loads from environment.

    Returns:
        Configured Graphiti client instance.

    Raises:
        ImportError: If graphiti-core is not installed.

    Example:
        client = create_graphiti_client()

        await client.add_episode(
            name="Document Title",
            episode_body="Document content...",
            source=EpisodeType.text,
            reference_time=datetime.now(timezone.utc),
        )

        results = await client.search("query text")
    """
    try:
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import (
            OpenAIRerankerClient,
        )
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    except ImportError as e:
        raise ImportError(
            "graphiti-core is required for knowledge graph support. "
            "Install it with: pip install graphiti-core"
        ) from e

    if config is None:
        config = get_graphiti_config()

    # Configure LLM client (OpenAI-compatible for Ollama)
    llm_config = LLMConfig(
        api_key=config.llm_api_key,
        model=config.llm_model,
        small_model=config.llm_small_model,
        base_url=config.llm_base_url,
    )
    llm_client = OpenAIGenericClient(config=llm_config)

    # Configure embedder
    embedder_config = OpenAIEmbedderConfig(
        api_key=config.embedding_api_key,
        embedding_model=config.embedding_model,
        embedding_dim=config.embedding_dim,
        base_url=config.embedding_base_url,
    )
    embedder = OpenAIEmbedder(config=embedder_config)

    # Configure cross-encoder reranker (optional, uses LLM)
    reranker = OpenAIRerankerClient(client=llm_client, config=llm_config)

    # Create Graphiti client
    client = Graphiti(
        config.neo4j_uri,
        config.neo4j_user,
        config.neo4j_password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=reranker,
    )

    logger.info(
        f"Graphiti client created with Neo4j at {config.neo4j_uri}, "
        f"LLM: {config.llm_model}, Embeddings: {config.embedding_model}"
    )

    return client


async def initialize_graphiti(client: Any, clear_existing: bool = False) -> None:
    """
    Initialize Graphiti graph database with required indices and constraints.

    Args:
        client: Graphiti client instance.
        clear_existing: If True, clears all existing data before initialization.
                       WARNING: This will delete all data in the graph!

    Example:
        client = create_graphiti_client()
        await initialize_graphiti(client, clear_existing=False)
    """
    if clear_existing:
        from graphiti_core.utils.maintenance.graph_data_operations import clear_data

        logger.warning("Clearing all existing data from Graphiti graph...")
        await clear_data(client.driver)

    await client.build_indices_and_constraints()
    logger.info("Graphiti indices and constraints initialized")

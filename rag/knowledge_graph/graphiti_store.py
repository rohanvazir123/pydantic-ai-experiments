"""
Graphiti Store wrapper for knowledge graph operations.

This module provides a high-level interface for interacting with the Graphiti
knowledge graph, supporting document ingestion, entity extraction, and
graph-based retrieval.

Usage:
    from rag.knowledge_graph import GraphitiStore

    store = GraphitiStore()
    await store.initialize()

    # Add documents
    await store.add_episode("doc content", name="Doc Title")

    # Search
    results = await store.search("query")
"""

import logging
from datetime import UTC, datetime
from typing import Any

from rag.knowledge_graph.graphiti_config import (
    GraphitiConfig,
    create_graphiti_client,
    get_graphiti_config,
)

logger = logging.getLogger(__name__)


class GraphitiStore:
    """
    High-level wrapper for Graphiti knowledge graph operations.

    This class provides a simplified interface for:
    - Adding documents/episodes to the graph
    - Searching for entities and relationships
    - Managing graph lifecycle

    Attributes:
        client: The underlying Graphiti client instance.
        config: Configuration used to create the client.
    """

    def __init__(self, config: GraphitiConfig | None = None):
        """
        Initialize the GraphitiStore.

        Args:
            config: Optional configuration. If None, loads from environment.
        """
        self.config = config or get_graphiti_config()
        self._client: Any | None = None
        self._initialized = False

    @property
    def client(self) -> Any:
        """Get the Graphiti client, creating it if necessary."""
        if self._client is None:
            self._client = create_graphiti_client(self.config)
        return self._client

    async def initialize(self, clear_existing: bool = False) -> None:
        """
        Initialize the graph database with required indices.

        Args:
            clear_existing: If True, clears all existing data.
                           WARNING: This deletes all graph data!
        """
        if self._initialized:
            return

        if clear_existing:
            from graphiti_core.utils.maintenance.graph_data_operations import (
                clear_data,
            )

            logger.warning("Clearing all existing Graphiti data...")
            await clear_data(self.client.driver)

        await self.client.build_indices_and_constraints()
        self._initialized = True
        logger.info("GraphitiStore initialized")

    async def close(self) -> None:
        """Close the Graphiti client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._initialized = False
            logger.info("GraphitiStore closed")

    async def add_episode(
        self,
        content: str,
        name: str,
        source_description: str = "document",
        episode_type: str = "text",
        reference_time: datetime | None = None,
    ) -> None:
        """
        Add an episode (document/text) to the knowledge graph.

        Graphiti will automatically extract entities and relationships
        from the content and add them to the graph.

        Args:
            content: The text content to add.
            name: A name/title for this episode.
            source_description: Description of the source (e.g., "PDF document").
            episode_type: Type of episode: "text", "json", or "message".
            reference_time: When this content was created/valid.

        Example:
            await store.add_episode(
                content="John works at Acme Corp as a software engineer.",
                name="Employee Record",
                source_description="HR Database",
            )
        """
        from graphiti_core.nodes import EpisodeType

        type_map = {
            "text": EpisodeType.text,
            "json": EpisodeType.json,
            "message": EpisodeType.message,
        }

        await self.client.add_episode(
            name=name,
            episode_body=content,
            source=type_map.get(episode_type, EpisodeType.text),
            source_description=source_description,
            reference_time=reference_time or datetime.now(UTC),
        )
        logger.debug(f"Added episode to graph: {name}")

    async def search(
        self,
        query: str,
        num_results: int = 10,
        center_node_uuid: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search the knowledge graph for relevant facts/relationships.

        Args:
            query: Natural language search query.
            num_results: Maximum number of results to return.
            center_node_uuid: Optional UUID of a node to center the search on.
                             Results closer to this node are ranked higher.

        Returns:
            List of edge results, each containing:
            - uuid: Unique identifier of the edge
            - fact: The fact/relationship as a string
            - source_node_uuid: UUID of the source entity
            - target_node_uuid: UUID of the target entity
            - valid_at: When this fact became valid (if temporal)
            - invalid_at: When this fact became invalid (if temporal)

        Example:
            results = await store.search("Who works at Acme?")
            for r in results:
                print(r["fact"])
        """
        edge_results = await self.client.search(
            query,
            center_node_uuid=center_node_uuid,
            num_results=num_results,
        )

        return [
            {
                "uuid": str(edge.uuid),
                "fact": edge.fact,
                "source_node_uuid": str(edge.source_node_uuid),
                "target_node_uuid": str(edge.target_node_uuid),
                "valid_at": getattr(edge, "valid_at", None),
                "invalid_at": getattr(edge, "invalid_at", None),
            }
            for edge in edge_results
        ]

    async def search_nodes(
        self,
        query: str,
        num_results: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search for entities (nodes) in the knowledge graph.

        Args:
            query: Natural language search query.
            num_results: Maximum number of results to return.

        Returns:
            List of node results, each containing:
            - uuid: Unique identifier of the node
            - name: Entity name
            - summary: Summary of the entity
            - labels: List of labels/types for this entity

        Example:
            nodes = await store.search_nodes("software engineers")
            for node in nodes:
                print(f"{node['name']}: {node['summary']}")
        """
        from graphiti_core.search.search_config_recipes import (
            NODE_HYBRID_SEARCH_RRF,
        )

        config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        config.limit = num_results

        result = await self.client._search(query=query, config=config)

        return [
            {
                "uuid": str(node.uuid),
                "name": node.name,
                "summary": getattr(node, "summary", ""),
                "labels": getattr(node, "labels", []),
                "created_at": getattr(node, "created_at", None),
            }
            for node in result.nodes
        ]

    async def get_node_by_name(self, name: str) -> dict[str, Any] | None:
        """
        Find a specific node by name.

        Args:
            name: The name of the entity to find.

        Returns:
            Node information dict, or None if not found.
        """
        from graphiti_core.search.search_config_recipes import (
            NODE_HYBRID_SEARCH_EPISODE_MENTIONS,
        )

        result = await self.client._search(name, NODE_HYBRID_SEARCH_EPISODE_MENTIONS)

        if result.nodes:
            node = result.nodes[0]
            return {
                "uuid": str(node.uuid),
                "name": node.name,
                "summary": getattr(node, "summary", ""),
                "labels": getattr(node, "labels", []),
            }
        return None

    async def search_as_context(
        self,
        query: str,
        num_results: int = 10,
    ) -> str:
        """
        Search and format results as context for an LLM.

        Args:
            query: Natural language search query.
            num_results: Maximum number of results to return.

        Returns:
            Formatted string with facts suitable for LLM context.

        Example:
            context = await store.search_as_context("company policies")
            prompt = f"Based on these facts:\\n{context}\\n\\nAnswer: ..."
        """
        edges = await self.search(query, num_results)

        if not edges:
            return "No relevant facts found in knowledge graph."

        facts = [f"- {edge['fact']}" for edge in edges]
        return "## Knowledge Graph Facts\n" + "\n".join(facts)

    def edges_to_facts_string(self, edges: list[dict[str, Any]]) -> str:
        """
        Convert edge results to a simple facts string.

        Args:
            edges: List of edge result dicts from search().

        Returns:
            Newline-separated string of facts.
        """
        if not edges:
            return "No facts found."
        return "- " + "\n- ".join(edge["fact"] for edge in edges)

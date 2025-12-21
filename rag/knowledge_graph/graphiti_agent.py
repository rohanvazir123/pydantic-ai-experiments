"""
Pydantic AI agent with Graphiti knowledge graph integration.

This module provides a conversational agent that uses Graphiti for:
- Persisting conversation facts to a knowledge graph
- Retrieving relevant context from prior conversations
- Searching product/entity information

Based on the Graphiti ShoeBot example, adapted for Pydantic AI.

Usage:
    from rag.knowledge_graph.graphiti_agent import (
        graphiti_agent,
        GraphitiAgentDeps,
        run_graphiti_agent,
    )

    # Simple usage
    result = await run_graphiti_agent("What products do you have?")
    print(result)

    # With deps for conversation context
    deps = GraphitiAgentDeps(user_name="john")
    result = await graphiti_agent.run("Hello!", deps=deps)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from rag.config.settings import load_settings
from rag.knowledge_graph.graphiti_config import create_graphiti_client

logger = logging.getLogger(__name__)


# =============================================================================
# DEPENDENCIES AND STATE
# =============================================================================


@dataclass
class GraphitiAgentDeps:
    """
    Dependencies for the Graphiti agent.

    Attributes:
        user_name: Name of the user for personalization.
        user_node_uuid: UUID of the user's node in the graph (auto-populated).
        graphiti_client: The Graphiti client instance (auto-created).
        center_node_uuid: Optional node UUID to center searches on.
    """

    user_name: str = "user"
    user_node_uuid: str = ""
    graphiti_client: Any = field(default=None, repr=False)
    center_node_uuid: str | None = None
    _initialized: bool = field(default=False, repr=False)

    async def initialize(self) -> None:
        """Initialize the Graphiti client and user node."""
        if self._initialized:
            return

        # Create client if not provided
        if self.graphiti_client is None:
            self.graphiti_client = create_graphiti_client()

        # Create or find user node
        try:
            from graphiti_core.nodes import EpisodeType
            from graphiti_core.search.search_config_recipes import (
                NODE_HYBRID_SEARCH_EPISODE_MENTIONS,
            )

            # Add user to graph if not exists
            await self.graphiti_client.add_episode(
                name="User Creation",
                episode_body=f"{self.user_name} is a user of the system",
                source=EpisodeType.text,
                reference_time=datetime.now(UTC),
                source_description="System",
            )

            # Get user node UUID
            nl = await self.graphiti_client._search(
                self.user_name, NODE_HYBRID_SEARCH_EPISODE_MENTIONS
            )
            if nl.nodes:
                self.user_node_uuid = str(nl.nodes[0].uuid)

            self._initialized = True
            logger.info(f"GraphitiAgentDeps initialized for user: {self.user_name}")

        except Exception as e:
            logger.error(f"Failed to initialize GraphitiAgentDeps: {e}")
            raise

    async def close(self) -> None:
        """Close the Graphiti client."""
        if self.graphiti_client is not None:
            await self.graphiti_client.close()
            self.graphiti_client = None
            self._initialized = False
            logger.info("GraphitiAgentDeps closed")


class GraphitiSearchResult(BaseModel):
    """Result from a knowledge graph search."""

    facts: list[str]
    query: str


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

# Default system prompt for the Graphiti agent
GRAPHITI_SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge graph.
You can search the knowledge graph for relevant information using the search_knowledge_graph tool.

When responding:
1. Use the knowledge graph to find relevant facts when the user asks questions
2. Provide accurate, helpful responses based on the available information
3. If you don't find relevant information, say so honestly
4. Be conversational and helpful

You have access to facts about users, products, and their relationships stored in the knowledge graph.
Use the search tool to find relevant information before answering questions about specific topics."""


def get_graphiti_model() -> OpenAIModel:
    """Get the LLM model for the Graphiti agent."""
    settings = load_settings()

    provider = OpenAIProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
    return OpenAIModel(settings.llm_model, provider=provider)


# Create the Graphiti agent
graphiti_agent = Agent(
    get_graphiti_model(),
    system_prompt=GRAPHITI_SYSTEM_PROMPT,
    deps_type=GraphitiAgentDeps,
)


# =============================================================================
# TOOLS
# =============================================================================


@graphiti_agent.tool
async def search_knowledge_graph(
    ctx: RunContext[GraphitiAgentDeps],
    query: str,
    num_results: int = 10,
) -> str:
    """
    Search the knowledge graph for relevant facts and relationships.

    Use this tool to find information about:
    - Users and their preferences
    - Products and their attributes
    - Relationships between entities
    - Historical facts from prior conversations

    Args:
        ctx: The run context with dependencies.
        query: Natural language search query.
        num_results: Maximum number of results to return.

    Returns:
        A formatted string of relevant facts from the knowledge graph.
    """
    deps = ctx.deps

    if deps.graphiti_client is None:
        return "Knowledge graph not available."

    try:
        # Search with optional center node for personalization
        edge_results = await deps.graphiti_client.search(
            query,
            center_node_uuid=deps.center_node_uuid or deps.user_node_uuid or None,
            num_results=num_results,
        )

        if not edge_results:
            return "No relevant facts found in the knowledge graph."

        # Format results
        facts = [f"- {edge.fact}" for edge in edge_results]
        return "Found the following facts:\n" + "\n".join(facts)

    except Exception as e:
        logger.error(f"Error searching knowledge graph: {e}")
        return f"Error searching knowledge graph: {e}"


@graphiti_agent.tool
async def search_entities(
    ctx: RunContext[GraphitiAgentDeps],
    query: str,
    num_results: int = 5,
) -> str:
    """
    Search for specific entities (people, products, organizations) in the knowledge graph.

    Use this tool when you need to find information about specific named entities.

    Args:
        ctx: The run context with dependencies.
        query: The entity name or description to search for.
        num_results: Maximum number of entities to return.

    Returns:
        A formatted string of matching entities.
    """
    deps = ctx.deps

    if deps.graphiti_client is None:
        return "Knowledge graph not available."

    try:
        from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

        config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        config.limit = num_results

        result = await deps.graphiti_client._search(query=query, config=config)

        if not result.nodes:
            return "No matching entities found."

        entities = []
        for node in result.nodes:
            summary = getattr(node, "summary", "No description available")
            entities.append(f"- **{node.name}**: {summary[:200]}")

        return "Found the following entities:\n" + "\n".join(entities)

    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        return f"Error searching entities: {e}"


# =============================================================================
# RESULT HANDLERS
# =============================================================================


@graphiti_agent.result_validator
async def persist_conversation(
    ctx: RunContext[GraphitiAgentDeps],
    result: str,
) -> str:
    """
    Persist the conversation to the knowledge graph after each response.

    This allows future queries to reference information from this conversation.
    """
    deps = ctx.deps

    if deps.graphiti_client is None or not deps._initialized:
        return result

    try:
        from graphiti_core.nodes import EpisodeType

        # Get the last user message from context if available
        # Note: In Pydantic AI, we don't have direct access to the message history
        # in the result validator, so we just store the response
        asyncio.create_task(
            deps.graphiti_client.add_episode(
                name="Agent Response",
                episode_body=f"Assistant responded to {deps.user_name}: {result[:500]}",
                source=EpisodeType.message,
                reference_time=datetime.now(UTC),
                source_description="Conversation",
            )
        )
    except Exception as e:
        logger.warning(f"Failed to persist conversation: {e}")

    return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def run_graphiti_agent(
    query: str,
    user_name: str = "user",
    message_history: list | None = None,
) -> str:
    """
    Run a query through the Graphiti agent.

    This is a convenience function that handles initialization and cleanup.

    Args:
        query: The user's query.
        user_name: Name of the user for personalization.
        message_history: Optional list of previous messages for context.

    Returns:
        The agent's response text.

    Example:
        response = await run_graphiti_agent(
            "What products do you have?",
            user_name="john",
        )
        print(response)
    """
    deps = GraphitiAgentDeps(user_name=user_name)

    try:
        await deps.initialize()

        if message_history:
            result = await graphiti_agent.run(
                query, deps=deps, message_history=message_history
            )
        else:
            result = await graphiti_agent.run(query, deps=deps)

        return result.output

    finally:
        await deps.close()


async def create_graphiti_session(
    user_name: str = "user",
) -> tuple[Agent, GraphitiAgentDeps]:
    """
    Create a Graphiti agent session for multi-turn conversations.

    Returns the agent and deps for use in a conversation loop.
    Remember to call deps.close() when done.

    Args:
        user_name: Name of the user for personalization.

    Returns:
        Tuple of (agent, deps) for running conversations.

    Example:
        agent, deps = await create_graphiti_session(user_name="alice")
        message_history = []

        try:
            # First turn
            result = await agent.run("Hello!", deps=deps)
            message_history.extend(result.new_messages())
            print(result.output)

            # Second turn with history
            result = await agent.run(
                "What did I just say?",
                deps=deps,
                message_history=message_history,
            )
            print(result.output)

        finally:
            await deps.close()
    """
    deps = GraphitiAgentDeps(user_name=user_name)
    await deps.initialize()
    return graphiti_agent, deps


# =============================================================================
# EXAMPLE USAGE
# =============================================================================


async def example_conversation():
    """Example of using the Graphiti agent with Pydantic AI."""
    print("Creating Graphiti agent session...")

    agent, deps = await create_graphiti_session(user_name="demo_user")
    message_history: list = []

    try:
        # First query
        print("\nUser: Hello! I'm interested in learning about your products.")
        result = await agent.run(
            "Hello! I'm interested in learning about your products.",
            deps=deps,
        )
        message_history.extend(result.new_messages())
        print(f"Assistant: {result.output}")

        # Follow-up with history
        print("\nUser: What categories do you have?")
        result = await agent.run(
            "What categories do you have?",
            deps=deps,
            message_history=message_history,
        )
        message_history.extend(result.new_messages())
        print(f"Assistant: {result.output}")

        # Use the search tool
        print("\nUser: Search for wool products")
        result = await agent.run(
            "Search for wool products in the knowledge graph",
            deps=deps,
            message_history=message_history,
        )
        print(f"Assistant: {result.output}")

    finally:
        await deps.close()
        print("\nSession closed.")


if __name__ == "__main__":
    asyncio.run(example_conversation())

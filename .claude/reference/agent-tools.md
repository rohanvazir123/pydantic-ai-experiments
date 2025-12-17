# Pydantic AI Agent & Tool Patterns

Reference guide for building Pydantic AI agents and tools for the RAG system.

## Agent Definition

### Basic Agent Setup

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps
from pydantic import BaseModel

class RAGState(BaseModel):
    """Minimal shared state for RAG agent."""
    pass

# Create agent with StateDeps
rag_agent = Agent(
    get_llm_model(),  # From providers.py
    deps_type=StateDeps[RAGState],
    system_prompt=MAIN_SYSTEM_PROMPT
)
```

### Agent with Dynamic Instructions

```python
from textwrap import dedent

@rag_agent.instructions
async def rag_instructions(ctx: RunContext[StateDeps[RAGState]]) -> str:
    """
    Dynamic instructions for the RAG agent.

    Args:
        ctx: The run context containing RAG state information.

    Returns:
        Instructions string for the RAG agent.
    """
    return dedent(
        """
        You are an intelligent RAG (Retrieval-Augmented Generation) assistant.

        INSTRUCTIONS:
        1. When the user asks a question, use the `search_knowledge_base` tool
        2. The tool will return relevant documents from the knowledge base
        3. Base your answer on the retrieved information
        4. Always cite which documents you're referencing
        5. If you cannot find relevant information, be honest about it
        6. Choose between:
           - "semantic" search for conceptual queries (default)
           - "hybrid" search for specific facts or keyword matching

        Be concise and helpful in your responses.
        """
    )
```

## Tool Definition

### Search Tool Pattern

```python
@rag_agent.tool
async def search_knowledge_base(
    ctx: RunContext[StateDeps[RAGState]],
    query: str,
    match_count: Optional[int] = 5,
    search_type: Optional[str] = "semantic"
) -> str:
    """
    Search the knowledge base for relevant information.

    Args:
        ctx: Agent runtime context with state dependencies
        query: Search query text
        match_count: Number of results to return (default: 5)
        search_type: Type of search - "semantic" or "hybrid" (default: semantic)

    Returns:
        String containing the retrieved information formatted for the LLM
    """
    try:
        # Initialize database connection
        agent_deps = AgentDependencies()
        await agent_deps.initialize()

        # Create context wrapper
        class DepsWrapper:
            def __init__(self, deps):
                self.deps = deps

        deps_ctx = DepsWrapper(agent_deps)

        # Perform search based on type
        if search_type == "hybrid":
            results = await hybrid_search(
                ctx=deps_ctx,
                query=query,
                match_count=match_count
            )
        else:
            results = await semantic_search(
                ctx=deps_ctx,
                query=query,
                match_count=match_count
            )

        # Clean up
        await agent_deps.cleanup()

        # Format results
        if not results:
            return "No relevant information found in the knowledge base."

        # Build formatted response
        response_parts = [f"Found {len(results)} relevant documents:\n"]

        for i, result in enumerate(results, 1):
            title = result.get('document_title', 'Unknown')
            content = result.get('content', '')
            similarity = result.get('combined_score', result.get('similarity', 0))

            response_parts.append(
                f"\n--- Document {i}: {title} (relevance: {similarity:.2f}) ---"
            )
            response_parts.append(content)

        return "\n".join(response_parts)

    except Exception as e:
        logger.exception("search_tool_failed", query=query, error=str(e))
        return f"Error searching knowledge base: {str(e)}"
```

### Tool Best Practices

1. **Return strings, not objects**: LLMs consume text, not Pydantic models
2. **Include context**: Format results with source attribution
3. **Handle errors gracefully**: Return helpful error messages, don't crash
4. **Clean up resources**: Use `try/finally` or context managers
5. **Log operations**: Log tool calls for debugging

### Alternative Tool Pattern (Direct Dependencies)

```python
@rag_agent.tool
async def search_knowledge_base(
    ctx: RunContext[StateDeps[RAGState]],
    query: str,
    match_count: Optional[int] = 5,
    search_type: Optional[str] = "semantic"
) -> str:
    """Search knowledge base with direct dependency access."""

    # Access dependencies from context
    state_deps = ctx.deps
    mongo_client = state_deps.mongo_client
    db = state_deps.db

    # Generate query embedding
    embedding = await state_deps.get_embedding(query)

    # Build and execute search pipeline
    if search_type == "hybrid":
        pipeline = build_hybrid_search_pipeline(query, embedding, match_count)
    else:
        pipeline = build_semantic_search_pipeline(embedding, match_count)

    results = await db.chunks.aggregate(pipeline).to_list(length=match_count)

    # Format and return
    return format_search_results(results)
```

## Streaming Implementation

### CLI Streaming Pattern

```python
async def stream_agent_interaction(
    user_input: str,
    message_history: List,
    deps: StateDeps[RAGState]
) -> tuple[str, List]:
    """
    Stream agent interaction with real-time tool call display.

    Args:
        user_input: The user's input text
        message_history: Conversation history
        deps: StateDeps with RAG state

    Returns:
        Tuple of (streamed_text, updated_message_history)
    """
    response_text = ""

    # Stream the agent execution
    async with rag_agent.iter(
        user_input,
        deps=deps,
        message_history=message_history
    ) as run:

        async for node in run:

            # User prompt node
            if Agent.is_user_prompt_node(node):
                pass  # Clean start

            # Model request node - stream thinking
            elif Agent.is_model_request_node(node):
                console.print("[bold blue]Assistant:[/bold blue] ", end="")

                async with node.stream(run.ctx) as request_stream:
                    async for event in request_stream:
                        # Text part start
                        if isinstance(event, PartStartEvent) and event.part.part_kind == 'text':
                            initial_text = event.part.content
                            if initial_text:
                                console.print(initial_text, end="")
                                response_text += initial_text

                        # Text delta (streaming)
                        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                            delta_text = event.delta.content_delta
                            if delta_text:
                                console.print(delta_text, end="")
                                response_text += delta_text

                console.print()  # New line

            # Tool calls
            elif Agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as tool_stream:
                    async for event in tool_stream:
                        event_type = type(event).__name__

                        if event_type == "FunctionToolCallEvent":
                            # Extract tool information
                            tool_name = "Unknown Tool"
                            args = None

                            if hasattr(event, 'part'):
                                part = event.part

                                if hasattr(part, 'tool_name'):
                                    tool_name = part.tool_name
                                elif hasattr(part, 'function_name'):
                                    tool_name = part.function_name

                                if hasattr(part, 'args'):
                                    args = part.args
                                elif hasattr(part, 'arguments'):
                                    args = part.arguments

                            console.print(f"  [cyan]Calling tool:[/cyan] [bold]{tool_name}[/bold]")

                            # Show search parameters
                            if args and isinstance(args, dict):
                                if 'query' in args:
                                    console.print(f"    [dim]Query:[/dim] {args['query']}")
                                if 'search_type' in args:
                                    console.print(f"    [dim]Type:[/dim] {args['search_type']}")
                                if 'match_count' in args:
                                    console.print(f"    [dim]Results:[/dim] {args['match_count']}")

                        elif event_type == "FunctionToolResultEvent":
                            console.print(f"  [green]✓ Search completed[/green]")

            # End node
            elif Agent.is_end_node(node):
                pass

    # Get new messages from run
    new_messages = run.result.new_messages()

    return (response_text.strip(), new_messages)
```

### Streaming Event Types

**PartStartEvent**: Initial text content
```python
if isinstance(event, PartStartEvent) and event.part.part_kind == 'text':
    initial_text = event.part.content
```

**PartDeltaEvent**: Streaming text updates
```python
if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
    delta_text = event.delta.content_delta
```

**FunctionToolCallEvent**: Tool execution started
```python
if type(event).__name__ == "FunctionToolCallEvent":
    tool_name = event.part.tool_name
    args = event.part.args
```

**FunctionToolResultEvent**: Tool execution completed
```python
if type(event).__name__ == "FunctionToolResultEvent":
    result = event.result
```

## Message History Management

### Maintaining Context

```python
class ConversationManager:
    """Manages conversation history for the agent."""

    def __init__(self):
        self.message_history: List = []

    async def send_message(
        self,
        user_input: str,
        deps: StateDeps[RAGState]
    ) -> str:
        """Send message and update history."""

        # Stream response
        response_text, new_messages = await stream_agent_interaction(
            user_input,
            self.message_history,
            deps
        )

        # Add new messages to history
        self.message_history.extend(new_messages)

        return response_text

    def clear_history(self):
        """Clear conversation history."""
        self.message_history = []

    def get_history_length(self) -> int:
        """Get number of messages in history."""
        return len(self.message_history)
```

### History Truncation

```python
def truncate_history(
    message_history: List,
    max_messages: int = 20
) -> List:
    """Keep only recent messages to avoid context limits."""
    if len(message_history) <= max_messages:
        return message_history

    # Keep most recent messages
    return message_history[-max_messages:]
```

## Dependencies Pattern

### AgentDependencies Class

```python
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
import openai

class AgentDependencies:
    """Dependencies injected into the agent context."""

    def __init__(self):
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.openai_client: Optional[openai.AsyncOpenAI] = None
        self.settings = None

    async def initialize(self):
        """Initialize external connections."""
        if not self.settings:
            self.settings = load_settings()

        # MongoDB
        if not self.mongo_client:
            self.mongo_client = AsyncIOMotorClient(self.settings.mongodb_uri)
            self.db = self.mongo_client[self.settings.mongodb_database]

            # Verify connection
            await self.mongo_client.admin.command('ping')

        # OpenAI client
        if not self.openai_client:
            self.openai_client = openai.AsyncOpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url
            )

    async def cleanup(self):
        """Clean up connections."""
        if self.mongo_client:
            self.mongo_client.close()
            self.mongo_client = None
            self.db = None

    async def get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        if not self.openai_client:
            await self.initialize()

        response = await self.openai_client.embeddings.create(
            model=self.settings.embedding_model,
            input=text
        )
        return response.data[0].embedding
```

## Provider Configuration

### LLM Provider Setup

```python
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel

def get_llm_model(model_choice: Optional[str] = None) -> OpenAIModel:
    """
    Get LLM model configuration.
    Supports any OpenAI-compatible API provider.
    """
    settings = load_settings()

    llm_choice = model_choice or settings.llm_model
    base_url = settings.llm_base_url
    api_key = settings.llm_api_key

    # Create provider
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)

    return OpenAIModel(llm_choice, provider=provider)
```

### Multiple Provider Support

```python
def get_llm_model_by_provider(provider_name: str) -> OpenAIModel:
    """Get model based on provider name."""
    settings = load_settings()

    providers = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini"
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "anthropic/claude-haiku-4.5"
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "model": "qwen2.5:14b-instruct"
        }
    }

    config = providers.get(provider_name, providers["openai"])

    provider = OpenAIProvider(
        base_url=config["base_url"],
        api_key=settings.llm_api_key
    )

    return OpenAIModel(config["model"], provider=provider)
```

## Error Handling in Tools

### Graceful Degradation

```python
@rag_agent.tool
async def search_knowledge_base(
    ctx: RunContext[StateDeps[RAGState]],
    query: str,
    match_count: Optional[int] = 5,
    search_type: Optional[str] = "semantic"
) -> str:
    """Search with comprehensive error handling."""

    try:
        # Initialize
        deps = AgentDependencies()
        await deps.initialize()

        # Perform search
        try:
            if search_type == "hybrid":
                results = await hybrid_search(...)
            else:
                results = await semantic_search(...)

        except OperationFailure as e:
            if e.code == 291:
                return (
                    "Vector search index is not configured. "
                    "Please set up indexes in MongoDB Atlas before searching."
                )
            raise

        except ConnectionFailure:
            return (
                "Could not connect to MongoDB. "
                "Please check your connection and try again."
            )

        # Format results
        if not results:
            return "No relevant information found in the knowledge base."

        return format_results(results)

    except Exception as e:
        logger.exception("search_tool_error", query=query)
        return f"An error occurred while searching: {str(e)}"

    finally:
        if deps:
            await deps.cleanup()
```

## Testing Agents and Tools

### Unit Testing Tools

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.unit
async def test_search_tool_success():
    """Test search tool with successful results."""

    # Mock context
    ctx = MagicMock()
    ctx.deps = MagicMock()

    # Mock dependencies
    mock_deps = AsyncMock()
    mock_deps.initialize = AsyncMock()
    mock_deps.cleanup = AsyncMock()

    # Mock search results
    mock_results = [
        {
            "document_title": "Test Doc",
            "content": "Test content",
            "similarity": 0.9
        }
    ]

    # Patch dependencies
    with patch('examples.tools.AgentDependencies', return_value=mock_deps):
        with patch('examples.tools.semantic_search', return_value=mock_results):
            result = await search_knowledge_base(
                ctx=ctx,
                query="test query",
                match_count=5,
                search_type="semantic"
            )

    assert "Found 1 relevant documents" in result
    assert "Test Doc" in result
    mock_deps.initialize.assert_called_once()
    mock_deps.cleanup.assert_called_once()
```

### Integration Testing Agent

```python
@pytest.mark.integration
async def test_agent_with_real_search():
    """Test agent with real MongoDB search."""

    # Setup
    state = RAGState()
    deps = StateDeps[RAGState](state=state)

    # Initialize real connections
    agent_deps = AgentDependencies()
    await agent_deps.initialize()

    try:
        # Run agent
        result = await rag_agent.run(
            "What is the technical architecture?",
            deps=deps
        )

        # Verify
        assert result.output
        assert len(result.output) > 0

    finally:
        await agent_deps.cleanup()
```

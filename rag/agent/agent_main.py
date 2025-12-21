#!/usr/bin/env python3
"""Conversational CLI with real-time streaming and tool call visibility."""

import asyncio
from typing import Any

# Import our agent and dependencies
import rag_agent
from config.settings import load_settings
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.ag_ui import StateDeps
from pydantic_ai.messages import (
    ModelMessage,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
)
from rag_agent import RAGState
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# Load environment variables
load_dotenv(override=True)

console = Console()


async def stream_agent_interaction(
    user_input: str,
    message_history: list[ModelMessage],
    deps: StateDeps[RAGState],
) -> tuple[str, list[ModelMessage]]:
    """
    Stream agent interaction with real-time tool call display.

    Args:
        user_input: The user's input text
        message_history: List of ModelMessage objects (ModelRequest/ModelResponse)
                        for conversation context
        deps: StateDeps with RAG state

    Returns:
        Tuple of (streamed_text, updated_message_history)
    """
    try:
        return await _stream_agent(user_input, deps, message_history)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        return ("", [])


# =============================================================================
# HELPER FUNCTIONS FOR STREAMING AGENT EXECUTION
# =============================================================================
# These helper functions break down the complex streaming logic into smaller,
# more manageable pieces. Each function handles a specific aspect of the
# streaming process.
# =============================================================================


def _extract_tool_info(event: Any) -> tuple[str, dict[str, Any] | None]:
    """
    Extract tool name and arguments from a FunctionToolCallEvent.

    This function navigates the event's structure to find the tool name and
    arguments, handling multiple possible attribute naming conventions that
    different Pydantic AI versions might use.

    Args:
        event: A FunctionToolCallEvent object containing tool call information.
               The event should have a 'part' attribute containing the tool details.

    Returns:
        A tuple of (tool_name, args) where:
        - tool_name: String name of the tool being called (defaults to "Unknown Tool")
        - args: Dictionary of arguments passed to the tool, or None if not found

    Example:
        >>> tool_name, args = _extract_tool_info(event)
        >>> print(f"Calling {tool_name} with {args}")
    """
    tool_name = "Unknown Tool"
    args = None

    # The event's 'part' attribute contains the actual tool call details
    if hasattr(event, "part"):
        part = event.part

        # Try multiple attribute names for tool name (API compatibility)
        # Different versions of Pydantic AI may use different attribute names
        if hasattr(part, "tool_name"):
            tool_name = part.tool_name
        elif hasattr(part, "function_name"):
            tool_name = part.function_name
        elif hasattr(part, "name"):
            tool_name = part.name

        # Try multiple attribute names for arguments (API compatibility)
        if hasattr(part, "args"):
            args = part.args
        elif hasattr(part, "arguments"):
            args = part.arguments

    return tool_name, args


def _display_tool_args(args: dict[str, Any] | Any | None) -> None:
    """
    Display tool arguments in a formatted, user-friendly way.

    For search-related tools, this function shows specific fields like query,
    search_type, and match_count in a structured format. For other tools,
    it displays a truncated string representation of all arguments.

    Args:
        args: The arguments passed to the tool. Can be:
              - A dictionary with named arguments
              - Any other object (will be converted to string)
              - None (nothing will be displayed)

    Side Effects:
        Prints formatted argument information to the console using Rich formatting.

    Example output for search tools:
        Query: What is machine learning?
        Type: hybrid
        Results: 5

    Example output for other tools:
        Args: {"param1": "value1", "param2": "value2"}
    """
    if not args:
        return

    if isinstance(args, dict):
        # For search tools, display specific arguments in a structured way
        # These are the common parameters used by the RAG search tool
        if "query" in args:
            console.print(f"    [dim]Query:[/dim] {args['query']}")
        if "search_type" in args:
            console.print(f"    [dim]Type:[/dim] {args['search_type']}")
        if "match_count" in args:
            console.print(f"    [dim]Results:[/dim] {args['match_count']}")
    else:
        # For non-dict arguments, show a truncated string representation
        # Truncate long argument strings to keep output readable
        args_str = str(args)
        if len(args_str) > 100:
            args_str = args_str[:97] + "..."
        console.print(f"    [dim]Args: {args_str}[/dim]")


async def _handle_model_request_node(node: Any, ctx: Any) -> str:
    """
    Handle streaming text output from the model request node.

    This function processes the real-time streaming of the LLM's response,
    displaying text as it's generated. It handles two types of events:
    1. PartStartEvent: The beginning of a text part with initial content
    2. PartDeltaEvent: Incremental text updates (token-by-token streaming)

    Args:
        node: The model request node from the agent execution graph.
              This node represents a request to the LLM for a response.
        ctx: The run context containing state and dependencies needed
             for streaming the node's output.

    Returns:
        The complete text that was streamed from the model.

    Side Effects:
        - Prints "[bold blue]Assistant:[/bold blue] " prefix before streaming
        - Prints each text chunk as it arrives (real-time streaming effect)
        - Prints a newline after streaming completes
    """
    response_text = ""

    # Display the assistant label before the streaming content
    console.print("[bold blue]Assistant:[/bold blue] ", end="")

    # Open a streaming context to receive real-time events from the model
    async with node.stream(ctx) as request_stream:
        async for event in request_stream:
            # PartStartEvent: Fired when a new part (text, tool call, etc.) begins
            # We only care about text parts here
            if isinstance(event, PartStartEvent) and event.part.part_kind == "text":
                initial_text = event.part.content
                if initial_text:
                    console.print(initial_text, end="")
                    response_text += initial_text

            # PartDeltaEvent with TextPartDelta: Incremental text updates
            # This is the main streaming mechanism - each delta contains a small
            # chunk of the response (usually a few tokens)
            elif isinstance(event, PartDeltaEvent) and isinstance(
                event.delta, TextPartDelta
            ):
                delta_text = event.delta.content_delta
                if delta_text:
                    console.print(delta_text, end="")
                    response_text += delta_text

    # Add newline after the complete response for proper formatting
    console.print()

    return response_text


async def _handle_tool_call_node(node: Any, ctx: Any) -> None:
    """
    Handle tool call events and display their execution status.

    This function monitors tool execution in real-time, showing which tools
    are being called, what arguments they receive, and when they complete.
    This provides visibility into the agent's "thinking" process when it
    decides to use tools like the RAG search.

    Args:
        node: The call tools node from the agent execution graph.
              This node represents one or more tool invocations.
        ctx: The run context containing state and dependencies needed
             for streaming the node's output.

    Side Effects:
        - Prints tool name when a tool is called
        - Prints tool arguments (formatted based on tool type)
        - Prints success message when tool execution completes

    Example console output:
        Calling tool: search_knowledge_base
          Query: What is machine learning?
          Type: hybrid
          Results: 5
        Search completed successfully
    """
    # Stream tool execution events in real-time
    async with node.stream(ctx) as tool_stream:
        async for event in tool_stream:
            # Get the event type name for comparison
            # We use string comparison because the event classes may not be
            # directly importable in all contexts
            event_type = type(event).__name__

            if event_type == "FunctionToolCallEvent":
                # A tool is being invoked - extract and display its information
                tool_name, args = _extract_tool_info(event)
                console.print(f"  [cyan]Calling tool:[/cyan] [bold]{tool_name}[/bold]")
                _display_tool_args(args)

            elif event_type == "FunctionToolResultEvent":
                # The tool has finished executing
                console.print("  [green]Search completed successfully[/green]")


# =============================================================================
# MAIN STREAMING FUNCTION
# =============================================================================


async def _stream_agent(
    user_input: str,
    deps: StateDeps[RAGState],
    message_history: list[ModelMessage],
) -> tuple[str, list[ModelMessage]]:
    """
    Stream the agent execution and return the response with updated message history.

    This is the core function that orchestrates the streaming of an agent's
    execution. It iterates through the agent's execution graph, handling each
    node type appropriately:

    1. User Prompt Node: The initial user input (no action needed)
    2. Model Request Node: LLM generating a response (stream text in real-time)
    3. Call Tools Node: Agent invoking tools like search (display tool activity)
    4. End Node: Execution complete (no action needed)

    The function uses Pydantic AI's streaming capabilities to provide real-time
    feedback to the user as the LLM generates its response and uses tools.

    Args:
        user_input: The user's input text/question to send to the agent.
        deps: StateDeps wrapper containing the RAGState with conversation context
              and any other dependencies the agent needs.
        message_history: List of previous ModelRequest/ModelResponse objects
                        that provide conversation context to the agent.

    Returns:
        A tuple containing:
        - response: The complete text response from the agent (either streamed
                   text or final output if streaming produced no text)
        - new_messages: List of new message objects from this run that should
                       be added to the conversation history

    Raises:
        Any exceptions from the agent execution are propagated to the caller
        (handled by stream_agent_interaction wrapper).

    Example:
        >>> response, new_msgs = await _stream_agent("What is RAG?", deps, history)
        >>> print(response)
        "RAG (Retrieval-Augmented Generation) is..."
    """
    response_text = ""

    # -------------------------------------------------------------------------
    # Stream the agent execution using Pydantic AI's iter() context manager.
    # This provides access to the execution graph as a series of nodes.
    # -------------------------------------------------------------------------
    async with rag_agent.iter(
        user_input, deps=deps, message_history=message_history
    ) as run:
        # Iterate through each node in the agent's execution graph
        async for node in run:
            # -----------------------------------------------------------------
            # USER PROMPT NODE
            # Represents the initial user input being processed.
            # No action needed - the prompt is already set up.
            # -----------------------------------------------------------------
            if Agent.is_user_prompt_node(node):
                pass  # Clean start, nothing to display

            # -----------------------------------------------------------------
            # MODEL REQUEST NODE
            # The LLM is generating a response. Stream the text in real-time
            # so the user sees tokens as they're generated.
            # -----------------------------------------------------------------
            elif Agent.is_model_request_node(node):
                response_text = await _handle_model_request_node(node, run.ctx)

            # -----------------------------------------------------------------
            # CALL TOOLS NODE
            # The agent is invoking one or more tools (e.g., RAG search).
            # Display tool calls and results for transparency.
            # -----------------------------------------------------------------
            elif Agent.is_call_tools_node(node):
                await _handle_tool_call_node(node, run.ctx)

            # -----------------------------------------------------------------
            # END NODE
            # The agent has finished execution. No action needed here as we
            # handle the results after the loop.
            # -----------------------------------------------------------------
            elif Agent.is_end_node(node):
                pass  # Execution complete

    # -------------------------------------------------------------------------
    # PROCESS RESULTS
    # Extract the new messages and final output from the completed run.
    # -------------------------------------------------------------------------

    # Get new messages from this run to add to conversation history
    # This includes both the user's message and the agent's response
    new_messages = run.result.new_messages()

    # Get the final output - use streamed text if available, otherwise
    # fall back to the result's output attribute
    final_output = (
        run.result.output if hasattr(run.result, "output") else str(run.result)
    )
    response = response_text.strip() or final_output

    return (response, new_messages)


def display_welcome():
    """Display welcome message with configuration info."""
    settings = load_settings()

    welcome = Panel(
        "[bold blue]MongoDB RAG Agent[/bold blue]\n\n"
        "[green]Intelligent knowledge base search with MongoDB Atlas Vector Search[/green]\n"
        f"[dim]LLM: {settings.llm_model}[/dim]\n\n"
        "[dim]Type 'exit' to quit, 'info' for system info, 'clear' to clear screen[/dim]",
        style="blue",
        padding=(1, 2),
    )
    console.print(welcome)
    console.print()


async def agent_main():
    """Main conversation loop."""

    # Show welcome
    display_welcome()

    # Create the state that the agent will use
    state = RAGState()

    # Create StateDeps wrapper with the state
    deps = StateDeps[RAGState](state=state)

    console.print("[bold green]✓[/bold green] Search system initialized\n")

    # Initialize message history with proper Pydantic AI message objects
    message_history: list[ModelMessage] = []

    try:
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("[bold green]You").strip()

                # Handle special commands
                if user_input.lower() in ["exit", "quit", "q"]:
                    console.print("\n[yellow]👋 Goodbye![/yellow]")
                    break

                elif user_input.lower() == "info":
                    settings = load_settings()
                    console.print(
                        Panel(
                            f"[cyan]LLM Provider:[/cyan] {settings.llm_provider}\n"
                            f"[cyan]LLM Model:[/cyan] {settings.llm_model}\n"
                            f"[cyan]Embedding Model:[/cyan] {settings.embedding_model}\n"
                            f"[cyan]Default Match Count:[/cyan] {settings.default_match_count}\n"
                            f"[cyan]Default Text Weight:[/cyan] {settings.default_text_weight}",
                            title="System Configuration",
                            border_style="magenta",
                        )
                    )
                    continue

                elif user_input.lower() == "clear":
                    console.clear()
                    display_welcome()
                    continue

                if not user_input:
                    continue

                # Stream the interaction and get response
                response_text, new_messages = await stream_agent_interaction(
                    user_input, message_history, deps
                )

                # Add new messages to history (includes both user prompt and agent response)
                message_history.extend(new_messages)

                # Add spacing after response
                console.print()

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
                continue

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                import traceback

                traceback.print_exc()
                continue

    finally:
        console.print("\n[dim]Goodbye![/dim]")


if __name__ == "__main__":
    asyncio.run(agent_main())

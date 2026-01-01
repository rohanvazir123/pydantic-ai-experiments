#!/usr/bin/env python3
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
Streamlit UI for the MongoDB RAG Agent.

This module provides a chat-based web interface for interacting with the RAG agent.
It displays real-time streaming responses and tool call visibility.

Usage:
    streamlit run rag/agent/streamlit_app.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.ag_ui import StateDeps
from pydantic_ai.messages import (
    ModelMessage,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
)

# Import our agent and dependencies (after path setup)
from rag.agent.rag_agent import RAGState, agent
from rag.config.settings import load_settings

# Load environment variables
load_dotenv(override=True)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="MongoDB RAG Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        # Chat messages for display (role, content format)
        st.session_state.messages = []

    if "message_history" not in st.session_state:
        # Pydantic AI message history for context
        st.session_state.message_history: list[ModelMessage] = []

    if "deps" not in st.session_state:
        # Create the state - retriever is lazy-initialized on first use
        # This avoids event loop issues (created in Streamlit loop, used in agent loop)
        state = RAGState()
        st.session_state.deps = StateDeps[RAGState](state=state)
        st.session_state.rag_state = state  # Keep reference for cleanup


# =============================================================================
# HELPER FUNCTIONS FOR STREAMING
# =============================================================================


def extract_tool_info(event: Any) -> tuple[str, dict[str, Any] | None]:
    """
    Extract tool name and arguments from a FunctionToolCallEvent.

    Args:
        event: A FunctionToolCallEvent object containing tool call information.

    Returns:
        A tuple of (tool_name, args).
    """
    tool_name = "Unknown Tool"
    args = None

    if hasattr(event, "part"):
        part = event.part

        # Try multiple attribute names for tool name (API compatibility)
        if hasattr(part, "tool_name"):
            tool_name = part.tool_name
        elif hasattr(part, "function_name"):
            tool_name = part.function_name
        elif hasattr(part, "name"):
            tool_name = part.name

        # Try multiple attribute names for arguments
        if hasattr(part, "args"):
            args = part.args
        elif hasattr(part, "arguments"):
            args = part.arguments

    return tool_name, args


def format_tool_args(args: dict[str, Any] | Any | None) -> str:
    """
    Format tool arguments for display.

    Args:
        args: The arguments passed to the tool.

    Returns:
        Formatted string representation of the arguments.
    """
    if not args:
        return ""

    if isinstance(args, dict):
        parts = []
        if "query" in args:
            parts.append(f"**Query:** {args['query']}")
        if "search_type" in args:
            parts.append(f"**Type:** {args['search_type']}")
        if "match_count" in args:
            parts.append(f"**Results:** {args['match_count']}")
        return "\n".join(parts) if parts else str(args)
    else:
        args_str = str(args)
        if len(args_str) > 200:
            args_str = args_str[:197] + "..."
        return args_str


async def stream_agent_response(
    user_input: str,
    deps: StateDeps[RAGState],
    message_history: list[ModelMessage],
    response_placeholder: Any,
    status_placeholder: Any,
) -> tuple[str, list[ModelMessage]]:
    """
    Stream the agent response and update the UI in real-time.

    Args:
        user_input: The user's input text.
        deps: StateDeps with RAG state.
        message_history: List of previous messages for context.
        response_placeholder: Streamlit placeholder for the response text.
        status_placeholder: Streamlit placeholder for status updates.

    Returns:
        Tuple of (response_text, new_messages).
    """
    response_text = ""
    tool_calls: list[str] = []

    async with agent.iter(
        user_input, deps=deps, message_history=message_history
    ) as run:
        async for node in run:
            # Handle user prompt node
            if Agent.is_user_prompt_node(node):
                pass  # Nothing to display

            # Handle model request node - stream the text response
            elif Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as request_stream:
                    async for event in request_stream:
                        # Handle text part start events
                        if (
                            isinstance(event, PartStartEvent)
                            and event.part.part_kind == "text"
                        ):
                            initial_text = event.part.content
                            if initial_text:
                                response_text += initial_text
                                response_placeholder.markdown(response_text + "▌")

                        # Handle text delta events for streaming
                        elif isinstance(event, PartDeltaEvent) and isinstance(
                            event.delta, TextPartDelta
                        ):
                            delta_text = event.delta.content_delta
                            if delta_text:
                                response_text += delta_text
                                response_placeholder.markdown(response_text + "▌")

            # Handle tool calls
            elif Agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as tool_stream:
                    async for event in tool_stream:
                        event_type = type(event).__name__

                        if event_type == "FunctionToolCallEvent":
                            tool_name, args = extract_tool_info(event)
                            tool_info = f"🔧 **Calling:** `{tool_name}`"
                            if args:
                                tool_info += f"\n{format_tool_args(args)}"
                            tool_calls.append(tool_info)
                            status_placeholder.info("\n\n".join(tool_calls))

                        elif event_type == "FunctionToolResultEvent":
                            tool_calls.append("✅ Search completed")
                            status_placeholder.success("\n\n".join(tool_calls))

            # Handle end node
            elif Agent.is_end_node(node):
                pass

    # Final update without cursor
    response_placeholder.markdown(response_text)

    # Get new messages from this run
    new_messages = run.result.new_messages()

    # Get final output if streaming produced no text
    if not response_text.strip():
        final_output = (
            run.result.output if hasattr(run.result, "output") else str(run.result)
        )
        response_text = final_output
        response_placeholder.markdown(response_text)

    return response_text, new_messages


# =============================================================================
# SIDEBAR
# =============================================================================


def render_sidebar() -> None:
    """Render the sidebar with configuration info and controls."""
    with st.sidebar:
        st.title("🔍 MongoDB RAG Agent")
        st.markdown("---")

        # Configuration info
        st.subheader("Configuration")
        settings = load_settings()

        st.markdown(f"**LLM Provider:** `{settings.llm_provider}`")
        st.markdown(f"**LLM Model:** `{settings.llm_model}`")
        st.markdown(f"**Embedding Model:** `{settings.embedding_model}`")
        st.markdown(f"**Default Results:** `{settings.default_match_count}`")

        st.markdown("---")

        # Clear conversation button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.message_history = []
            st.rerun()

        st.markdown("---")

        # Help section
        with st.expander("ℹ️ Help"):
            st.markdown(
                """
            **How to use:**
            1. Type your question in the chat input
            2. The agent will search the knowledge base
            3. You'll see tool calls and streaming responses

            **Example queries:**
            - What does the company do?
            - What is the PTO policy?
            - What technologies are used?
            """
            )


# =============================================================================
# MAIN CHAT INTERFACE
# =============================================================================


def render_chat() -> None:
    """Render the main chat interface."""
    st.title("💬 Chat with RAG Agent")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the knowledge base..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response with streaming
        with st.chat_message("assistant"):
            # Create placeholders for streaming content
            status_placeholder = st.empty()
            response_placeholder = st.empty()

            # Run the async agent
            response_text, new_messages = asyncio.run(
                stream_agent_response(
                    user_input=prompt,
                    deps=st.session_state.deps,
                    message_history=st.session_state.message_history,
                    response_placeholder=response_placeholder,
                    status_placeholder=status_placeholder,
                )
            )

            # Clear status after completion
            status_placeholder.empty()

        # Add assistant message to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )

        # Update Pydantic AI message history
        st.session_state.message_history.extend(new_messages)


# =============================================================================
# MAIN APPLICATION
# =============================================================================


def main() -> None:
    """Main application entry point."""
    # Initialize session state
    init_session_state()

    # Render sidebar
    render_sidebar()

    # Render main chat interface
    render_chat()


if __name__ == "__main__":
    main()

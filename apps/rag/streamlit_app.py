"""
RAG + PDF Question Generator — Streamlit UI.

Chat interface for the Legal Contract RAG agent.
Streams tool calls and responses in real time.

Usage:
    streamlit run apps/rag/streamlit_app.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

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

from rag.agent.rag_agent import RAGState, agent
from rag.config.settings import load_settings

load_dotenv(override=True)

st.set_page_config(
    page_title="Legal Contract Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "message_history" not in st.session_state:
        st.session_state.message_history: list[ModelMessage] = []
    if "deps" not in st.session_state:
        state = RAGState()
        st.session_state.deps = StateDeps[RAGState](state=state)
        st.session_state.rag_state = state


def extract_tool_info(event: Any) -> tuple[str, dict[str, Any] | None]:
    tool_name = "Unknown Tool"
    args = None
    if hasattr(event, "part"):
        part = event.part
        for attr in ("tool_name", "function_name", "name"):
            if hasattr(part, attr):
                tool_name = getattr(part, attr)
                break
        for attr in ("args", "arguments"):
            if hasattr(part, attr):
                args = getattr(part, attr)
                break
    return tool_name, args


def format_tool_args(args: dict[str, Any] | Any | None) -> str:
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
    args_str = str(args)
    return args_str[:197] + "..." if len(args_str) > 200 else args_str


async def stream_agent_response(
    user_input: str,
    deps: StateDeps[RAGState],
    message_history: list[ModelMessage],
    response_placeholder: Any,
    status_placeholder: Any,
) -> tuple[str, list[ModelMessage]]:
    response_text = ""
    tool_calls: list[str] = []

    async with agent.iter(user_input, deps=deps, message_history=message_history) as run:
        async for node in run:
            if Agent.is_user_prompt_node(node):
                pass
            elif Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as request_stream:
                    async for event in request_stream:
                        if isinstance(event, PartStartEvent) and event.part.part_kind == "text":
                            if event.part.content:
                                response_text += event.part.content
                                response_placeholder.markdown(response_text + "▌")
                        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                            if event.delta.content_delta:
                                response_text += event.delta.content_delta
                                response_placeholder.markdown(response_text + "▌")
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
            elif Agent.is_end_node(node):
                pass

    response_placeholder.markdown(response_text)
    new_messages = run.result.new_messages()

    if not response_text.strip():
        final_output = run.result.output if hasattr(run.result, "output") else str(run.result)
        response_text = final_output
        response_placeholder.markdown(response_text)

    return response_text, new_messages


def render_sidebar() -> None:
    with st.sidebar:
        st.title("⚖️ Legal Contract Assistant")
        st.markdown("---")
        st.subheader("Configuration")
        settings = load_settings()
        st.markdown(f"**LLM Provider:** `{settings.llm_provider}`")
        st.markdown(f"**LLM Model:** `{settings.llm_model}`")
        st.markdown(f"**Embedding Model:** `{settings.embedding_model}`")
        st.markdown(f"**Default Results:** `{settings.default_match_count}`")
        st.markdown("---")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.message_history = []
            st.rerun()
        st.markdown("---")
        with st.expander("ℹ️ Help"):
            st.markdown(
                """
**How to use:**
1. Type your question in the chat input
2. The agent picks the right tool(s) and streams the response
3. Tool calls are shown as they happen

**Tools available:**
- `search_knowledge_base` — hybrid vector + text search over contract chunks
- `search_knowledge_graph` — entity & single-hop relationship lookup
- `run_graph_query` — custom Cypher for multi-hop traversal

**Example queries:**
- Which contracts have Amazon as a party?
- Find all contracts governed by Delaware law
- What does the license grant say in contracts where Google is the licensee?

**API:** `uvicorn apps.rag.api:app --port 8000`
            """
            )


def render_chat() -> None:
    st.title("💬 Ask about your contracts")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask a question about the knowledge base..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            response_text, new_messages = asyncio.run(
                stream_agent_response(
                    user_input=prompt,
                    deps=st.session_state.deps,
                    message_history=st.session_state.message_history,
                    response_placeholder=response_placeholder,
                    status_placeholder=status_placeholder,
                )
            )
            status_placeholder.empty()
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.session_state.message_history.extend(new_messages)


def main() -> None:
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()

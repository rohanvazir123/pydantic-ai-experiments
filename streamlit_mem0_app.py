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
Standalone Streamlit app with Pydantic AI agent and Mem0 memory.

A simple chat application that demonstrates:
- Pydantic AI agent for conversation
- Mem0 for persistent user memory across sessions
- Streamlit for the UI

Usage:
    streamlit run streamlit_mem0_app.py
"""

import asyncio

import streamlit as st
from mem0 import Memory
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Page config
st.set_page_config(
    page_title="Chat with Memory",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Chat with Memory")
st.caption("Powered by Pydantic AI + Mem0")


# --- Configuration ---
@st.cache_resource
def get_settings():
    """Load settings from environment."""
    import os

    from dotenv import load_dotenv

    load_dotenv()

    return {
        "llm_model": os.getenv("LLM_MODEL", "llama3.1:8b"),
        "llm_base_url": os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        "llm_api_key": os.getenv("LLM_API_KEY", "ollama"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest"),
        "embedding_dimension": int(os.getenv("EMBEDDING_DIMENSION", "768")),
        "database_url": os.getenv("DATABASE_URL", ""),
        "mem0_collection": os.getenv("MEM0_COLLECTION_NAME", "mem0_memories"),
    }


def _parse_database_url(database_url: str) -> dict:
    """Parse DATABASE_URL into connection parameters for pgvector."""
    from urllib.parse import parse_qs, urlparse

    if not database_url:
        raise ValueError("DATABASE_URL not configured")

    parsed = urlparse(database_url)

    return {
        "user": parsed.username or "",
        "password": parsed.password or "",
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "dbname": parsed.path.lstrip("/") if parsed.path else "",
    }


# --- Mem0 Setup ---
@st.cache_resource
def get_mem0():
    """Initialize Mem0 with PostgreSQL/pgvector backend."""
    settings = get_settings()

    ollama_base = settings["llm_base_url"].replace("/v1", "")
    db_config = _parse_database_url(settings["database_url"])

    config = {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": settings["llm_model"],
                "ollama_base_url": ollama_base,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": settings["embedding_model"],
                "ollama_base_url": ollama_base,
            },
        },
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "collection_name": settings["mem0_collection"],
                "embedding_model_dims": settings["embedding_dimension"],
                "dbname": db_config["dbname"],
                "user": db_config["user"],
                "password": db_config["password"],
                "host": db_config["host"],
                "port": db_config["port"],
            },
        },
        "version": "v1.1",
    }

    return Memory.from_config(config)


# --- Agent Setup ---
@st.cache_resource
def get_agent():
    """Create a simple Pydantic AI agent."""
    settings = get_settings()

    provider = OpenAIProvider(
        base_url=settings["llm_base_url"],
        api_key=settings["llm_api_key"],
    )
    model = OpenAIChatModel(settings["llm_model"], provider=provider)

    system_prompt = """You are a helpful assistant with memory capabilities.
You remember information about users from previous conversations.
When given user context, use it to personalize your responses.
Be concise and friendly."""

    return Agent(model, system_prompt=system_prompt)


def get_user_context(user_id: str) -> str:
    """Retrieve all memories for the user (simple approach without vector search)."""
    import psycopg2

    try:
        settings = get_settings()
        db_config = _parse_database_url(settings["database_url"])

        conn = psycopg2.connect(
            dbname=db_config["dbname"],
            user=db_config["user"],
            password=db_config["password"],
            host=db_config["host"],
            port=db_config["port"],
            sslmode="require",
        )
        cursor = conn.cursor()

        # Mem0 stores memories in a table named after the collection_name
        table_name = settings["mem0_collection"]

        # Query for memories belonging to this user
        cursor.execute(
            f"""
            SELECT metadata->>'data' as memory_text
            FROM {table_name}
            WHERE metadata->>'user_id' = %s
            LIMIT 10
            """,
            (user_id,),
        )

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return ""

        lines = ["[User Memory Context]"]
        for row in rows:
            memory_text = row[0]
            if memory_text:
                lines.append(f"- {memory_text}")

        return "\n".join(lines)

    except Exception:
        return ""


def save_to_memory(mem0: Memory, user_id: str, text: str):
    """Save conversation to memory."""
    try:
        mem0.add(text, user_id=user_id, infer=True)
    except Exception as e:
        st.warning(f"Could not save memory: {e}")


def delete_all_memories(user_id: str):
    """Delete all memories for a user (workaround for Mem0 bug)."""
    import psycopg2

    settings = get_settings()
    db_config = _parse_database_url(settings["database_url"])

    conn = psycopg2.connect(
        dbname=db_config["dbname"],
        user=db_config["user"],
        password=db_config["password"],
        host=db_config["host"],
        port=db_config["port"],
        sslmode="require",
    )
    cursor = conn.cursor()

    table_name = settings["mem0_collection"]

    # Delete all rows where metadata.user_id matches
    cursor.execute(
        f"""
        DELETE FROM {table_name}
        WHERE metadata->>'user_id' = %s
        """,
        (user_id,),
    )

    deleted_count = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()

    return deleted_count


async def run_agent(agent: Agent, prompt: str) -> str:
    """Run the agent and return response."""
    result = await agent.run(prompt)
    return result.output


# --- Sidebar ---
with st.sidebar:
    st.header("Settings")

    user_id = st.text_input(
        "User ID",
        value="default_user",
        help="Unique identifier for memory storage",
    )

    st.divider()

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear All Memories"):
        try:
            count = delete_all_memories(user_id)
            st.success(f"Cleared {count} memories for {user_id}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    # Show current memories
    if st.button("Show My Memories"):
        try:
            mem0 = get_mem0()
            memories = mem0.get_all(user_id=user_id)

            if isinstance(memories, dict):
                memories = memories.get("results", [])

            if memories:
                st.write(f"**{len(memories)} memories found:**")
                for mem in memories:
                    memory_text = mem.get("memory", str(mem))
                    st.info(memory_text)
            else:
                st.write("No memories stored yet.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    st.caption("Model: " + get_settings()["llm_model"])


# --- Chat Interface ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent and mem0
    agent = get_agent()
    mem0 = get_mem0()

    # Build prompt with memory context
    user_context = get_user_context(user_id)

    if user_context:
        enhanced_prompt = f"{user_context}\n\nUser: {prompt}"
    else:
        enhanced_prompt = prompt

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = asyncio.run(run_agent(agent, enhanced_prompt))
            st.markdown(response)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Save conversation to memory (async in background)
    conversation = f"User said: {prompt}\nAssistant replied: {response}"
    save_to_memory(mem0, user_id, conversation)

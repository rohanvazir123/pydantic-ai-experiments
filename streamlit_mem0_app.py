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
        "mongodb_uri": os.getenv("MONGODB_URI", ""),
        "mongodb_database": os.getenv("MONGODB_DATABASE", "rag_db"),
        "mem0_collection": os.getenv("MEM0_COLLECTION_NAME", "mem0_memories"),
    }


# --- Mem0 Setup ---
@st.cache_resource
def get_mem0():
    """Initialize Mem0 with MongoDB backend."""
    settings = get_settings()

    ollama_base = settings["llm_base_url"].replace("/v1", "")

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
            "provider": "mongodb",
            "config": {
                "collection_name": settings["mem0_collection"],
                "embedding_model_dims": settings["embedding_dimension"],
                "mongo_uri": settings["mongodb_uri"],
                "db_name": settings["mongodb_database"],
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
    from pymongo import MongoClient

    try:
        settings = get_settings()
        client = MongoClient(settings["mongodb_uri"])
        db = client[settings["mongodb_database"]]
        col = db[settings["mem0_collection"]]

        # Get all memories for this user (simple filter, no vector search)
        docs = list(col.find({"payload.user_id": user_id}).limit(10))
        client.close()

        if not docs:
            return ""

        lines = ["[User Memory Context]"]
        for doc in docs:
            payload = doc.get("payload", {})
            memory_text = payload.get("data", "")
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

    from dotenv import load_dotenv
    from pymongo import MongoClient

    load_dotenv()
    settings = get_settings()

    client = MongoClient(settings["mongodb_uri"])
    db = client[settings["mongodb_database"]]
    col = db[settings["mem0_collection"]]

    # Delete all documents where payload.user_id matches
    result = col.delete_many({"payload.user_id": user_id})
    client.close()

    return result.deleted_count


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

"""
Standalone Streamlit app with Pydantic AI agent and Mem0 memory.

A simple chat application that demonstrates:
- Pydantic AI agent for conversation
- Mem0 for persistent user memory across sessions
- Streamlit for the UI

Usage:
    streamlit run rag/app/streamlit/streamlit_mem0_app.py
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from rag.config.settings import load_settings
from rag.memory.mem0_store import create_mem0_store

st.set_page_config(
    page_title="Chat with Memory",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Chat with Memory")
st.caption("Powered by Pydantic AI + Mem0")


@st.cache_resource
def get_mem0_store():
    return create_mem0_store()


@st.cache_resource
def get_agent():
    settings = load_settings()
    provider = OpenAIProvider(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    model = OpenAIChatModel(settings.llm_model, provider=provider)
    system_prompt = """You are a helpful assistant with memory capabilities.
You remember information about users from previous conversations.
When given user context, use it to personalize your responses.
Be concise and friendly."""
    return Agent(model, system_prompt=system_prompt)


async def run_agent(agent: Agent, prompt: str) -> str:
    result = await agent.run(prompt)
    return result.output


with st.sidebar:
    st.header("Settings")
    user_id = st.text_input("User ID", value="default_user", help="Unique identifier for memory storage")
    st.divider()

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear All Memories"):
        try:
            get_mem0_store().delete_all(user_id)
            st.success(f"Cleared memories for {user_id}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    if st.button("Show My Memories"):
        try:
            memories = get_mem0_store().get_all(user_id=user_id)
            if memories:
                st.write(f"**{len(memories)} memories found:**")
                for mem in memories:
                    st.info(mem.get("memory", str(mem)))
            else:
                st.write("No memories stored yet.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    st.caption("Model: " + load_settings().llm_model)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's on your mind?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = get_agent()
    mem0_store = get_mem0_store()
    user_context = mem0_store.get_context_string(query=prompt, user_id=user_id)
    enhanced_prompt = f"{user_context}\n\nUser: {prompt}" if user_context else prompt

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = asyncio.run(run_agent(agent, enhanced_prompt))
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

    try:
        mem0_store.add(f"User said: {prompt}\nAssistant replied: {response}", user_id=user_id, infer=True)
    except Exception as e:
        st.warning(f"Could not save memory: {e}")

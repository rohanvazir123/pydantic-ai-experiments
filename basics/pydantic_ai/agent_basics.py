"""
Pydantic AI Agent Basics
========================
Three ways to run a Pydantic AI agent:

  1. agent.run()       — async, awaitable, returns when the agent finishes
  2. agent.run_sync()  — sync wrapper around run(), for scripts / notebooks
  3. agent.iter()      — async, node-by-node execution with event-level streaming

Run:
    python basics/pydantic_ai/agent_basics.py

FastAPI streaming endpoint at the bottom — start with:
    uvicorn basics.pydantic_ai.agent_basics:app --reload
    curl "http://localhost:8000/chat?prompt=What+is+the+weather+in+NYC"
"""

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
)

async def _aiter(stream):
    if hasattr(stream, "__aiter__"):
        async for event in stream:
            yield event
    else:
        for event in stream:
            yield event

# ---------------------------------------------------------------------------
# Agent + tool shared by all examples
# ---------------------------------------------------------------------------

agent: Agent[None, str] = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant. Answer concisely.",
)


@agent.tool_plain
async def get_weather(city: str) -> str:
    """Returns current weather for a city."""
    # Stub — replace with a real API call
    return f"The weather in {city} is 72°F and sunny."


@agent.tool_plain
async def get_time(timezone: str) -> str:
    """Returns current time in a given timezone."""
    return f"The current time in {timezone} is 14:32."


# ---------------------------------------------------------------------------
# 1. agent.run()  — async, blocks until the full response is ready
# ---------------------------------------------------------------------------

async def demo_run() -> None:
    """
    agent.run(prompt) awaits the complete agent turn, including all tool
    calls and the final text response.

    result.output     — the final string (or structured type if result_type is set)
    result.usage()    — token counts
    result.all_messages() — full message history (useful for multi-turn)
    """
    result = await agent.run("What is the weather in Tokyo and what time is it there?")

    print("=== agent.run() ===")
    print(result.output)
    print(f"Tokens used: {result.usage()}")


# ---------------------------------------------------------------------------
# 2. agent.run_sync()  — synchronous, for scripts and notebooks
# ---------------------------------------------------------------------------

def demo_run_sync() -> None:
    """
    agent.run_sync() is a thin wrapper that calls asyncio.run() internally.
    Use it when you are NOT inside an async context (plain scripts, pytest
    without pytest-asyncio, Jupyter cells that already have a running loop
    need asyncio.get_event_loop().run_until_complete() instead).

    The return value and attributes are identical to agent.run().
    """
    result = agent.run_sync("Summarise the weather in London in one sentence.")

    print("=== agent.run_sync() ===")
    print(result.output)


# ---------------------------------------------------------------------------
# 3. agent.iter()  — stream events node by node
# ---------------------------------------------------------------------------

async def demo_iter_streaming() -> None:
    """
    agent.iter() gives you fine-grained control over the agent execution loop.

    The agent alternates between two node types:
      ModelRequestNode  — the LLM is called; yields text/tool-call events
      CallToolsNode     — registered tools are executed; yields tool result events

    Calling node.stream(run.ctx) returns an async stream of typed events:
      PartStartEvent        — a new content part begins (text, tool call, …)
      PartDeltaEvent        — incremental update to the current part
        └── delta: TextPartDelta       — text streamed token by token
      FunctionToolCallEvent — the LLM has decided to call a tool
      FunctionToolResultEvent — the tool has returned a result
    """
    print("=== agent.iter() streaming ===")

    async with agent.iter("What is the weather in Paris and what time is it there?") as run:
        async for node in run:
            if not hasattr(node, 'stream'):
                continue
            async for event in node.stream(run.ctx):

                # --- Text streaming -----------------------------------
                if isinstance(event, PartStartEvent):
                    if event.part.part_kind == "text":
                        print(event.part.content, end="", flush=True)

                elif isinstance(event, PartDeltaEvent):
                    if isinstance(event.delta, TextPartDelta):
                        print(event.delta.content_delta, end="", flush=True)

                # --- Tool call ----------------------------------------
                elif isinstance(event, FunctionToolCallEvent):
                    print(
                        f"\n[tool call] {event.part.tool_name}"
                        f"({event.part.args})"
                    )

                # --- Tool result --------------------------------------
                elif isinstance(event, FunctionToolResultEvent):
                    print(f"[tool result] {event.result.content}")

    print()  # newline after streamed output


# ---------------------------------------------------------------------------
# 4. FastAPI SSE endpoint using agent.iter()
# ---------------------------------------------------------------------------

app = FastAPI()


async def _agent_event_generator(prompt: str) -> AsyncGenerator[str]:
    """
    Yields Server-Sent Events (SSE) as the agent runs.

    Event types emitted:
      {"type": "text",        "content": "..."}   — text token
      {"type": "tool_call",   "tool": "...", "args": "..."}
      {"type": "tool_result", "output": "..."}
      {"type": "done"}

    SSE format requires each message to be "data: <payload>\n\n".
    """
    async with agent.iter(prompt) as run:
        async for node in run:
            if not hasattr(node, 'stream'):
                continue
            async for event in node.stream(run.ctx):

                    # Text tokens
                    if isinstance(event, PartStartEvent):
                        if event.part.part_kind == "text" and event.part.content:
                            yield f"data: {json.dumps({'type': 'text', 'content': event.part.content})}\n\n"

                    elif isinstance(event, PartDeltaEvent):
                        if isinstance(event.delta, TextPartDelta) and event.delta.content_delta:
                            yield f"data: {json.dumps({'type': 'text', 'content': event.delta.content_delta})}\n\n"

                    # Tool call initialised by the LLM
                    elif isinstance(event, FunctionToolCallEvent):
                        yield f"data: {json.dumps({
                            'type': 'tool_call',
                            'tool': event.part.tool_name,
                            'args': event.part.args,
                        })}\n\n"

                    # Tool has returned its result
                    elif isinstance(event, FunctionToolResultEvent):
                        yield f"data: {json.dumps({
                            'type': 'tool_result',
                            'output': str(event.result.content),
                        })}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.get("/chat")
async def chat(prompt: str) -> StreamingResponse:
    """
    GET /chat?prompt=<your question>

    Returns a text/event-stream response. Each line is a JSON SSE event.

    Example (curl):
        curl "http://localhost:8000/chat?prompt=What+is+the+weather+in+NYC"
    """
    return StreamingResponse(
        _agent_event_generator(prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def main() -> None:
    await demo_run()
    print()
    demo_run_sync()
    print()
    await demo_iter_streaming()


if __name__ == "__main__":
    asyncio.run(main())

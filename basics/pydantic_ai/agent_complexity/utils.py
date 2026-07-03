"""Shared helpers for printing an agent's step-by-step trace.

Levels 3-5 let the model decide which tools to call, so seeing the tool-call /
tool-return sequence is the whole point. :func:`print_agent_trace` walks the
message history a run produced and prints it in reading order.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)


def print_agent_trace(result: Any) -> None:
    """Print the tool calls, tool returns, and final text of an agent run.

    Args:
        result: The object returned by ``agent.run()`` / ``agent.run_sync()``.
            Any object exposing ``all_messages()`` works.
    """
    print("\n" + "=" * 60)
    print("AGENT TRACE")
    print("=" * 60)

    step = 0
    for message in result.all_messages():
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    step += 1
                    print(f"\n[Step {step}] Tool call: {part.tool_name}")
                    print(f"         Args: {part.args}")
                elif isinstance(part, TextPart):
                    step += 1
                    print(f"\n[Step {step}] Final response")
                    print(f"         {part.content[:200]}")
        elif isinstance(message, ModelRequest):
            for req_part in message.parts:
                if isinstance(req_part, ToolReturnPart):
                    print(
                        f"         <- {req_part.tool_name} returned: "
                        f"{str(req_part.content)[:150]}"
                    )

    print("\n" + "=" * 60)

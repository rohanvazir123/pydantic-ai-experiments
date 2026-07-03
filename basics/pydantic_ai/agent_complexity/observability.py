"""
Opt-in observability for the examples via Logfire.

Pydantic AI has first-class Logfire integration: one call instruments every
agent run, model request, and tool call as a span — invaluable once you're past
Level 3 and a single run fans out into many tool calls or sub-agents.

This is a **no-op unless** ``AGENT_LOGFIRE=1`` is set, so default runs and the
test suite are unaffected. It uses ``send_to_logfire="if-token-present"``: spans
print to the console locally, and *also* stream to the Logfire web UI only if
you've authenticated (``logfire auth``) or set ``LOGFIRE_TOKEN`` — so nothing
leaves your machine unless you opt in.

Usage:

    AGENT_LOGFIRE=1 python l5_multi_agent.py          # console spans, no signup
    logfire auth && AGENT_LOGFIRE=1 python l4_agent_harness.py  # + web UI

or in code:

    import observability
    observability.enable_logfire()               # honors AGENT_LOGFIRE
"""

from __future__ import annotations

import os


def enable_logfire(service_name: str = "agent-complexity") -> bool:
    """Enable Logfire tracing for Pydantic AI if ``AGENT_LOGFIRE=1``.

    Returns True if instrumentation was enabled, False otherwise (env not set,
    or Logfire not installed). Safe to call unconditionally at the top of a
    script's ``main()``.
    """
    if os.getenv("AGENT_LOGFIRE") != "1":
        return False
    try:
        import logfire
    except ImportError:
        print("[observability] AGENT_LOGFIRE=1 but logfire is not installed.")
        return False

    # Console spans always; stream to the Logfire UI only if a token is present
    # (via `logfire auth` or LOGFIRE_TOKEN) — nothing leaves the machine otherwise.
    logfire.configure(service_name=service_name, send_to_logfire="if-token-present")
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)  # exact payloads to/from Ollama
    print("[observability] Logfire tracing enabled.")
    return True

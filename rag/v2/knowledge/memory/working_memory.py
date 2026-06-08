"""Tier 1 working memory — context assembly and token-budget trimming.

Assembles the LLM context from all five memory tiers:
  Tier 5: system prompt (procedural)
  Tier 3: user memory snippets (semantic/user)
  Tier 2: conversation history (episodic)
  Tier 4: retrieved chunks (semantic/world)
  query:  current user query

Trim order when budget exceeded (lowest priority dropped first):
  1. Drop lowest-confidence retrieved chunks
  2. Replace oldest turns with summary + last 4 turns
  3. Reduce user memories to top-1
  4. Emit context_truncated: True — never silent
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AssembledContext:
    system_prompt:      str
    user_memory_prefix: str
    history_text:       str
    chunks_text:        str
    query:              str
    context_truncated:  bool      = False
    token_count:        int       = 0
    metadata:           dict[str, Any] = field(default_factory=dict)

    def as_message_history(self) -> list[dict[str, str]]:
        """Format history_text as a list of message dicts for agent.run()."""
        if not self.history_text:
            return []
        messages = []
        for line in self.history_text.split("\n"):
            if line.startswith("User: "):
                messages.append({"role": "user", "content": line[6:]})
            elif line.startswith("Assistant: "):
                messages.append({"role": "assistant", "content": line[11:]})
        return messages


def _rough_token_count(text: str) -> int:
    """Rough token estimate: ~4 chars per token (GPT-3 rule of thumb)."""
    return max(1, len(text) // 4)


def count_tokens(parts: list[str]) -> int:
    return sum(_rough_token_count(p) for p in parts)


def format_history(messages: list[dict[str, Any]]) -> str:
    lines = []
    for m in messages:
        role    = m.get("role", "user").capitalize()
        content = m.get("content", "")[:500]
        if role == "System":
            lines.append(f"[{content}]")
        else:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def format_chunks(results: list[Any]) -> str:
    """Format SearchResult list as LLM-readable context with [chunk_id] anchors."""
    if not results:
        return ""
    lines = []
    for r in results:
        chunk_id = getattr(r, "chunk_id", "?")
        title    = getattr(r, "document_title", "")
        content  = getattr(r, "content", "")[:600]
        lines.append(f"[chunk_id: {chunk_id}] {title}\n{content}")
    return "\n\n".join(lines)


def assemble(
    system_prompt:      str,
    user_memories:      list[str],
    history_messages:   list[dict[str, Any]],
    retrieved_chunks:   list[Any],
    query:              str,
    budget:             int = 8192,
) -> AssembledContext:
    """Assemble working memory from all tiers and trim to budget."""

    # Build initial components
    user_prefix  = ("User context:\n" + "\n".join(f"- {m}" for m in user_memories)) if user_memories else ""
    history_text = format_history(history_messages)
    chunks_text  = format_chunks(retrieved_chunks)
    chunks       = list(retrieved_chunks)

    context_truncated = False

    # ── Trim loop ─────────────────────────────────────────────────────────────

    while True:
        total = count_tokens([system_prompt, user_prefix, history_text, chunks_text, query])
        if total <= budget:
            break

        # 1. Drop lowest-confidence chunk
        if chunks:
            chunks.sort(key=lambda r: getattr(r, "confidence", 0.0) or 0.0)
            chunks.pop(0)
            chunks_text = format_chunks(chunks)
            context_truncated = True
            continue

        # 2. Shrink history to last 4 turns
        if len(history_messages) > 4:
            history_messages = history_messages[-4:]
            history_text = format_history(history_messages)
            context_truncated = True
            continue

        # 3. Reduce user memories to top-1
        if len(user_memories) > 1:
            user_memories    = user_memories[:1]
            user_prefix      = "User context:\n- " + user_memories[0]
            context_truncated = True
            continue

        # 4. Can't trim further — emit warning
        logger.warning(
            "Context budget %d exceeded after all trimming (total=%d); proceeding",
            budget, total,
        )
        break

    return AssembledContext(
        system_prompt=system_prompt,
        user_memory_prefix=user_prefix,
        history_text=history_text,
        chunks_text=chunks_text,
        query=query,
        context_truncated=context_truncated,
        token_count=count_tokens([system_prompt, user_prefix, history_text, chunks_text, query]),
    )

"""Conversation auto-summarizer — nano model, non-blocking background task.

Triggered when turn_count == SUMMARIZE_THRESHOLD (20).
Summarizes all turns except the last ACTIVE_WINDOW (8).
Stores result in conversations.summary.
"""

import logging
from typing import TYPE_CHECKING

from knowledge.agent.prompts import SUMMARIZER_PROMPT

if TYPE_CHECKING:
    from knowledge.memory.conversation_store import ConversationStore

logger = logging.getLogger(__name__)

KEEP_LAST = 8   # turns kept verbatim; everything else gets summarized


async def summarize_conversation(
    conversation_id: str,
    store: "ConversationStore",
) -> None:
    """Summarize a long conversation and store the result.

    Called as asyncio.create_task() — never blocks the response path.
    """
    assert store._pool
    async with store._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT role, content FROM messages
            WHERE conversation_id=$1
            ORDER BY created_at ASC
            """,
            conversation_id,
        )

    if len(rows) <= KEEP_LAST:
        return

    # Summarize all turns except the most recent KEEP_LAST
    to_summarize = rows[:-KEEP_LAST]
    transcript   = "\n".join(
        f"{r['role'].capitalize()}: {r['content'][:300]}"
        for r in to_summarize
    )

    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        from knowledge.config.settings import load_settings

        s        = load_settings()
        provider = OpenAIProvider(base_url=s.llm_base_url, api_key=s.llm_api_key)
        model    = OpenAIChatModel(s.model_tier_nano, provider=provider)
        agent    = Agent(model, system_prompt=SUMMARIZER_PROMPT)

        result  = await agent.run(transcript)
        summary = str(result.output).strip()

        await store.store_summary(conversation_id, summary)
        logger.info("Summarized conversation %s (%d turns)", conversation_id, len(rows))

    except Exception as exc:
        logger.warning("Conversation summarization failed for %s: %s", conversation_id, exc)

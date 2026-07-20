# Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
# See LICENSE file in the project root for details.

"""Tier 2 episodic memory — conversation and message CRUD.

Server-side conversation history: client sends only session_id;
server loads history from this store. Replaces client-side message_history.

Active window policy:
  turn_count <= 20 → return last 20 messages verbatim
  turn_count > 20  → return [SummaryMessage] + last 8 messages
"""

import logging
import uuid as _uuid
from typing import Any

import asyncpg

from knowledge.config.settings import Settings, load_settings

logger = logging.getLogger(__name__)

SUMMARIZE_THRESHOLD = 20   # trigger summarization at this turn count
ACTIVE_WINDOW       = 8    # messages kept after summary


class ConversationStore:
    """asyncpg-backed episodic memory store."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or load_settings()
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._settings.database_url,
            min_size=2,
            max_size=10,
            command_timeout=self._settings.db_query_timeout_s,
        )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    # ── Conversations ─────────────────────────────────────────────────────────

    async def get_or_create_conversation(
        self,
        session_id: str,
        tenant_id: str,
        user_id: str,
        corpus_ids: list[str],
    ) -> dict[str, Any]:
        assert self._pool
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM conversations WHERE session_id=$1 AND tenant_id=$2",
                session_id, tenant_id,
            )
            if row:
                return dict(row)
            conv_id = str(_uuid.uuid4())
            row = await conn.fetchrow(
                """
                INSERT INTO conversations (id, session_id, tenant_id, user_id, corpus_ids)
                VALUES ($1,$2,$3,$4,$5) RETURNING *
                """,
                conv_id, session_id, tenant_id, user_id, corpus_ids,
            )
            return dict(row)

    async def list_conversations(
        self,
        user_id: str,
        tenant_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        assert self._pool
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, session_id, title, summary, turn_count, last_turn_at
                FROM conversations
                WHERE user_id=$1 AND tenant_id=$2 AND deleted_at IS NULL
                ORDER BY last_turn_at DESC LIMIT $3
                """,
                user_id, tenant_id, limit,
            )
        return [dict(r) for r in rows]

    async def get_conversation(self, conv_id: str, user_id: str) -> dict[str, Any] | None:
        assert self._pool
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM conversations WHERE id=$1 AND user_id=$2 AND deleted_at IS NULL",
                conv_id, user_id,
            )
        return dict(row) if row else None

    async def delete_conversation(self, conv_id: str, user_id: str) -> None:
        assert self._pool
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE conversations SET deleted_at=NOW() WHERE id=$1 AND user_id=$2",
                conv_id, user_id,
            )

    # ── Messages ──────────────────────────────────────────────────────────────

    async def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        **metadata: Any,
    ) -> str:
        assert self._pool
        import json
        msg_id = str(_uuid.uuid4())
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO messages
                  (id, conversation_id, role, content,
                   citations, pipeline_status, confidence, model_tier,
                   prompt_tokens, completion_tokens, cost_usd, cache_hit, request_id)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                """,
                msg_id, conversation_id, role, content,
                json.dumps(metadata.get("citations")) if metadata.get("citations") else None,
                metadata.get("pipeline_status"),
                metadata.get("confidence"),
                metadata.get("model_tier"),
                metadata.get("prompt_tokens"),
                metadata.get("completion_tokens"),
                metadata.get("cost_usd"),
                metadata.get("cache_hit"),
                metadata.get("request_id"),
            )
            # Increment turn count and update last_turn_at
            row = await conn.fetchrow(
                """
                UPDATE conversations
                SET turn_count = turn_count + 1,
                    last_turn_at = NOW(),
                    title = COALESCE(title, LEFT($2, 60))
                WHERE id = $1
                RETURNING turn_count
                """,
                conversation_id, content,
            )
            return_turn = row["turn_count"] if row else 0

        # Trigger summarization when threshold crossed (async, non-blocking)
        if return_turn == SUMMARIZE_THRESHOLD:
            import asyncio

            from knowledge.memory.summarizer import summarize_conversation
            asyncio.create_task(summarize_conversation(conversation_id, self))

        return msg_id

    async def load_active_window(self, session_id: str, tenant_id: str) -> list[dict[str, Any]]:
        """Load the active message window for a session.

        Returns last 20 messages if turn_count <= 20,
        or [summary_placeholder] + last 8 if turn_count > 20.
        """
        assert self._pool
        async with self._pool.acquire() as conn:
            conv = await conn.fetchrow(
                "SELECT id, turn_count, summary FROM conversations WHERE session_id=$1 AND tenant_id=$2",
                session_id, tenant_id,
            )
            if conv is None:
                return []

            conv_id    = conv["id"]
            turn_count = conv["turn_count"]
            summary    = conv["summary"]

            limit = ACTIVE_WINDOW if turn_count > SUMMARIZE_THRESHOLD else 20
            rows  = await conn.fetch(
                """
                SELECT id, role, content, pipeline_status, created_at
                FROM messages
                WHERE conversation_id=$1
                ORDER BY created_at DESC LIMIT $2
                """,
                conv_id, limit,
            )

        messages = [dict(r) for r in reversed(rows)]

        if turn_count > SUMMARIZE_THRESHOLD and summary:
            messages.insert(0, {
                "role":    "system",
                "content": f"[Conversation summary: {summary}]",
                "id":      "summary",
            })

        return messages

    async def store_summary(self, conversation_id: str, summary: str) -> None:
        assert self._pool
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE conversations SET summary=$1 WHERE id=$2",
                summary, conversation_id,
            )

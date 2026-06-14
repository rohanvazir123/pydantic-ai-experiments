"""Background pruning jobs for Tier 2 (episodic) and Tier 3 (semantic/user) memory.

All jobs run via APScheduler in the FastAPI lifespan.
All are idempotent — safe to run multiple times.
"""

import logging

import asyncpg

from knowledge.config.settings import Settings, load_settings

logger = logging.getLogger(__name__)

_USER_MEMORY_CAP       = 200   # hard cap per user per tenant
_LRU_DAYS              = 60    # evict memories not retrieved in 60 days
_LRU_MIN_AGE_DAYS      = 90    # only evict if also older than 90 days
_LRU_EVICT_BATCH       = 20    # evict this many at a time
_CONV_GRACE_DAYS       = 7     # hard-delete soft-deleted conversations after 7 days


async def prune_expired_conversations(pool: asyncpg.Pool) -> int:
    """Hard-delete conversations that were soft-deleted more than 7 days ago."""
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM conversations
            WHERE deleted_at < NOW() - INTERVAL '7 days'
            """
        )
    deleted = int(result.split()[-1])
    if deleted:
        logger.info("Pruned %d soft-deleted conversations", deleted)
    return deleted


async def expire_old_conversations(pool: asyncpg.Pool, retention_days: int = 90) -> int:
    """Soft-delete conversations older than retention_days (configurable per tenant)."""
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE conversations
            SET deleted_at = NOW()
            WHERE expires_at < NOW()
              AND deleted_at IS NULL
            """
        )
    updated = int(result.split()[-1])
    if updated:
        logger.info("Soft-deleted %d expired conversations", updated)
    return updated


async def prune_user_memories_lru(pool: asyncpg.Pool) -> int:
    """Evict memories per user/tenant that exceed the hard cap using LRU.

    Only evicts memories that:
    - Have not been retrieved in LRU_DAYS days, AND
    - Were created more than LRU_MIN_AGE_DAYS days ago
    """
    async with pool.acquire() as conn:
        # Find user/tenant pairs that are over the cap
        over_cap = await conn.fetch(
            """
            SELECT user_id, tenant_id, COUNT(*) AS cnt
            FROM user_memories
            GROUP BY user_id, tenant_id
            HAVING COUNT(*) > $1
            """,
            _USER_MEMORY_CAP,
        )

        total_deleted = 0
        for row in over_cap:
            result = await conn.execute(
                """
                DELETE FROM user_memories
                WHERE user_id = $1 AND tenant_id = $2
                  AND (last_retrieved_at < NOW() - INTERVAL '60 days'
                       OR last_retrieved_at IS NULL)
                  AND created_at < NOW() - INTERVAL '90 days'
                ORDER BY COALESCE(last_retrieved_at, created_at) ASC
                LIMIT $3
                """,
                row["user_id"], row["tenant_id"], _LRU_EVICT_BATCH,
            )
            n = int(result.split()[-1])
            total_deleted += n

    if total_deleted:
        logger.info("LRU evicted %d user memories", total_deleted)
    return total_deleted


async def prune_semantic_cache(pool: asyncpg.Pool) -> int:
    """Delete expired semantic_cache rows (expires_at < NOW())."""
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM semantic_cache WHERE expires_at < NOW()"
        )
    deleted = int(result.split()[-1])
    if deleted:
        logger.info("Pruned %d expired semantic cache entries", deleted)
    return deleted


async def run_all_pruning_jobs(settings: Settings | None = None) -> dict[str, int]:
    """Run all pruning jobs. Called by APScheduler (nightly)."""
    _settings = settings or load_settings()
    pool = await asyncpg.create_pool(
        _settings.database_url, min_size=1, max_size=3,
        command_timeout=_settings.db_query_timeout_s,
    )
    try:
        results = {
            "expired_conversations_soft_deleted": await expire_old_conversations(pool),
            "conversations_hard_deleted":         await prune_expired_conversations(pool),
            "user_memories_evicted":              await prune_user_memories_lru(pool),
            "semantic_cache_entries_pruned":      await prune_semantic_cache(pool),
        }
        logger.info("Nightly pruning complete: %s", results)
        return results
    finally:
        await pool.close()

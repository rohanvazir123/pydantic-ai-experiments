-- purge_default_corpus.sql
--
-- Deletes all ingested content for the default tenant + neuralflow corpus.
-- Schema, migrations, tenant config, evaluation data, and memory are preserved.
-- Run this before a forced full re-ingestion.
--
-- What is purged:
--   documents          → all rows for tenant_id='default' AND corpus_id='neuralflow'
--   chunks             → cascade from documents (ON DELETE CASCADE)
--   kg_entity_index    → entity shadow table for AGE hybrid search
--   semantic_cache     → L3 answer cache (now invalid after content change)
--
-- What is NOT purged (intentionally preserved):
--   audit_events       → append-only compliance log
--   token_usage        → financial records (7yr retention)
--   billing_events     → financial records
--   conversations      → user episodic memory (Tier 2)
--   messages           → cascade from conversations
--   user_memories      → user semantic memory (Tier 3)
--   system_prompts     → procedural memory (Tier 5)
--   tenants            → tenant config
--   tenant_quotas      → quota config
--   gold_samples       → evaluation gold dataset
--   eval_runs          → evaluation history
--   eval_results       → evaluation results
--   scheduled_jobs     → scheduler config
--
-- Note: Redis fingerprint cache (cache:doc_fingerprint:*) must also be cleared
-- so the incremental check does not skip re-ingestion of existing files.
-- The purge-corpus Makefile target handles this automatically.
--
-- Note: Apache AGE graph for this corpus is dropped separately via the
-- purge.py script (AGE uses a separate PostgreSQL connection on port 5433).

BEGIN;

-- 1. Delete documents (chunks cascade automatically via FK ON DELETE CASCADE)
DELETE FROM documents
WHERE tenant_id = 'default'
  AND corpus_id = 'neuralflow';

-- 2. Clear entity shadow index (AGE entities — no FK, must delete explicitly)
DELETE FROM kg_entity_index
WHERE tenant_id = 'default'
  AND corpus_id = 'neuralflow';

-- 3. Invalidate L3 semantic cache entries that covered this corpus
DELETE FROM semantic_cache
WHERE tenant_id = 'default'
  AND 'neuralflow' = ANY(corpus_ids);

COMMIT;

-- Report
SELECT
    (SELECT COUNT(*) FROM documents    WHERE tenant_id='default' AND corpus_id='neuralflow') AS remaining_documents,
    (SELECT COUNT(*) FROM chunks       WHERE tenant_id='default' AND corpus_id='neuralflow') AS remaining_chunks,
    (SELECT COUNT(*) FROM kg_entity_index WHERE tenant_id='default' AND corpus_id='neuralflow') AS remaining_entities,
    (SELECT COUNT(*) FROM semantic_cache  WHERE tenant_id='default' AND 'neuralflow'=ANY(corpus_ids)) AS remaining_cache_entries;

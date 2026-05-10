-- =============================================================
-- 01_verify_tables.sql
-- Run in DBeaver connected to: localhost:5434 / rag_db / meeting_analytics
-- Verifies all 9 Final Version tables exist and have expected row counts.
-- =============================================================

-- ── Row counts for all 9 tables ──────────────────────────────
SELECT 'meetings'                AS table_name, count(*) AS rows FROM meeting_analytics.meetings
UNION ALL SELECT 'meeting_participants',         count(*) FROM meeting_analytics.meeting_participants
UNION ALL SELECT 'meeting_summaries',            count(*) FROM meeting_analytics.meeting_summaries
UNION ALL SELECT 'key_moments',                  count(*) FROM meeting_analytics.key_moments
UNION ALL SELECT 'action_items',                 count(*) FROM meeting_analytics.action_items
UNION ALL SELECT 'transcript_lines',             count(*) FROM meeting_analytics.transcript_lines
UNION ALL SELECT 'semantic_clusters',            count(*) FROM meeting_analytics.semantic_clusters
UNION ALL SELECT 'semantic_phrases',             count(*) FROM meeting_analytics.semantic_phrases
UNION ALL SELECT 'semantic_meeting_themes',      count(*) FROM meeting_analytics.semantic_meeting_themes
ORDER BY table_name;

-- Expected:
--   action_items               397
--   key_moments                402
--   meeting_participants       311
--   meeting_summaries          100
--   meetings                   100
--   semantic_clusters           26
--   semantic_meeting_themes    516
--   semantic_phrases           343
--   transcript_lines          4313


-- ── Spot check: every meeting has exactly one primary semantic theme ──
SELECT count(*) AS meetings_with_primary
FROM meeting_analytics.semantic_meeting_themes
WHERE is_primary = true;
-- Expected: 100


-- ── Spot check: action_items_by_theme view ────────────────────
SELECT count(*) AS action_items_with_theme
FROM meeting_analytics.action_items_by_theme;
-- Expected: 397


-- ── Spot check: Final Version clusters ───────────────────────
SELECT cluster_id, theme_title, audience, phrase_count
FROM meeting_analytics.semantic_clusters
ORDER BY cluster_id;


-- ── Spot check: call type distribution (LLM-generated, 3 values) ─
SELECT call_type, count(*) AS meetings
FROM meeting_analytics.semantic_meeting_themes
WHERE is_primary = true
GROUP BY call_type
ORDER BY meetings DESC;


-- ── Spot check: product coverage ─────────────────────────────
SELECT unnest(products) AS product, count(*) AS meetings
FROM meeting_analytics.meeting_summaries
GROUP BY product
ORDER BY meetings DESC;
-- Expected: Comply 59, Detect 59, Protect 24, Identity 23  (8 untagged)

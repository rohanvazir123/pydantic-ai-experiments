-- =============================================================
-- 01_verify_tables.sql
-- Run in DBeaver connected to: localhost:5434 / rag_db / meeting_analytics
-- Verifies all 16 tables exist and have expected row counts.
-- =============================================================

-- ── Row counts for all 16 tables ─────────────────────────────
SELECT 'meetings'                AS table_name, count(*) AS rows FROM meeting_analytics.meetings
UNION ALL SELECT 'meeting_participants',         count(*) FROM meeting_analytics.meeting_participants
UNION ALL SELECT 'meeting_summaries',            count(*) FROM meeting_analytics.meeting_summaries
UNION ALL SELECT 'summary_topics',               count(*) FROM meeting_analytics.summary_topics
UNION ALL SELECT 'action_items',                 count(*) FROM meeting_analytics.action_items
UNION ALL SELECT 'key_moments',                  count(*) FROM meeting_analytics.key_moments
UNION ALL SELECT 'transcript_lines',             count(*) FROM meeting_analytics.transcript_lines
UNION ALL SELECT 'meeting_themes',               count(*) FROM meeting_analytics.meeting_themes
UNION ALL SELECT 'call_types',                   count(*) FROM meeting_analytics.call_types
UNION ALL SELECT 'sentiment_features',           count(*) FROM meeting_analytics.sentiment_features
UNION ALL SELECT 'kmeans_clusters',              count(*) FROM meeting_analytics.kmeans_clusters
UNION ALL SELECT 'kmeans_cluster_terms',         count(*) FROM meeting_analytics.kmeans_cluster_terms
UNION ALL SELECT 'kmeans_meeting_clusters',      count(*) FROM meeting_analytics.kmeans_meeting_clusters
UNION ALL SELECT 'semantic_clusters',            count(*) FROM meeting_analytics.semantic_clusters
UNION ALL SELECT 'semantic_phrases',             count(*) FROM meeting_analytics.semantic_phrases
UNION ALL SELECT 'semantic_meeting_themes',      count(*) FROM meeting_analytics.semantic_meeting_themes
ORDER BY table_name;

-- Expected:
--   meetings                  100
--   meeting_participants      622
--   meeting_summaries         100
--   summary_topics            600
--   action_items              397
--   key_moments               402
--   transcript_lines         4313
--   meeting_themes            466
--   call_types                100
--   sentiment_features        100
--   kmeans_clusters             8
--   kmeans_cluster_terms       96
--   kmeans_meeting_clusters   100
--   semantic_clusters          26
--   semantic_phrases          343
--   semantic_meeting_themes   516


-- ── Spot check: every meeting has exactly one primary theme (Take A) ─
SELECT count(*) AS meetings_with_primary
FROM meeting_analytics.meeting_themes
WHERE is_primary = true;
-- Expected: 100


-- ── Spot check: every meeting assigned to exactly one KMeans cluster (Take B) ─
SELECT count(DISTINCT meeting_id) AS meetings_in_kmeans
FROM meeting_analytics.kmeans_meeting_clusters;
-- Expected: 100


-- ── Spot check: every meeting has exactly one primary semantic theme (Take C) ─
SELECT count(*) AS meetings_with_primary
FROM meeting_analytics.semantic_meeting_themes
WHERE is_primary = true;
-- Expected: 100


-- ── Spot check: Take A themes — distinct values ─
SELECT theme, count(*) AS meetings,
       sum(CASE WHEN is_primary THEN 1 ELSE 0 END) AS primary_count
FROM meeting_analytics.meeting_themes
GROUP BY theme
ORDER BY meetings DESC;


-- ── Spot check: Take A call types — distinct values ─
SELECT call_type, count(*) AS meetings
FROM meeting_analytics.call_types
GROUP BY call_type
ORDER BY meetings DESC;


-- ── Spot check: Take B clusters ─
SELECT cluster_id, label, meeting_count,
       round(silhouette_score, 4) AS silhouette
FROM meeting_analytics.kmeans_clusters
ORDER BY cluster_id;


-- ── Spot check: Take C clusters ─
SELECT cluster_id, theme_title, audience, phrase_count
FROM meeting_analytics.semantic_clusters
ORDER BY cluster_id;


-- ── Spot check: Take C call types (LLM-generated — 3 values) ─
SELECT call_type, count(*) AS meetings
FROM meeting_analytics.semantic_meeting_themes
WHERE is_primary = true
GROUP BY call_type
ORDER BY meetings DESC;

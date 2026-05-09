-- =============================================================
-- 02_insight_queries.sql
-- Run in DBeaver connected to: localhost:5434 / rag_db / meeting_analytics
-- 10 stakeholder insight queries — all use Take A tables.
-- Take C equivalents noted where relevant.
-- =============================================================


-- ── INSIGHT 1: Theme volume and evidence strength ─────────────
-- Question: Which themes are most prevalent, and how confidently assigned?
-- Audience: Leadership — context-setting baseline for every other chart.
-- Note: total_meetings = how many meetings touch this theme (multi-label).
--       primary_assignments = how many have it as their dominant theme.
--       avg_evidence_score = keyword match density (higher = more certain).

SELECT theme,
       count(*)                                                AS total_meetings,
       sum(CASE WHEN is_primary THEN 1 ELSE 0 END)            AS primary_assignments,
       round(avg(evidence_count)::numeric, 1)                  AS avg_evidence_score
FROM meeting_analytics.meeting_themes
GROUP BY theme
ORDER BY total_meetings DESC;


-- ── INSIGHT 2: Theme × sentiment heatmap ─────────────────────
-- Question: Which themes correlate with negative sentiment?
-- Audience: Support leaders, Engineering leads.
-- Note: Use primary theme only to avoid double-counting.
--       Pivot this in Excel/Python for the heatmap chart.

SELECT mt.theme,
       ms.overall_sentiment,
       count(*)                                                AS meetings
FROM meeting_analytics.meeting_themes mt
JOIN meeting_analytics.meeting_summaries ms ON mt.meeting_id = ms.meeting_id
WHERE mt.is_primary = true
GROUP BY mt.theme, ms.overall_sentiment
ORDER BY mt.theme, meetings DESC;


-- ── INSIGHT 3: Net sentiment by theme (transcript-grounded) ──
-- Question: What does actual conversation tone look like per theme?
-- Audience: Support / CX leadership — cross-check against AI summary scores.
-- Note: net_sentiment = positive_ratio - negative_ratio.
--       Range: -1.0 (all negative) to +1.0 (all positive).
--       Independent of dataset provider's sentimentScore.

SELECT mt.theme,
       count(DISTINCT mt.meeting_id)                          AS meetings,
       round(avg(sf.net_sentiment)::numeric, 3)               AS avg_net_sentiment,
       round(avg(sf.positive_ratio)::numeric, 3)              AS avg_positive_ratio,
       round(avg(sf.negative_ratio)::numeric, 3)              AS avg_negative_ratio
FROM meeting_analytics.meeting_themes mt
JOIN meeting_analytics.sentiment_features sf ON mt.meeting_id = sf.meeting_id
WHERE mt.is_primary = true
GROUP BY mt.theme
ORDER BY avg_net_sentiment ASC;


-- ── INSIGHT 4: Churn signal density by theme ─────────────────
-- Question: Where is churn risk concentrated?
-- Audience: Sales leadership, CSMs.
-- Finding: Reliability generates MORE churn signals per meeting (1.04)
--          than the Customer Retention theme itself (0.71). Outage calls
--          are commercially more dangerous than explicit renewal discussions.

SELECT mt.theme,
       count(DISTINCT mt.meeting_id)                          AS meetings,
       sum(sf.churn_signal_count)                             AS total_churn_signals,
       round(avg(sf.churn_signal_count)::numeric, 2)          AS churn_per_meeting,
       sum(sf.concern_count)                                  AS total_concerns
FROM meeting_analytics.meeting_themes mt
JOIN meeting_analytics.sentiment_features sf ON mt.meeting_id = sf.meeting_id
WHERE mt.is_primary = true
GROUP BY mt.theme
ORDER BY churn_per_meeting DESC;


-- ── INSIGHT 5: Call type distribution ────────────────────────
-- Question: What proportion of meetings are customer-facing vs internal?
-- Audience: Operations, leadership — capacity and resourcing baseline.

SELECT call_type,
       count(*)                                               AS meetings,
       round(100.0 * count(*) / sum(count(*)) OVER (), 1)    AS pct,
       round(avg(confidence)::numeric, 2)                     AS avg_confidence
FROM meeting_analytics.call_types
GROUP BY call_type
ORDER BY meetings DESC;


-- ── INSIGHT 6: Call type × theme matrix ──────────────────────
-- Question: Which themes dominate each call type?
-- Audience: Support leads (what support calls are really about),
--           Sales leads (what renewal calls are threatened by).

SELECT ct.call_type,
       mt.theme,
       count(*)                                               AS meetings
FROM meeting_analytics.call_types ct
JOIN meeting_analytics.meeting_themes mt ON ct.meeting_id = mt.meeting_id
WHERE mt.is_primary = true
GROUP BY ct.call_type, mt.theme
ORDER BY ct.call_type, meetings DESC;


-- ── INSIGHT 7: High-risk meeting watchlist ───────────────────
-- Question: Which specific meetings need immediate CSM attention?
-- Audience: CSMs, Account Executives — named accounts, actionable.
-- Filter: churn_signal >= 1 AND sentiment negative/mixed-negative.

SELECT m.meeting_id,
       m.title,
       ct.call_type,
       mt.theme                                               AS primary_theme,
       ms.overall_sentiment,
       sf.churn_signal_count,
       round(sf.net_sentiment::numeric, 3)                    AS net_sentiment,
       ms.sentiment_score
FROM meeting_analytics.meetings m
JOIN meeting_analytics.call_types ct         ON m.meeting_id = ct.meeting_id
JOIN meeting_analytics.meeting_themes mt     ON m.meeting_id = mt.meeting_id AND mt.is_primary = true
JOIN meeting_analytics.meeting_summaries ms  ON m.meeting_id = ms.meeting_id
JOIN meeting_analytics.sentiment_features sf ON m.meeting_id = sf.meeting_id
WHERE sf.churn_signal_count >= 1
  AND ms.overall_sentiment IN ('negative', 'very-negative', 'mixed-negative')
ORDER BY sf.churn_signal_count DESC, sf.net_sentiment ASC;


-- ── INSIGHT 8: Reliability-to-commercial bleed ───────────────
-- Question: How many meetings span both Reliability and Commercial Risk?
-- Audience: Revenue leadership — quantifies cost of reliability failures.
-- Finding: Outage conversations routinely appear in renewal discussions.

SELECT m.meeting_id,
       m.title,
       ct.call_type,
       ms.overall_sentiment,
       sf.churn_signal_count,
       round(sf.net_sentiment::numeric, 3)                    AS net_sentiment
FROM meeting_analytics.meetings m
JOIN meeting_analytics.call_types ct         ON m.meeting_id = ct.meeting_id
JOIN meeting_analytics.meeting_summaries ms  ON m.meeting_id = ms.meeting_id
JOIN meeting_analytics.sentiment_features sf ON m.meeting_id = sf.meeting_id
WHERE m.meeting_id IN (
    SELECT meeting_id FROM meeting_analytics.meeting_themes
    WHERE theme = 'Reliability / Incidents / Outages'
)
AND m.meeting_id IN (
    SELECT meeting_id FROM meeting_analytics.meeting_themes
    WHERE theme = 'Customer Retention / Renewal / Commercial Risk'
)
ORDER BY sf.churn_signal_count DESC, sf.net_sentiment ASC;


-- ── INSIGHT 9: Feature gap prioritisation by theme ───────────
-- Question: Which themes generate the most product gap signals,
--           and is customer sentiment around those gaps positive or negative?
-- Audience: Product managers — gaps under duress vs constructive wishlist.

SELECT mt.theme,
       count(DISTINCT mt.meeting_id)                          AS meetings_with_gaps,
       sum(sf.feature_gap_count)                              AS total_feature_gaps,
       round(avg(sf.net_sentiment)::numeric, 3)               AS avg_net_sentiment
FROM meeting_analytics.meeting_themes mt
JOIN meeting_analytics.sentiment_features sf ON mt.meeting_id = sf.meeting_id
WHERE mt.is_primary = true
  AND sf.feature_gap_count > 0
GROUP BY mt.theme
ORDER BY total_feature_gaps DESC;


-- ── INSIGHT 10: Take C theme co-occurrence ───────────────────
-- Question: Which semantic themes consistently appear together?
-- Audience: Product and Engineering leads — compound problems.
-- Note: Uses Take C semantic_meeting_themes table (26 clusters).

SELECT ca.theme_title                                         AS theme_a,
       cb.theme_title                                         AS theme_b,
       count(*)                                               AS meetings
FROM meeting_analytics.semantic_meeting_themes a
JOIN meeting_analytics.semantic_meeting_themes b
  ON a.meeting_id = b.meeting_id AND a.cluster_id < b.cluster_id
JOIN meeting_analytics.semantic_clusters ca ON a.cluster_id = ca.cluster_id
JOIN meeting_analytics.semantic_clusters cb ON b.cluster_id = cb.cluster_id
GROUP BY ca.theme_title, cb.theme_title
ORDER BY meetings DESC
LIMIT 15;

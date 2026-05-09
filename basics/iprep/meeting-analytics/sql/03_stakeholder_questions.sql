-- =============================================================
-- 03_stakeholder_questions.sql
-- Run in DBeaver connected to: localhost:5434 / rag_db / meeting_analytics
-- 16 stakeholder questions beyond the 10 core insight queries.
-- All verified against rag_db @ localhost:5434.
-- =============================================================


-- ── SALES ─────────────────────────────────────────────────────

-- ── S1: Pricing signal concentration ─────────────────────────
-- Question: Where do pricing_offer key moments appear? Which themes/call types?
--           Are they paired with churn signals in the same meeting?
-- Audience: Sales leadership.

SELECT mt.theme,
       ct.call_type,
       count(*)                                           AS pricing_moments,
       sum(sf.churn_signal_count)                        AS co_churn_signals,
       round(avg(sf.net_sentiment)::numeric, 3)           AS avg_net_sentiment
FROM meeting_analytics.key_moments km
JOIN meeting_analytics.meeting_themes mt  ON km.meeting_id = mt.meeting_id
JOIN meeting_analytics.call_types ct      ON km.meeting_id = ct.meeting_id
JOIN meeting_analytics.sentiment_features sf ON km.meeting_id = sf.meeting_id
WHERE km.moment_type = 'pricing_offer'
  AND mt.is_primary = true
GROUP BY mt.theme, ct.call_type
ORDER BY pricing_moments DESC;


-- ── S2: Repeat organizers in high-risk meetings ───────────────
-- Question: Which organizer emails recur in meetings with churn_signal >= 1
--           and negative sentiment?
-- Audience: Sales leadership, CSMs — named contacts, actionable.

SELECT m.organizer_email,
       count(*)                                           AS high_risk_meetings,
       sum(sf.churn_signal_count)                        AS total_churn_signals,
       round(avg(sf.net_sentiment)::numeric, 3)           AS avg_net_sentiment
FROM meeting_analytics.meetings m
JOIN meeting_analytics.sentiment_features sf  ON m.meeting_id = sf.meeting_id
JOIN meeting_analytics.meeting_summaries ms   ON m.meeting_id = ms.meeting_id
WHERE sf.churn_signal_count >= 1
  AND ms.overall_sentiment IN ('negative', 'very-negative', 'mixed-negative')
GROUP BY m.organizer_email
ORDER BY high_risk_meetings DESC, total_churn_signals DESC;


-- ── S3: Churn signal text samples ────────────────────────────
-- Question: What are customers actually saying in churn signal moments?
-- Audience: Sales leadership, CSMs — verbatim quotes for playbook.

SELECT km.meeting_id,
       mt.theme                                           AS primary_theme,
       ct.call_type,
       ms.overall_sentiment,
       km.speaker_name,
       km.text
FROM meeting_analytics.key_moments km
JOIN meeting_analytics.meeting_themes mt  ON km.meeting_id = mt.meeting_id
JOIN meeting_analytics.call_types ct      ON km.meeting_id = ct.meeting_id
JOIN meeting_analytics.meeting_summaries ms ON km.meeting_id = ms.meeting_id
WHERE km.moment_type = 'churn_signal'
  AND mt.is_primary = true
ORDER BY mt.theme, ct.call_type;


-- ── SUPPORT ───────────────────────────────────────────────────

-- ── S4: Action item volume by theme ──────────────────────────
-- Question: Which themes generate the most follow-up work?
-- Audience: Support leaders, operations — workload and capacity planning.
-- Note: is_primary = true prevents double-counting meetings with multiple themes.

SELECT mt.theme,
       count(ai.action_item)                              AS total_action_items,
       count(DISTINCT ai.meeting_id)                      AS meetings,
       round(count(ai.action_item)::numeric /
             count(DISTINCT ai.meeting_id), 2)            AS action_items_per_meeting
FROM meeting_analytics.meeting_themes mt
JOIN meeting_analytics.action_items ai ON mt.meeting_id = ai.meeting_id
WHERE mt.is_primary = true
GROUP BY mt.theme
ORDER BY action_items_per_meeting DESC;


-- ── S5: Meeting duration by theme / call type ─────────────────
-- Question: Are reliability meetings longer? Quantifies operational cost.
-- Audience: Support leaders, operations.

SELECT mt.theme,
       count(DISTINCT m.meeting_id)                       AS meetings,
       round(avg(m.duration_minutes)::numeric, 1)         AS avg_duration_min,
       round(min(m.duration_minutes)::numeric, 1)         AS min_duration_min,
       round(max(m.duration_minutes)::numeric, 1)         AS max_duration_min
FROM meeting_analytics.meetings m
JOIN meeting_analytics.meeting_themes mt ON m.meeting_id = mt.meeting_id
WHERE mt.is_primary = true
GROUP BY mt.theme
ORDER BY avg_duration_min DESC;

-- Same broken down by call type:
SELECT ct.call_type,
       round(avg(m.duration_minutes)::numeric, 1)         AS avg_duration_min,
       count(*)                                           AS meetings
FROM meeting_analytics.meetings m
JOIN meeting_analytics.call_types ct ON m.meeting_id = ct.meeting_id
GROUP BY ct.call_type
ORDER BY avg_duration_min DESC;


-- ── S6: Technical issue concentration by theme ────────────────
-- Question: Which themes generate the most technical_issue key moments?
-- Audience: Support leaders, engineering.

SELECT mt.theme,
       count(*)                                           AS technical_issues,
       count(DISTINCT km.meeting_id)                      AS meetings,
       round(count(*)::numeric /
             count(DISTINCT km.meeting_id), 2)            AS issues_per_meeting
FROM meeting_analytics.key_moments km
JOIN meeting_analytics.meeting_themes mt ON km.meeting_id = mt.meeting_id
WHERE km.moment_type = 'technical_issue'
  AND mt.is_primary = true
GROUP BY mt.theme
ORDER BY issues_per_meeting DESC;


-- ── S7: Key moment type breakdown ────────────────────────────
-- Question: Distribution of all 8 moment types across the full dataset.
-- Audience: All — sets context and baselines for every signal-based query.

SELECT moment_type,
       count(*)                                           AS total,
       count(DISTINCT meeting_id)                         AS meetings_with_signal,
       round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct_of_all_moments
FROM meeting_analytics.key_moments
GROUP BY moment_type
ORDER BY total DESC;


-- ── ENGINEERING ───────────────────────────────────────────────

-- ── E1: Technical issue text samples ─────────────────────────
-- Question: What are the recurring technical problems, grouped by theme?
-- Audience: Engineering leads — pattern recognition for incident reduction.

SELECT mt.theme,
       km.meeting_id,
       km.speaker_name,
       km.text
FROM meeting_analytics.key_moments km
JOIN meeting_analytics.meeting_themes mt ON km.meeting_id = mt.meeting_id
WHERE km.moment_type = 'technical_issue'
  AND mt.is_primary = true
ORDER BY mt.theme, km.meeting_id;


-- ── E2: Positive pivot signals ───────────────────────────────
-- Question: Which themes recover mid-call (positive_pivot moments)?
--           Where are we turning the conversation around?
-- Audience: Engineering and support leads — counter-narrative for retros.

SELECT mt.theme,
       count(*)                                           AS positive_pivots,
       count(DISTINCT km.meeting_id)                      AS meetings,
       round(avg(sf.net_sentiment)::numeric, 3)           AS avg_net_sentiment
FROM meeting_analytics.key_moments km
JOIN meeting_analytics.meeting_themes mt     ON km.meeting_id = mt.meeting_id
JOIN meeting_analytics.sentiment_features sf ON km.meeting_id = sf.meeting_id
WHERE km.moment_type = 'positive_pivot'
  AND mt.is_primary = true
GROUP BY mt.theme
ORDER BY positive_pivots DESC;


-- ── PRODUCT ───────────────────────────────────────────────────

-- ── P1: Feature gap text samples ─────────────────────────────
-- Question: What exactly are customers asking for?
-- Audience: Product managers — direct customer voice for roadmap input.

SELECT mt.theme,
       km.meeting_id,
       km.speaker_name,
       km.text
FROM meeting_analytics.key_moments km
JOIN meeting_analytics.meeting_themes mt ON km.meeting_id = mt.meeting_id
WHERE km.moment_type = 'feature_gap'
  AND mt.is_primary = true
ORDER BY mt.theme, km.meeting_id;


-- ── P2: Top summary topic tags ───────────────────────────────
-- Question: Most frequent topic tags from AI-generated summaries.
-- Audience: Product managers — product area frequency signal.

SELECT normalized_topic                                    AS topic,
       count(*)                                           AS meetings
FROM meeting_analytics.summary_topics
GROUP BY normalized_topic
ORDER BY meetings DESC
LIMIT 30;


-- ── P3: Praise signal concentration ──────────────────────────
-- Question: Which themes generate praise moments? Where are we winning?
-- Audience: Product managers, marketing — positive signal for positioning.

SELECT mt.theme,
       count(*)                                           AS praise_moments,
       count(DISTINCT km.meeting_id)                      AS meetings,
       round(avg(sf.net_sentiment)::numeric, 3)           AS avg_net_sentiment
FROM meeting_analytics.key_moments km
JOIN meeting_analytics.meeting_themes mt     ON km.meeting_id = mt.meeting_id
JOIN meeting_analytics.sentiment_features sf ON km.meeting_id = sf.meeting_id
WHERE km.moment_type = 'praise'
  AND mt.is_primary = true
GROUP BY mt.theme
ORDER BY praise_moments DESC;


-- ── P4: Take C cluster size + call type breakdown ─────────────
-- Question: For each semantic cluster — meeting count, dominant call type,
--           avg sentiment. Which clusters are support-heavy vs commercial?
-- Audience: Product and engineering leads.

SELECT sc.cluster_id,
       sc.theme_title,
       sc.phrase_count,
       count(DISTINCT smt.meeting_id)                     AS meetings,
       mode() WITHIN GROUP (ORDER BY ct.call_type)        AS dominant_call_type,
       round(avg(sf.net_sentiment)::numeric, 3)           AS avg_net_sentiment
FROM meeting_analytics.semantic_clusters sc
JOIN meeting_analytics.semantic_meeting_themes smt ON sc.cluster_id = smt.cluster_id
JOIN meeting_analytics.call_types ct               ON smt.meeting_id = ct.meeting_id
JOIN meeting_analytics.sentiment_features sf       ON smt.meeting_id = sf.meeting_id
WHERE smt.is_primary = true
GROUP BY sc.cluster_id, sc.theme_title, sc.phrase_count
ORDER BY meetings DESC;


-- ── OPS / CROSS-CUTTING ───────────────────────────────────────

-- ── O1: Participant count vs meeting outcome ──────────────────
-- Question: Do larger meetings correlate with worse sentiment or more churn?
-- Audience: Operations, engineering — meeting hygiene signal.

SELECT count(mp.email)                                    AS participant_count,
       round(avg(sf.net_sentiment)::numeric, 3)           AS avg_net_sentiment,
       round(avg(sf.churn_signal_count)::numeric, 3)      AS avg_churn_signals,
       count(DISTINCT mp.meeting_id)                      AS meetings
FROM meeting_analytics.meeting_participants mp
JOIN meeting_analytics.sentiment_features sf ON mp.meeting_id = sf.meeting_id
GROUP BY mp.meeting_id
ORDER BY participant_count DESC;

-- Bucketed version for charting:
SELECT CASE
         WHEN participant_count <= 2  THEN '1-2'
         WHEN participant_count <= 4  THEN '3-4'
         WHEN participant_count <= 6  THEN '5-6'
         ELSE '7+'
       END                                                AS size_bucket,
       count(*)                                           AS meetings,
       round(avg(net_sentiment)::numeric, 3)              AS avg_net_sentiment,
       round(avg(churn_signals)::numeric, 3)              AS avg_churn_signals
FROM (
    SELECT mp.meeting_id,
           count(mp.email)           AS participant_count,
           sf.net_sentiment,
           sf.churn_signal_count     AS churn_signals
    FROM meeting_analytics.meeting_participants mp
    JOIN meeting_analytics.sentiment_features sf ON mp.meeting_id = sf.meeting_id
    GROUP BY mp.meeting_id, sf.net_sentiment, sf.churn_signal_count
) sub
GROUP BY size_bucket
ORDER BY size_bucket;


-- ── O2: Action item ownership ─────────────────────────────────
-- Question: Which owners appear most? Are specific teams overloaded?
-- Audience: Operations, support leads.

SELECT owner,
       count(*)                                           AS action_items,
       count(DISTINCT meeting_id)                         AS meetings
FROM meeting_analytics.action_items
WHERE owner IS NOT NULL
GROUP BY owner
ORDER BY action_items DESC
LIMIT 20;


-- ── O3: Take B vs Take C cross-tab ────────────────────────────
-- Question: How do KMeans clusters (B) map to semantic clusters (C)?
--           Validates that themes are real signal, not method artefacts.
-- Audience: All — methodological cross-validation.
-- Note: agreement proxy ~37% reflects resolution difference (8 B vs 26 C),
--       not disagreement. See compare_b_vs_c.py for full analysis.

SELECT kc.label                                           AS kmeans_cluster,
       sc.theme_title                                     AS semantic_cluster,
       count(*)                                           AS meetings
FROM meeting_analytics.kmeans_meeting_clusters kmc
JOIN meeting_analytics.kmeans_clusters kc     ON kmc.cluster_id = kc.cluster_id
JOIN meeting_analytics.semantic_meeting_themes smt ON kmc.meeting_id = smt.meeting_id
JOIN meeting_analytics.semantic_clusters sc   ON smt.cluster_id = sc.cluster_id
WHERE smt.is_primary = true
GROUP BY kc.label, sc.theme_title
ORDER BY kc.label, meetings DESC;

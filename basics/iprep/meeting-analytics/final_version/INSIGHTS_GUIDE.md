# Notebook Insights Guide — Meeting Analytics
Last updated: 2026-05-09

---

## Company & Product Context

**Company:** AegisCloud (also "Aegis Cloud Security") — B2B enterprise cybersecurity / data protection SaaS.

**Four products** — confirmed across **66 of 100 meetings** (177 occurrences), consistently capitalized:

| Product | Full brand name | What it does | Primary threat | Notes |
|---------|----------------|-------------|----------------|-------|
| **Detect** | Aegis Detect | Threat monitoring, SIEM integration, event ingestion pipeline | SentinelShield (strong SIEM story, LogVault connector gap) | March outage is the dominant cross-cutting narrative |
| **Protect** | Aegis Protect | Data backup and recovery, agent-based | VaultEdge (growing backup competitor) | Version notification gap (customers not told of fixes) |
| **Comply** | Aegis Comply | Compliance reporting: SOC 2, HIPAA, ISO 27001, PCI DSS | VaultEdge (emerging compliance play) | **Comply v2** is an active launch milestone — on-demand auditor-ready reports, multi-framework view |
| **Identity** | Aegis Identity | IAM, MFA, SSO, SCIM provisioning, audit trail | CyberNova (aggressive mid-market pricing) | Cross-module integration with Comply (audit trail API) in progress |

**Key competitor names to track in the data:** SentinelShield, VaultEdge, CyberNova, Fortiguard Pro

**Critical cross-product finding:**
The March Detect outage is not just an engineering incident — it is a **company-wide narrative** appearing in post-mortems, customer success calls, renewal discussions, and internal planning meetings simultaneously. It is also stealing engineering resources from the Comply v2 launch: *"Detect outage pulled two engineers for four days, causing a 5–6 day slip on HIPAA and ISO 27001 report generation."* Comply runs on separate infrastructure and was unaffected — this is a key message for at-risk account communications.

**Comply v2 as the counter-narrative:**
Multiple meetings reference pairing the outage recovery story with the Comply v2 launch to give at-risk accounts a forward-looking reason to stay. This is an organic sales/CS strategy visible in the transcripts — worth surfacing explicitly.

**~34 meetings have no product name mention** — likely generic support calls (billing, backup agent issues) that don't reference a named product by name.

**Product attribution strategy for queries:** Use case-sensitive LIKE on `meeting_summaries.summary_text` and `key_moments.text` (product names are consistently capitalized):
```sql
text LIKE '%Detect%'   -- Aegis Detect
text LIKE '%Protect%'  -- Aegis Protect
text LIKE '%Comply%'   -- Aegis Comply (catches both "Comply" and "Comply v2")
text LIKE '%Identity%' -- Aegis Identity
```

---

## Narrative Arc

Two parallel stories:

**The problem:** Detect's ingestion pipeline has a systemic single-point-of-failure — known since Q3, deprioritized for feature work, caused a complete outage (zero threat monitoring, 6–12h). Crestwood Health and Hartwell Financial are actively evaluating SentinelShield. Protect has a version notification gap. The Detect outage is bleeding into renewal conversations.

**The opportunity:** Comply v2 is emerging as a differentiator — ahead of VaultEdge on on-demand auditor-ready reports. Compliance meetings have the highest sentiment in the dataset. Customers asking for Comply features are in growth mode, not distress.

---

## Slide Deck — 3 Visuals Only

> The panel cares about the finding and the "so what." Charts are proof, not the presentation.
> Everything else lives in the notebook for Q&A.

| # | Visual | One-line finding |
|---|--------|-----------------|
| **V1** | Theme × sentiment heatmap | Detect is the only theme with majority-negative meetings. Comply is the opposite. |
| **V2** | Churn signal density bar | Detect generates 1.04 churn signals/meeting. Customer Retention generates 0.71. Outage calls are more commercially dangerous than renewal calls. |
| **V3** | High-risk watchlist table | Named meetings with churn signals + negative sentiment. These accounts need a call this week. |

Everything else is **notebook only** — pull it up during Q&A if asked.

---

## Data Source Note

All queries use base tables + Final Version semantic tables only:
- `meetings`, `meeting_summaries`, `key_moments`, `action_items`, `transcript_lines`, `meeting_participants`
- `semantic_clusters`, `semantic_phrases`, `semantic_meeting_themes`

Substitutions for derived columns from aggregation tables:
- Signal counts → `count(*) FILTER (WHERE km.moment_type = '...')` from `key_moments`
- Sentiment → `meeting_summaries.sentiment_score` and `overall_sentiment`
- Theme → `semantic_meeting_themes` JOIN `semantic_clusters`
- Call type → `semantic_meeting_themes.call_type`

---

## Section 1 — Orientation

**Two charts. Sets scale and composition of the dataset.**

### N1 — Theme landscape
- **Chart:** Horizontal bar, sorted by meeting count, colored by `audience` field
- **Tables:** `semantic_clusters` + `semantic_meeting_themes`
- **Key finding:** Which 3–4 themes dominate (expected: Detect reliability, Compliance, IAM, Backup)
- **Slide:** Notebook only

```sql
SELECT sc.theme_title, sc.audience, sc.phrase_count,
       count(DISTINCT smt.meeting_id) AS meetings
FROM meeting_analytics.semantic_clusters sc
LEFT JOIN meeting_analytics.semantic_meeting_themes smt
       ON sc.cluster_id = smt.cluster_id AND smt.is_primary = true
GROUP BY sc.cluster_id, sc.theme_title, sc.audience, sc.phrase_count
ORDER BY meetings DESC NULLS LAST;
```

### N2 — Call type split
- **Chart:** Donut chart (support / external / internal %)
- **Tables:** `semantic_meeting_themes`
- **Key finding:** ~47% support, ~27% renewal/external, ~12% internal incident (from prior analysis)
- **Slide:** Notebook only

```sql
SELECT call_type, count(*) AS meetings,
       round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
FROM meeting_analytics.semantic_meeting_themes
WHERE is_primary = true AND call_type IS NOT NULL
GROUP BY call_type ORDER BY meetings DESC;
```

---

## Section 2 — UMAP Scatter

### N3 — Semantic cluster map
- **Chart:** 2D scatter from `viz_coords.csv`, points colored by cluster_id, top 5 clusters labeled
- **Tables:** `final_version/outputs/viz_coords.csv` + `semantic_clusters` (join on cluster_id for label)
- **Key finding:** Reliability/Detect clusters are dense and near each other; Comply/Identity are distinct islands
- **Slide:** Notebook only

---

## Section 3 — The Problem Story (Negative Signals)

**Four charts that build the Detect/Reliability case.**

### N4 — Theme × sentiment heatmap
- **Chart:** Heatmap (themes on Y, sentiment buckets on X, cell = meeting count)
- **Tables:** `semantic_meeting_themes` + `meeting_summaries` + `semantic_clusters`
- **Key finding:** Detect/Reliability themes show majority mixed-negative or negative; Comply shows majority positive
- **Slide:** **V1** (slide deck)

```sql
SELECT sc.theme_title, ms.overall_sentiment, count(*) AS meetings
FROM meeting_analytics.semantic_meeting_themes smt
JOIN meeting_analytics.semantic_clusters sc ON smt.cluster_id = sc.cluster_id
JOIN meeting_analytics.meeting_summaries ms ON smt.meeting_id = ms.meeting_id
WHERE smt.is_primary = true
GROUP BY sc.theme_title, ms.overall_sentiment
ORDER BY sc.theme_title, meetings DESC;
```

### N5 — Average sentiment score by theme (ranked)
- **Chart:** Horizontal bar chart, sorted ascending (worst at bottom)
- **Tables:** `semantic_meeting_themes` + `meeting_summaries` + `semantic_clusters`
- **Key finding:** Reliability/Detect at the bottom (worst), Compliance at the top — largest gap in dataset
- **Slide:** Notebook only

```sql
SELECT sc.theme_title,
       count(DISTINCT smt.meeting_id) AS meetings,
       round(avg(ms.sentiment_score)::numeric, 2) AS avg_sentiment_score
FROM meeting_analytics.semantic_meeting_themes smt
JOIN meeting_analytics.semantic_clusters sc ON smt.cluster_id = sc.cluster_id
JOIN meeting_analytics.meeting_summaries ms ON smt.meeting_id = ms.meeting_id
WHERE smt.is_primary = true
GROUP BY sc.theme_title
ORDER BY avg_sentiment_score ASC;
```

### N6 — Churn signal density by theme (HEADLINE CHART)
- **Chart:** Bar chart sorted descending, bar for Detect/Reliability highlighted
- **Tables:** `semantic_meeting_themes` + `key_moments` + `semantic_clusters`
- **Key finding:** Reliability = 1.04 churn signals/meeting > Customer Retention = 0.71 — outage calls are MORE commercially dangerous than renewal calls
- **Slide:** **V2** (slide deck)

```sql
SELECT sc.theme_title,
       count(DISTINCT smt.meeting_id) AS meetings,
       count(*) FILTER (WHERE km.moment_type = 'churn_signal') AS total_churn_signals,
       round(
           count(*) FILTER (WHERE km.moment_type = 'churn_signal')::numeric
           / nullif(count(DISTINCT smt.meeting_id), 0), 2
       ) AS churn_per_meeting
FROM meeting_analytics.semantic_meeting_themes smt
JOIN meeting_analytics.semantic_clusters sc ON smt.cluster_id = sc.cluster_id
LEFT JOIN meeting_analytics.key_moments km ON smt.meeting_id = km.meeting_id
WHERE smt.is_primary = true
GROUP BY sc.theme_title
ORDER BY churn_per_meeting DESC NULLS LAST;
```

### N7 — Reliability-to-commercial bleed
- **Chart:** Count stat + supporting table of meetings that span both Reliability AND Retention/Renewal themes
- **Tables:** `semantic_meeting_themes` + `semantic_clusters` + `meeting_summaries`
- **Key finding:** X meetings contain BOTH a reliability/incident theme AND a customer retention/renewal theme — outages are spilling into commercial conversations
- **Slide:** Notebook only

```sql
-- Count of bleed meetings
SELECT count(DISTINCT a.meeting_id) AS bleed_meetings
FROM meeting_analytics.semantic_meeting_themes a
JOIN meeting_analytics.semantic_meeting_themes b ON a.meeting_id = b.meeting_id
JOIN meeting_analytics.semantic_clusters ca ON a.cluster_id = ca.cluster_id
JOIN meeting_analytics.semantic_clusters cb ON b.cluster_id = cb.cluster_id
WHERE (ca.theme_title ILIKE '%reliab%' OR ca.theme_title ILIKE '%incident%' OR ca.theme_title ILIKE '%outage%')
  AND (cb.theme_title ILIKE '%retention%' OR cb.theme_title ILIKE '%renewal%' OR cb.theme_title ILIKE '%commercial%');

-- Full list with sentiment + churn count
SELECT m.title, ms.overall_sentiment, ms.sentiment_score,
       count(*) FILTER (WHERE km.moment_type = 'churn_signal') AS churn_signals
FROM meeting_analytics.semantic_meeting_themes a
JOIN meeting_analytics.semantic_meeting_themes b ON a.meeting_id = b.meeting_id
JOIN meeting_analytics.semantic_clusters ca ON a.cluster_id = ca.cluster_id
JOIN meeting_analytics.semantic_clusters cb ON b.cluster_id = cb.cluster_id
JOIN meeting_analytics.meetings m ON a.meeting_id = m.meeting_id
JOIN meeting_analytics.meeting_summaries ms ON a.meeting_id = ms.meeting_id
LEFT JOIN meeting_analytics.key_moments km ON a.meeting_id = km.meeting_id
WHERE (ca.theme_title ILIKE '%reliab%' OR ca.theme_title ILIKE '%incident%' OR ca.theme_title ILIKE '%outage%')
  AND (cb.theme_title ILIKE '%retention%' OR cb.theme_title ILIKE '%renewal%' OR cb.theme_title ILIKE '%commercial%')
GROUP BY m.title, ms.overall_sentiment, ms.sentiment_score
ORDER BY churn_signals DESC;
```

---

## Section 4 — The Opportunity Story (Positive Signals)

**Three charts. Comply is winning; praise patterns show what customers value.**

### N8 — Praise signal concentration by theme
- **Chart:** Bar chart showing praise_moment count per theme; highlight Comply and Identity themes
- **Tables:** `semantic_meeting_themes` + `key_moments` + `semantic_clusters`
- **Key finding:** Which themes generate praise — these are where AegisCloud is winning
- **Slide:** Notebook only

```sql
SELECT sc.theme_title,
       count(*) FILTER (WHERE km.moment_type = 'praise') AS praise_moments,
       count(DISTINCT smt.meeting_id) AS meetings,
       round(avg(ms.sentiment_score)::numeric, 2) AS avg_sentiment
FROM meeting_analytics.semantic_meeting_themes smt
JOIN meeting_analytics.semantic_clusters sc ON smt.cluster_id = sc.cluster_id
LEFT JOIN meeting_analytics.key_moments km ON smt.meeting_id = km.meeting_id
JOIN meeting_analytics.meeting_summaries ms ON smt.meeting_id = ms.meeting_id
WHERE smt.is_primary = true
GROUP BY sc.theme_title
HAVING count(DISTINCT smt.meeting_id) >= 3
ORDER BY praise_moments DESC;
```

### N9 — Positive pivot rate by theme (where we recover conversations)
- **Chart:** Horizontal bar — positive_pivot moments per meeting by theme
- **Tables:** `semantic_meeting_themes` + `key_moments` + `semantic_clusters`
- **Key finding:** Which themes show the team successfully turning negative conversations around
- **Slide:** Notebook only

```sql
SELECT sc.theme_title,
       count(*) FILTER (WHERE km.moment_type = 'positive_pivot') AS positive_pivots,
       count(DISTINCT smt.meeting_id) AS meetings,
       round(count(*) FILTER (WHERE km.moment_type = 'positive_pivot')::numeric
             / nullif(count(DISTINCT smt.meeting_id), 0), 2) AS pivots_per_meeting
FROM meeting_analytics.semantic_meeting_themes smt
JOIN meeting_analytics.semantic_clusters sc ON smt.cluster_id = sc.cluster_id
LEFT JOIN meeting_analytics.key_moments km ON smt.meeting_id = km.meeting_id
WHERE smt.is_primary = true
GROUP BY sc.theme_title
ORDER BY pivots_per_meeting DESC NULLS LAST;
```

### N10 — Theme co-occurrence: Comply + Product Expansion
- **Chart:** Top 15 co-occurring theme pairs as a heatmap or sorted bar
- **Tables:** `semantic_meeting_themes` + `semantic_clusters`
- **Key finding:** "Compliance + Product Expansion" co-occur in 33 meetings — customers asking for MORE in Comply, not just reporting problems
- **Slide:** Notebook only

```sql
SELECT ca.theme_title AS theme_a, cb.theme_title AS theme_b, count(*) AS meetings
FROM meeting_analytics.semantic_meeting_themes a
JOIN meeting_analytics.semantic_meeting_themes b
  ON a.meeting_id = b.meeting_id AND a.cluster_id < b.cluster_id
JOIN meeting_analytics.semantic_clusters ca ON a.cluster_id = ca.cluster_id
JOIN meeting_analytics.semantic_clusters cb ON b.cluster_id = cb.cluster_id
GROUP BY ca.theme_title, cb.theme_title
ORDER BY meetings DESC
LIMIT 15;
```

---

## Section 5 — Per-Audience Breakdowns

### Engineering — what's on fire, what's working

**E1 — Technical issue concentration by product (NEW)**
- **Chart:** Grouped bar — technical_issue moment count, broken out by Detect/Protect/Comply/Identity
- **Key finding:** Detect generates the most technical issues by product; signals which platform needs reliability investment
- **Slide:** Notebook only

```sql
-- Tag meetings by product mention in summary text
SELECT
    CASE
        WHEN ms.summary_text LIKE '%Detect%' THEN 'Detect'
        WHEN ms.summary_text LIKE '%Protect%' THEN 'Protect'
        WHEN ms.summary_text LIKE '%Comply%' THEN 'Comply'
        WHEN ms.summary_text LIKE '%Identity%' THEN 'Identity'
        ELSE 'Untagged'
    END AS product,
    count(*) FILTER (WHERE km.moment_type = 'technical_issue') AS technical_issues,
    count(DISTINCT km.meeting_id) AS meetings
FROM meeting_analytics.meeting_summaries ms
LEFT JOIN meeting_analytics.key_moments km ON ms.meeting_id = km.meeting_id
GROUP BY product
ORDER BY technical_issues DESC;
```

> **Note:** A meeting may mention multiple products. This assigns to the first match in CASE order. For a multi-product breakdown, use `UNION` or pivot in Python.

**E2 — Positive pivot rate by theme (recovery signal)**
- Covered in N9 above — engineering wants to see where conversations are recoverable

**E3 — Action item volume by theme**
- **Chart:** Bar chart — action_items per meeting by theme
- **Key finding:** Which themes generate the most follow-up engineering work
- **Slide:** No (notebook only)

```sql
SELECT sc.theme_title,
       count(ai.id) AS total_action_items,
       count(DISTINCT smt.meeting_id) AS meetings,
       round(count(ai.id)::numeric / nullif(count(DISTINCT smt.meeting_id), 0), 2) AS items_per_meeting
FROM meeting_analytics.semantic_meeting_themes smt
JOIN meeting_analytics.semantic_clusters sc ON smt.cluster_id = sc.cluster_id
LEFT JOIN meeting_analytics.action_items ai ON smt.meeting_id = ai.meeting_id
WHERE smt.is_primary = true
GROUP BY sc.theme_title
ORDER BY items_per_meeting DESC;
```

---

### Sales — what's at risk, where we're winning

**S1 — Competitor mention map (NEW)**
- **Chart:** Bar chart — competitor mentions by product (Detect/Protect/Comply/Identity)
- **Key finding:** SentinelShield dominates Detect, CyberNova in Identity, VaultEdge in Comply/Protect
- **Slide:** Yes

```sql
SELECT
    CASE
        WHEN km.text ILIKE '%SentinelShield%' THEN 'SentinelShield'
        WHEN km.text ILIKE '%VaultEdge%' THEN 'VaultEdge'
        WHEN km.text ILIKE '%CyberNova%' THEN 'CyberNova'
        WHEN km.text ILIKE '%Fortiguard%' THEN 'Fortiguard Pro'
        ELSE 'Other'
    END AS competitor,
    count(*) AS mentions,
    count(DISTINCT km.meeting_id) AS meetings_mentioned
FROM meeting_analytics.key_moments km
WHERE km.text ILIKE '%SentinelShield%'
   OR km.text ILIKE '%VaultEdge%'
   OR km.text ILIKE '%CyberNova%'
   OR km.text ILIKE '%Fortiguard%'
GROUP BY competitor
ORDER BY mentions DESC;
```

**S2 — Churn signal density by theme** (= N6 above, same chart)
- Most relevant for Sales — leads with the business risk framing

**S3 — Pricing signal paired with churn (deal risk)**
- **Chart:** Table or scatter — meetings with BOTH pricing_offer AND churn_signal moments
- **Key finding:** Where pricing pressure coincides with flight risk — these deals are at acute risk

```sql
SELECT m.meeting_id, m.title, sc.theme_title AS primary_theme, ms.overall_sentiment,
       count(*) FILTER (WHERE km.moment_type = 'pricing_offer') AS pricing_moments,
       count(*) FILTER (WHERE km.moment_type = 'churn_signal') AS churn_signals
FROM meeting_analytics.meetings m
JOIN meeting_analytics.semantic_meeting_themes smt ON m.meeting_id = smt.meeting_id AND smt.is_primary = true
JOIN meeting_analytics.semantic_clusters sc ON smt.cluster_id = sc.cluster_id
JOIN meeting_analytics.meeting_summaries ms ON m.meeting_id = ms.meeting_id
LEFT JOIN meeting_analytics.key_moments km ON m.meeting_id = km.meeting_id
GROUP BY m.meeting_id, m.title, sc.theme_title, ms.overall_sentiment
HAVING count(*) FILTER (WHERE km.moment_type = 'pricing_offer') >= 1
   AND count(*) FILTER (WHERE km.moment_type = 'churn_signal') >= 1
ORDER BY churn_signals DESC;
```

**S4 — Praise moments in external/renewal meetings (what's closing deals)**
- **Chart:** Bar — praise moments per theme, filtered to external call type only
- **Key finding:** Comply praise in external calls = the product that's helping Sales win

```sql
SELECT sc.theme_title,
       count(*) FILTER (WHERE km.moment_type = 'praise') AS praise_moments,
       count(DISTINCT smt.meeting_id) AS meetings
FROM meeting_analytics.semantic_meeting_themes smt
JOIN meeting_analytics.semantic_clusters sc ON smt.cluster_id = sc.cluster_id
LEFT JOIN meeting_analytics.key_moments km ON smt.meeting_id = km.meeting_id
WHERE smt.call_type = 'external'
GROUP BY sc.theme_title
ORDER BY praise_moments DESC;
```

---

### Product — gaps vs growth signals

**P1 — Feature gap requests by theme + sentiment (urgency vs wishlist)**
- **Chart:** Scatter plot — X = feature gap count, Y = avg sentiment score. Bottom-right = urgent gaps under duress. Top-right = constructive wishlist.
- **Tables:** `semantic_meeting_themes` + `key_moments` + `meeting_summaries` + `semantic_clusters`
- **Key finding:** Detect feature gaps raised under distress (low sentiment); Comply gaps are constructive asks (positive meetings)
- **Slide:** Notebook only

```sql
SELECT sc.theme_title,
       count(DISTINCT smt.meeting_id) AS meetings_with_gaps,
       count(*) FILTER (WHERE km.moment_type = 'feature_gap') AS total_gaps,
       round(avg(ms.sentiment_score)::numeric, 2) AS avg_sentiment_score
FROM meeting_analytics.semantic_meeting_themes smt
JOIN meeting_analytics.semantic_clusters sc ON smt.cluster_id = sc.cluster_id
JOIN meeting_analytics.meeting_summaries ms ON smt.meeting_id = ms.meeting_id
LEFT JOIN meeting_analytics.key_moments km ON smt.meeting_id = km.meeting_id
WHERE smt.is_primary = true
GROUP BY sc.theme_title
HAVING count(*) FILTER (WHERE km.moment_type = 'feature_gap') > 0
ORDER BY total_gaps DESC;
```

**P2 — Verbatim feature gap quotes by product (customer voice)**
- **Chart:** Formatted table — "what customers are actually asking for"
- **Key finding:** Direct roadmap input from customer calls; filter by product keyword in meeting_summaries

```sql
SELECT
    CASE
        WHEN ms.summary_text LIKE '%Detect%' THEN 'Detect'
        WHEN ms.summary_text LIKE '%Protect%' THEN 'Protect'
        WHEN ms.summary_text LIKE '%Comply%' THEN 'Comply'
        WHEN ms.summary_text LIKE '%Identity%' THEN 'Identity'
        ELSE 'General'
    END AS product,
    km.text AS customer_ask,
    ms.overall_sentiment
FROM meeting_analytics.key_moments km
JOIN meeting_analytics.meeting_summaries ms ON km.meeting_id = ms.meeting_id
WHERE km.moment_type = 'feature_gap'
ORDER BY product, ms.overall_sentiment;
```

**P3 — Top topic phrases from meeting summaries (frequency signal)**
- **Chart:** Word cloud or top-30 bar chart of raw topic phrases
- **Key finding:** What vocabulary customers are using most — feeds into roadmap naming and positioning

```sql
SELECT unnest(topics) AS topic, count(*) AS meetings
FROM meeting_analytics.meeting_summaries
GROUP BY topic
ORDER BY meetings DESC
LIMIT 30;
```

**P4 — Semantic cluster profile: size, call type, avg sentiment**
- Full table showing all 26 clusters — audience, phrase_count, meeting_count, avg_sentiment
- Notebook only — too dense for a slide but essential reference

---

### Support — volume, recovery, and systemic patterns

**Su1 — Key moment type breakdown (baseline)**
- **Chart:** Stacked bar or donut — distribution of all 8 moment types across the full dataset
- **Tables:** `key_moments`
- **Key finding:** What signal types dominate overall — sets context for every support chart
- **Slide:** No (notebook orientation)

```sql
SELECT moment_type, count(*) AS total,
       count(DISTINCT meeting_id) AS meetings_with_signal,
       round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct_of_all_moments
FROM meeting_analytics.key_moments
GROUP BY moment_type
ORDER BY total DESC;
```

**Su2 — Support escalation themes (what's driving ticket volume)**
- **Chart:** Horizontal bar — technical_issue + concern moments per theme, filtered to support call type
- **Key finding:** Which product/theme combination drives most support load; where to focus defect reduction
- **Slide:** Notebook only

```sql
SELECT sc.theme_title,
       count(*) FILTER (WHERE km.moment_type = 'technical_issue') AS technical_issues,
       count(*) FILTER (WHERE km.moment_type = 'concern') AS concerns,
       count(DISTINCT smt.meeting_id) AS meetings
FROM meeting_analytics.semantic_meeting_themes smt
JOIN meeting_analytics.semantic_clusters sc ON smt.cluster_id = sc.cluster_id
LEFT JOIN meeting_analytics.key_moments km ON smt.meeting_id = km.meeting_id
WHERE smt.call_type = 'support'
GROUP BY sc.theme_title
ORDER BY (count(*) FILTER (WHERE km.moment_type = 'technical_issue')
        + count(*) FILTER (WHERE km.moment_type = 'concern')) DESC;
```

**Su3 — Support recovery: positive pivot rate by theme**
- Same as N9 above, filtered to `call_type = 'support'`
- **Key finding:** Where support teams are successfully de-escalating; model for training

**Su4 — Meeting duration by theme (time cost of reliability failures)**
- **Chart:** Horizontal bar — avg meeting duration by theme
- **Key finding:** Reliability meetings run longer, consuming more support capacity

```sql
SELECT sc.theme_title,
       round(avg(m.duration_minutes)::numeric, 1) AS avg_duration_min,
       count(DISTINCT m.meeting_id) AS meetings
FROM meeting_analytics.meetings m
JOIN meeting_analytics.semantic_meeting_themes smt ON m.meeting_id = smt.meeting_id AND smt.is_primary = true
JOIN meeting_analytics.semantic_clusters sc ON smt.cluster_id = sc.cluster_id
GROUP BY sc.theme_title
ORDER BY avg_duration_min DESC;
```

---

## Section 6 — Actionable Outputs

**Close the notebook on these. Most directly useful for the panel.**

### N_final1 — High-risk meeting watchlist
- **Chart:** Sorted table — churn_signal count + sentiment + theme + call_type + meeting title
- **Tables:** `meetings` + `semantic_meeting_themes` + `meeting_summaries` + `key_moments` + `semantic_clusters`
- **Key finding:** Named meetings that need immediate CSM follow-up
- **Slide:** **V3** (slide deck)

```sql
SELECT m.meeting_id, m.title, sc.theme_title AS primary_theme,
       smt.call_type, ms.overall_sentiment, ms.sentiment_score,
       count(*) FILTER (WHERE km.moment_type = 'churn_signal') AS churn_signals,
       count(*) FILTER (WHERE km.moment_type = 'concern') AS concerns
FROM meeting_analytics.meetings m
JOIN meeting_analytics.semantic_meeting_themes smt ON m.meeting_id = smt.meeting_id AND smt.is_primary = true
JOIN meeting_analytics.semantic_clusters sc ON smt.cluster_id = sc.cluster_id
JOIN meeting_analytics.meeting_summaries ms ON m.meeting_id = ms.meeting_id
LEFT JOIN meeting_analytics.key_moments km ON m.meeting_id = km.meeting_id
WHERE ms.overall_sentiment IN ('negative', 'very-negative', 'mixed-negative')
GROUP BY m.meeting_id, m.title, sc.theme_title, smt.call_type, ms.overall_sentiment, ms.sentiment_score
HAVING count(*) FILTER (WHERE km.moment_type = 'churn_signal') >= 1
ORDER BY churn_signals DESC, ms.sentiment_score ASC;
```

### N_final2 — Verbatim churn signal quotes
- **Chart:** Table filtered by theme — "what customers are actually saying when they're at risk of leaving"
- **Slide:** No (notebook drill-down)

```sql
SELECT sc.theme_title, smt.call_type, ms.overall_sentiment, km.speaker, km.text
FROM meeting_analytics.key_moments km
JOIN meeting_analytics.semantic_meeting_themes smt ON km.meeting_id = smt.meeting_id AND smt.is_primary = true
JOIN meeting_analytics.semantic_clusters sc ON smt.cluster_id = sc.cluster_id
JOIN meeting_analytics.meeting_summaries ms ON km.meeting_id = ms.meeting_id
WHERE km.moment_type = 'churn_signal'
ORDER BY sc.theme_title, ms.overall_sentiment;
```

---

## Slide Deck Sequence (30 min)

| # | Slide | Visual | Key message |
|---|-------|--------|-------------|
| 1 | Title | — | "What 100 AegisCloud calls tell us" |
| 2 | What's in the data | — | 100 meetings, 4 products, 26 discovered themes, 4,313 transcript lines |
| 3 | Three approaches evaluated | Comparison table (see below) | Rule-based → TF-IDF/KMeans → Embedding+HDBSCAN. Each iteration fixed the last one's failure mode. |
| 4 | Why this pipeline | — | No fixed K. Discovers 26 themes naturally. Density-adaptive — handles irregular clusters and noise. LLM names the result; it doesn't do the clustering. |
| 5 | **Sentiment divergence** | **V1** — Theme × sentiment heatmap | Detect is the only theme with majority-negative meetings. Comply is the opposite. |
| 6 | **Commercial risk** | **V2** — Churn signal density bar | Detect 1.04 churn signals/meeting. Customer Retention 0.71. Outage calls are more commercially dangerous than renewal calls. |
| 7 | Verbal findings | — | (1) Detect outage is bleeding into renewal conversations. (2) Comply v2 is the counter-narrative — on separate infrastructure, unaffected. (3) Protect has a version notification gap. |
| 8 | **Action required** | **V3** — High-risk watchlist table | Named meetings with churn signals + negative sentiment. These accounts need a call this week. |
| 9 | Recommendations | — | Stabilize Detect pipeline. Accelerate Comply v2 communication to at-risk accounts. Fix Protect version notification. |

### Slide 3 — Approaches compared

| Approach | How it works | Failure mode |
|----------|-------------|--------------|
| Rule-based keyword matching | Fixed taxonomy, LIKE/regex on text | Brittle — misses paraphrase, can't discover unknown themes |
| TF-IDF + KMeans | Bag-of-words vectors, fixed k=8 | Must guess k upfront; spherical clusters only; centroid terms mix topics ("backup / hybrid / recovery / onboarding") |
| **Embedding + HDBSCAN** *(chosen)* | Semantic embeddings, density-based clustering | k discovered naturally (26); density-adaptive; noise-robust; LLM names clusters from top phrases |

Key tradeoff: more complex pipeline + LLM labels are stochastic between runs. Mitigated by sorting phrases by centroid proximity before prompting — labels are stable in practice.

---

## Open Questions Before Building

1. **Verify theme names in N7** — run N1 first. The ILIKE patterns for reliability/retention need to match actual `theme_title` values. Adjust.
2. **Product attribution coverage** — run E1/S1 queries first and check what % of meetings are "Untagged". If >20%, switch to searching `transcript_lines.sentence` for a broader match.
3. **`sentiment_score` range** — confirm it's 1–5 from the dataset provider (the samples show 1.8, 2.4, 2.8, 3.4 — consistent with 1–5).
4. **`action_items` table column name** — check whether the column is `id`, `action_item`, or `text`. The schema says `text` field. Adjust count queries accordingly.
5. **Competitor mentions via `key_moments` vs `meeting_summaries`** — S1 queries `key_moments.text`; try also against `meeting_summaries.summary_text` for broader coverage. The sample confirms competitor names appear in both.

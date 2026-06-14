# Stakeholder Test Questions

Questions that Head of Engineering, Product, and Customer Support would ask
after meetings are ingested. Each question includes the expected answer from
the dataset and the SQL query that retrieves it from PostgreSQL.

## Table of Contents

1. [PostgreSQL Schema Reference](#1-postgresql-schema-reference)
2. [Head of Engineering](#2-head-of-engineering)
3. [Product](#3-product)
4. [Customer Support](#4-customer-support)
5. [Evaluation Rubric](#5-evaluation-rubric)

---

## 1. PostgreSQL Schema Reference

```sql
meetings (id, title, processed_at, participants, meeting_date)
meeting_insights (id, meeting_id, insight_type, speaker, content, polarity, created_at)
  -- insight_type: 'sentiment_shift' | 'pain_point' | 'competitor'
action_items (id, meeting_id, owner, action, deadline, verdict, reason, created_at)
  -- verdict: 'valid' | 'invalid'
```

---

## 2. Head of Engineering

### Q1 — What meetings had active infrastructure incidents?

**Expected answer (from dataset)**:
Meeting `01KQ03B0303900521BB089CA` — "Detect Outage - Remediation Plan Review" —
is a day-3 incident call about an event-processing pipeline outage affecting 112
customer accounts, with zero threat-monitoring visibility for ~6 hours.

**SQL**:
```sql
SELECT m.id, m.title, m.meeting_date, mi.content
FROM meetings m
JOIN meeting_insights mi ON mi.meeting_id = m.id
WHERE mi.insight_type = 'pain_point'
  AND mi.content ILIKE ANY (ARRAY['%outage%', '%incident%', '%downtime%', '%failure%'])
ORDER BY m.meeting_date DESC;
```

---

### Q2 — How many open action items are assigned to engineers, and which are overdue?

**Expected answer**:
- Raj Kapoor: send evening status update once phase-1 rollout completes (deadline: tonight)
- Raj Kapoor: deliver internal retroactive event analysis summary (deadline: Wednesday)
- Brian Cho: prepare affected account breakdown with priority tiers (deadline: Thursday)

**SQL**:
```sql
SELECT ai.owner, ai.action, ai.deadline, m.title, m.meeting_date
FROM action_items ai
JOIN meetings m ON m.id = ai.meeting_id
WHERE ai.verdict = 'valid'
  AND ai.deadline <> 'Unspecified'
ORDER BY ai.owner, ai.deadline;
```

---

### Q3 — What are the most common engineering pain points across all meetings?

**Expected answer**:
Single-point-of-failure in event ingestion layer, lack of circuit-breaker patterns,
no active-active redundancy in the processing pipeline.

**SQL**:
```sql
SELECT mi.content, COUNT(*) AS occurrences
FROM meeting_insights mi
WHERE mi.insight_type = 'pain_point'
GROUP BY mi.content
ORDER BY occurrences DESC
LIMIT 10;
```

---

### Q4 — Which meetings had action items with no owner assigned?

**Expected answer**:
No unowned action items in the test dataset — all items have named owners.
This query is a quality check.

**SQL**:
```sql
SELECT m.title, ai.action, ai.deadline
FROM action_items ai
JOIN meetings m ON m.id = ai.meeting_id
WHERE ai.verdict = 'valid'
  AND (ai.owner IS NULL OR TRIM(ai.owner) = '')
ORDER BY m.meeting_date DESC;
```

---

### Q5 — How many action items were hallucinated (marked invalid) per meeting?

**Expected answer**:
Shows which meetings produced unreliable extractions — useful for tuning the
commitments and validation agents.

**SQL**:
```sql
SELECT m.title,
       COUNT(*) FILTER (WHERE ai.verdict = 'valid')   AS valid_items,
       COUNT(*) FILTER (WHERE ai.verdict = 'invalid') AS invalid_items,
       ROUND(
         COUNT(*) FILTER (WHERE ai.verdict = 'invalid')::numeric /
         NULLIF(COUNT(*), 0) * 100, 1
       ) AS hallucination_pct
FROM meetings m
JOIN action_items ai ON ai.meeting_id = m.id
GROUP BY m.id, m.title
ORDER BY hallucination_pct DESC;
```

---

## 3. Product

### Q6 — Which features or products were mentioned positively across meetings?

**Expected answer**:
Active-active redundant processing nodes and the circuit-breaker pattern were
framed positively in the outage remediation meeting.

**SQL**:
```sql
SELECT m.title, mi.speaker, mi.content, mi.polarity
FROM meeting_insights mi
JOIN meetings m ON m.id = mi.meeting_id
WHERE mi.insight_type = 'sentiment_shift'
  AND mi.polarity IN ('positive', 'mixed')
ORDER BY m.meeting_date DESC;
```

---

### Q7 — Which competitors were mentioned and in what context?

**Expected answer**:
No competitor mentions in the outage-remediation meeting (expected — it's an
internal call). This query will populate once sales/product calls are ingested.

**SQL**:
```sql
SELECT m.title, mi.content AS competitor, m.meeting_date
FROM meeting_insights mi
JOIN meetings m ON m.id = mi.meeting_id
WHERE mi.insight_type = 'competitor'
ORDER BY m.meeting_date DESC;
```

---

### Q8 — Which pain points appear most frequently across all meetings?

**Expected answer**:
In the current dataset: timeline credibility issues, single-point-of-failure
in infrastructure, customer communication gaps.

**SQL**:
```sql
SELECT mi.content, COUNT(DISTINCT mi.meeting_id) AS meeting_count
FROM meeting_insights mi
WHERE mi.insight_type = 'pain_point'
GROUP BY mi.content
ORDER BY meeting_count DESC, mi.content
LIMIT 20;
```

---

### Q9 — How has overall customer sentiment shifted over the last 30 days?

**Expected answer**:
Requires multiple ingested meetings. The query below aggregates polarity counts
per week.

**SQL**:
```sql
SELECT DATE_TRUNC('week', m.meeting_date) AS week,
       mi.polarity,
       COUNT(*) AS count
FROM meeting_insights mi
JOIN meetings m ON m.id = mi.meeting_id
WHERE mi.insight_type = 'sentiment_shift'
  AND m.meeting_date >= NOW() - INTERVAL '30 days'
GROUP BY 1, 2
ORDER BY 1, 2;
```

---

### Q10 — Which speakers drove the most negative sentiment moments?

**Expected answer**:
Brian Cho (customer ticket pressure) and Raj Kapoor (timeline pushback) are the
primary sources of negative/mixed sentiment in the outage meeting.

**SQL**:
```sql
SELECT mi.speaker,
       COUNT(*) FILTER (WHERE mi.polarity = 'negative')      AS negative,
       COUNT(*) FILTER (WHERE mi.polarity = 'mixed')         AS mixed,
       COUNT(*) FILTER (WHERE mi.polarity = 'positive')      AS positive,
       COUNT(*) FILTER (WHERE mi.polarity = 'neutral')       AS neutral
FROM meeting_insights mi
JOIN meetings m ON m.id = mi.meeting_id
WHERE mi.insight_type = 'sentiment_shift'
  AND mi.speaker IS NOT NULL
GROUP BY mi.speaker
ORDER BY negative DESC, mixed DESC;
```

---

## 4. Customer Support

### Q11 — Which meetings generated explicit customer-facing commitments?

**Expected answer**:
"Detect Outage" meeting — Megan Lawson committed to draft updated customer
communication within the hour explaining phased rollout.

**SQL**:
```sql
SELECT m.title, ai.owner, ai.action, ai.deadline
FROM action_items ai
JOIN meetings m ON m.id = ai.meeting_id
WHERE ai.verdict = 'valid'
  AND (
    ai.action ILIKE '%customer%'
    OR ai.action ILIKE '%communication%'
    OR ai.action ILIKE '%email%'
    OR ai.action ILIKE '%notify%'
  )
ORDER BY m.meeting_date DESC;
```

---

### Q12 — What SLA or timeline commitments were made to customers?

**Expected answer**:
Phased visibility restoration by tomorrow morning; redundant nodes live by end
of day; circuit-breaker by Wednesday evening.

**SQL**:
```sql
SELECT m.title, ai.owner, ai.action, ai.deadline
FROM action_items ai
JOIN meetings m ON m.id = ai.meeting_id
WHERE ai.verdict = 'valid'
  AND ai.deadline <> 'Unspecified'
  AND ai.deadline NOT ILIKE '%unspecified%'
ORDER BY m.meeting_date DESC, ai.deadline;
```

---

### Q13 — What are customers most frustrated about?

**Expected answer**:
Based on sentiment shifts with negative polarity: timeline delays, credibility
loss after day-3 of outage, zero threat-monitoring visibility for 6 hours.

**SQL**:
```sql
SELECT mi.content AS frustration, mi.speaker, m.title, m.meeting_date
FROM meeting_insights mi
JOIN meetings m ON m.id = mi.meeting_id
WHERE mi.insight_type = 'sentiment_shift'
  AND mi.polarity = 'negative'
ORDER BY m.meeting_date DESC;
```

---

### Q14 — Which action items are assigned to the support team?

**Expected answer**:
Brian Cho (support/account lead) owns: prepare affected account breakdown
with priority tiers and impact durations (deadline: Thursday).

**SQL**:
```sql
SELECT ai.owner, ai.action, ai.deadline, m.title
FROM action_items ai
JOIN meetings m ON m.id = ai.meeting_id
WHERE ai.verdict = 'valid'
  AND ai.owner ILIKE ANY (ARRAY['%support%', '%brian%', '%account%'])
ORDER BY ai.deadline;
```

---

### Q15 — Which meetings had no resolved action items?

**Expected answer**:
Quality check — meetings where the pipeline produced zero valid action items.
These may indicate a transcript that is too short, too informal, or a failure
of the commitments agent.

**SQL**:
```sql
SELECT m.id, m.title, m.meeting_date
FROM meetings m
WHERE NOT EXISTS (
    SELECT 1
    FROM action_items ai
    WHERE ai.meeting_id = m.id
      AND ai.verdict = 'valid'
)
ORDER BY m.meeting_date DESC;
```

---

## 5. Evaluation Rubric

For each question, score the pipeline's answer on:

| Dimension | 0 | 1 | 2 |
|-----------|---|---|---|
| **Factual accuracy** | Wrong facts from transcript | Partially correct | Matches transcript exactly |
| **Completeness** | Key items missing | Some items missing | All key items captured |
| **Hallucination** | Items not in transcript | Borderline (discussed, not agreed) | Only agreed items |
| **Deadline precision** | Wrong or absent | Approximate | Exact as stated in meeting |
| **Owner attribution** | Wrong person | Ambiguous | Exact named person |

**Minimum passing score per question**: 8 / 10 (average ≥ 1.6 across all dimensions)

**Gold standard**: `dataset/<meeting_id>/summary.json → actionItems[]` field.

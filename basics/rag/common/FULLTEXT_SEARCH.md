# PostgreSQL Full-Text Search: tsvector & tsquery Crash Course

How PostgreSQL full-text search works, and exactly how this codebase uses it.

## Table of Contents

- [The Core Idea](#the-core-idea-in-one-sentence)
- [Part 1: tsvector — The Indexed Form of Text](#part-1-tsvector--the-indexed-form-of-text)
- [Part 2: tsquery — The Query Expression](#part-2-tsquery--the-query-expression)
- [Part 3: The Match Operator @@](#part-3-the-match-operator-)
- [Part 4: ts_rank — Scoring Relevance](#part-4-ts_rank--scoring-relevance)
- [Part 5: The Complete text_search Query](#part-5-the-complete-text_search-query)
- [Part 6: How It Fits Into Hybrid Search](#part-6-how-it-fits-into-hybrid-search)
- [Part 7: Language Configuration](#part-7-language-configuration)
- [Part 8: Gotchas](#part-8-gotchas)
- [Quick Reference](#quick-reference)

---

## The Core Idea in One Sentence

PostgreSQL full-text search works by converting text into a sorted list of normalised lexemes (`tsvector`), then converting a query into a match expression (`tsquery`), and checking whether the document matches the query using the `@@` operator.

---

## Part 1: tsvector — The Indexed Form of Text

### What it is

A `tsvector` is not the original text. It's a **preprocessed, normalised representation** optimised for searching:

```sql
SELECT to_tsvector('english', 'The employees are entitled to 30 days PTO per year');
-- Result:
-- 'day':7 'entitl':5 'employe':2 'per':9 'pto':8 'year':10
```

Three things happened:
1. **Stop words removed** — "The", "are", "to", "30" dropped (too common to be useful)
2. **Stemming applied** — "employees" → `employe`, "entitled" → `entitl` (reduces variants to root)
3. **Position tracking** — each lexeme is tagged with its position(s) in the original text

### In this codebase

The `chunks` table has a **generated column** that auto-computes the tsvector whenever `content` changes:

```sql
-- postgres.py:178
content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
```

`GENERATED ALWAYS AS ... STORED` means:
- PostgreSQL computes it automatically on INSERT and UPDATE
- The value is stored on disk (not recomputed at query time)
- You never write to it manually

There's a **GIN index** on it:

```sql
-- postgres.py:198
CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx
ON chunks
USING GIN(content_tsv)
```

GIN (Generalized Inverted Index) is the right index type for tsvector — it maps each lexeme to the set of rows containing it, enabling fast keyword lookup.

---

## Part 2: tsquery — The Query Expression

### What it is

A `tsquery` is a boolean search expression over lexemes. Several functions create one:

| Function | Input | Behaviour |
|----------|-------|-----------|
| `plainto_tsquery('english', text)` | `'PTO policy'` | AND of all non-stop words: `'pto' & 'polici'` |
| `phraseto_tsquery('english', text)` | `'PTO policy'` | Words must appear in order: `'pto' <-> 'polici'` |
| `to_tsquery('english', text)` | `'PTO & policy'` | Raw boolean — you control `&`, `\|`, `!` |
| `websearch_to_tsquery('english', text)` | `'"PTO policy" OR benefits'` | Google-style syntax |

### What this codebase uses

```python
# postgres.py:347
WHERE c.content_tsv @@ plainto_tsquery('english', $1)
```

`plainto_tsquery` is the safe, simple choice — takes raw user input and ANDs the words. No injection risk, no syntax errors from user input.

Example:
```sql
plainto_tsquery('english', 'employee benefits PTO')
-- produces: 'employe' & 'benefit' & 'pto'
-- matches any row containing all three lexemes
```

---

## Part 3: The Match Operator `@@`

```sql
content_tsv @@ plainto_tsquery('english', $1)
```

Returns `true` if the tsvector contains a match for the tsquery. This is what the `WHERE` clause uses to filter rows.

The GIN index makes this fast — PostgreSQL looks up each lexeme in the index rather than scanning every row.

---

## Part 4: ts_rank — Scoring Relevance

After filtering with `@@`, you want to order by relevance. `ts_rank` does this:

```sql
-- postgres.py:341-342
ts_rank(c.content_tsv, plainto_tsquery('english', $1)) as similarity
```

`ts_rank` produces a float score based on:
- How many query lexemes appear in the document
- How often they appear (term frequency)
- (Optionally) how close together they appear

The score range is roughly 0.0–1.0 for typical documents, but can exceed 1.0 for very dense matches. There's also `ts_rank_cd` which considers cover density (proximity of matching terms).

---

## Part 5: The Complete text_search Query

Here's the full query from `postgres.py:335-353`, annotated:

```sql
SELECT
    c.id          as chunk_id,
    c.document_id,
    c.content,
    -- Score: how relevant is this chunk to the query?
    ts_rank(c.content_tsv, plainto_tsquery('english', $1)) as similarity,
    c.metadata,
    d.title       as document_title,
    d.source      as document_source
FROM chunks c
JOIN documents d ON c.document_id = d.id
-- Filter: only rows that match the query at all
WHERE c.content_tsv @@ plainto_tsquery('english', $1)
-- Order best matches first
ORDER BY ts_rank(c.content_tsv, plainto_tsquery('english', $1)) DESC
LIMIT $2
```

`$1` = the query string (e.g. `"employee benefits"`)
`$2` = `match_count * 2` (over-fetches for RRF merging — see hybrid search)

Note: `plainto_tsquery('english', $1)` is called three times. PostgreSQL will evaluate it once per row (it's not automatically cached within a query). You could factor it out with a CTE or lateral join for efficiency at large scale, but for typical RAG workloads it's fine.

---

## Part 6: How It Fits Into Hybrid Search

The codebase runs **two searches in parallel** and merges them:

```python
# postgres.py:401-404
semantic_results, text_results = await asyncio.gather(
    self.semantic_search(query_embedding, fetch_count),  # pgvector cosine distance
    self.text_search(query, fetch_count),                # tsvector full-text
)
```

Then merges with **Reciprocal Rank Fusion (RRF)**:

```python
# postgres.py:426-471
rrf_score = 1.0 / (k + rank)   # k=60 is a smoothing constant
```

Each chunk gets a score from each ranked list, and scores are summed. A chunk that ranks #2 in semantic AND #3 in text gets a combined RRF score higher than one that only appears in one list. This rewards chunks that are simultaneously semantically similar AND lexically matching.

| Search type | Wins when… |
|-------------|-----------|
| Semantic | Query and document use different words for the same concept ("salary" vs "compensation") |
| Text (tsvector) | Query contains exact keywords, acronyms, or proper nouns ("PTO", "NeuralFlow", "llama3") |
| Hybrid (RRF) | Default — captures both |

---

## Part 7: Language Configuration

All functions in this codebase use `'english'` as the language config:

```sql
to_tsvector('english', content)
plainto_tsquery('english', $1)
```

The language config controls:
- **Stop word list** — words to discard ("the", "a", "is")
- **Stemming dictionary** — how words are reduced to roots (English uses the Snowball stemmer)

Other options: `'simple'` (no stemming, no stop words), `'french'`, `'spanish'`, etc. `'simple'` is useful when you have lots of technical terms or codes that shouldn't be stemmed.

---

## Part 8: Gotchas

### 1. Numbers are stop words in `'english'`
```sql
SELECT to_tsvector('english', '30 days PTO');
-- 'day':2 'pto':3
-- "30" is dropped
```
If you need to search numbers, use `'simple'` config or store them separately.

### 2. Stemming can surprise you
```sql
SELECT to_tsvector('english', 'running runs ran');
-- 'ran':3 'run':1,2
-- All stem to 'run', positions tracked separately
```
This is a feature — searching "run" matches "running", "runs", "ran". But it means searching for "PTO" and "PTs" would both become the same lexeme.

### 3. `plainto_tsquery` silently drops stop words
```sql
SELECT plainto_tsquery('english', 'what is the PTO policy');
-- 'pto' & 'polici'
-- "what", "is", "the" dropped — they're stop words
```
If ALL words in a query are stop words, the tsquery is empty and matches nothing.

### 4. GIN index is only used with `@@`
```sql
WHERE content_tsv @@ plainto_tsquery(...)  -- uses GIN index ✓
WHERE content LIKE '%pto%'                 -- full table scan ✗
```
Always go through tsvector/tsquery for indexed full-text search.

---

## Quick Reference

```sql
-- Create tsvector from text
SELECT to_tsvector('english', 'your text here');

-- Create tsquery from plain text (safe for user input)
SELECT plainto_tsquery('english', 'user query here');

-- Match check
SELECT content_tsv @@ plainto_tsquery('english', 'query');

-- Score a match
SELECT ts_rank(content_tsv, plainto_tsquery('english', 'query'));

-- Full search pattern
SELECT *, ts_rank(content_tsv, plainto_tsquery('english', $1)) AS score
FROM chunks
WHERE content_tsv @@ plainto_tsquery('english', $1)
ORDER BY score DESC
LIMIT 10;

-- Highlight matching terms (useful for debugging)
SELECT ts_headline('english', content, plainto_tsquery('english', $1))
FROM chunks
WHERE content_tsv @@ plainto_tsquery('english', $1);
```

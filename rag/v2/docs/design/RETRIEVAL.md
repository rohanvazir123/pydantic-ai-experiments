# RAG v2 — Retrieval Pipeline

## Table of Contents

- [Retrieval Pipeline](#retrieval-pipeline)
- [Confidence-Based Scoring](#confidence-based-scoring)
  - [Why the Current `similarity` Field Is Not a Confidence Score](#why-the-current-similarity-field-is-not-a-confidence-score)
  - [Design: Dual-Score `SearchResult`](#design-dual-score-searchresult)
  - [Confidence Population](#confidence-population)
  - [Confidence Threshold Filter](#confidence-threshold-filter)
  - [Confidence in the Response](#confidence-in-the-response)
  - [EvalResult Extension](#evalresult-extension)
- [Confidence-Aware Pipeline](#confidence-aware-pipeline)
  - [Architecture Mapping](#architecture-mapping)
  - [Layer 1 — Retrieval Gate (`knowledge/retrieval/retriever.py`)](#layer-1--retrieval-gate-knowledgeretrievalretrieverpy)
  - [Layer 2 — Citation Gate (`knowledge/agent/agent.py`)](#layer-2--citation-gate-knowledgeagentagentpy)
  - [Layer 3 — Judge Gate (`knowledge/agent/judge.py`)](#layer-3--judge-gate-knowledgeagentjudgepy)
  - [Pipeline Orchestrator (`knowledge/agent/pipeline.py`)](#pipeline-orchestrator-knowledgeagentpipelinepy)
  - [Hook Integration](#hook-integration)
  - [Evaluation Extension](#evaluation-extension)
  - [Threshold Calibration Workflow](#threshold-calibration-workflow)
- [Model Tiering](#model-tiering)
  - [Tier Definitions](#tier-definitions)
  - [Encoder Roles — Bi-Encoder vs. Cross-Encoder](#encoder-roles--bi-encoder-vs-cross-encoder)
  - [LLM-as-Judge (Citation Faithfulness)](#llm-as-judge-citation-faithfulness)
  - [Routing Logic (`knowledge/agent/model_router.py`)](#routing-logic-knowledgeagentmodel_routerpy)
  - [Configuration (`settings.py` additions)](#configuration-settingspy-additions)
  - [Observability](#observability)

---

### Retrieval Pipeline

Three firm defaults (non-negotiable in this architecture):
- **Reranking is always on.** `reranker_enabled = True` out of the box. CrossEncoder (`BAAI/bge-reranker-base`) is the default — it runs locally via `sentence-transformers`, no API call. LLMReranker is an opt-in alternative.
- **Sources/citations are always included.** Every response carries `SearchResult.chunk_id`, `document_title`, `document_source`, and `similarity` score. The agent prompt mandates inline citation. Clients receive a structured `citations: list[Citation]` field alongside the answer text — never a bare string.
- **All models are local (Ollama).** No cloud LLM calls by default. Cloud model IDs in the tiering table are available but require explicit `cloud_models_enabled = True` in settings. Local model IDs are the defaults for every tier.

```
POST /v1/search
    │
    ├─► L3 semantic cache check (pgvector cosine sim)
    │       └── HIT → decrypt JWE → return { answer, citations } cached
    │
    ├─► L2 Redis cache check (exact query hash)
    │       └── HIT → return cached { chunks, citations }
    │
    ├─► hybrid retrieval (parallel):
    │       ├── vector_store.semantic_search(query_embedding, k × overfetch_factor)
    │       ├── vector_store.text_search(query_text, k × overfetch_factor)
    │       └── (optional) graph_retriever.query(query_text)   ← NL→Cypher → AGE
    │
    ├─► RRF fusion (k=60)
    │
    ├─► CrossEncoder rerank  ← ALWAYS ON (local BAAI/bge-reranker-base)
    │       trim to match_count; attach similarity + chunk_id to each result
    │
    ├─► score filter (min_relevance_score threshold)
    │
    ├─► Pydantic AI agent (search_knowledge_base tool)
    │       system prompt: "Always cite your sources using [chunk_id]."
    │       structured output: { answer: str, citations: list[Citation] }
    │
    ├─► populate L2 Redis cache  (async)
    └─► populate L3 semantic cache  (async, JWE-encrypted)

Citation model:
    class Citation(BaseModel):
        chunk_id: UUID
        document_title: str
        document_source: str      # file path or URL
        relevance_score: float    # = SearchResult.confidence (post-rerank sigmoid score, 0-1); never raw_score
        excerpt: str              # ≤ 200 chars of the supporting chunk
```

---

### Confidence-Based Scoring

#### Why the Current `similarity` Field Is Not a Confidence Score

The existing `SearchResult.similarity` field is a catch-all that holds fundamentally different values depending on search mode:

| Search mode | What `similarity` actually contains | Calibrated 0–1? |
|-------------|-------------------------------------|-----------------|
| `semantic` | `1 - pgvector cosine distance` | Yes — but cosine similarity ≠ relevance |
| `text` | `ts_rank` output | No — unbounded, no IDF, no length norm |
| `fuzzy` | `pg_trgm word_similarity` | Yes — but trigram overlap ≠ semantic relevance |
| `hybrid` (default) | RRF score `Σ 1/(60+rank)` | No — rank-based, max ~0.05 |

The `min_relevance_score` guardrail in `Retriever` only fires for `search_type == "semantic"` for exactly this reason — applying a threshold to an RRF score or `ts_rank` float would be meaningless. This means the guardrail is effectively dead in the default hybrid path.

After CrossEncoder reranking, scores are normalised to 0–1 and carry real signal — but they are used only for ordering and trimming, not for filtering. No chunk is ever dropped based on post-rerank confidence.

#### Design: Dual-Score `SearchResult`

The `knowledge/` module will separate raw search scores from calibrated confidence:

```python
class SearchResult(BaseModel):
    chunk_id: UUID
    document_id: UUID
    document_title: str
    document_source: str
    content: str
    metadata: dict[str, Any]

    # Raw score from the search leg — scale varies by search_type
    raw_score: float
    raw_score_type: Literal["cosine_similarity", "ts_rank", "trigram_similarity", "rrf"]

    # Calibrated confidence — populated after reranking; None until then
    # Always 0-1; comparable across search types and corpus sizes
    confidence: float | None = None

# NOTE: SearchResult and Citation are defined in knowledge/ingestion/models.py for
# historical reasons (matching v1 layout). They are shared across ingestion, retrieval,
# agent, and API layers. If circular imports arise, move them to knowledge/models.py (root).
```

`Citation.relevance_score` maps to `confidence`, never `raw_score`. The agent and the API response only expose `confidence`.

#### Confidence Population

```
hybrid_search() → SearchResult[] with raw_score=rrf, raw_score_type="rrf", confidence=None
      │
      ▼
CrossEncoderReranker.rerank()
      │  scores all (query, chunk) pairs in one batch forward pass
      │  normalises raw logits → [0, 1] via sigmoid
      │
      └─► SearchResult.confidence = sigmoid(cross_encoder_logit)   # populated here
          SearchResult.raw_score  = rrf_score                      # unchanged

semantic_search() (standalone) → confidence = raw cosine similarity  # already 0-1
text_search()    (standalone) → confidence = None (ts_rank is not calibrated)
```

`confidence` is set on every result returned from the retriever whenever reranking is on (which is always, per design). For standalone semantic search without reranking, `confidence` falls back to the cosine similarity score.

#### Confidence Threshold Filter

Replace the current `search_type == "semantic"` guardrail with a mode-agnostic confidence filter applied post-rerank:

```python
# knowledge/retrieval/retriever.py
MIN_CONFIDENCE_THRESHOLD: float = settings.min_confidence_score  # default 0.10

results = [r for r in reranked if r.confidence is not None and r.confidence >= MIN_CONFIDENCE_THRESHOLD]
```

Settings additions:
```python
min_confidence_score: float = 0.10   # drop chunks with post-rerank confidence < this
confidence_warn_threshold: float = 0.40  # log warning if best chunk confidence < this
```

If the top result's `confidence < confidence_warn_threshold`, the agent receives a low-confidence context flag and the system prompt includes: *"The retrieved context has low confidence scores. State any uncertainty explicitly."*

#### Confidence in the Response

Every API response exposes per-citation confidence:

```json
{
  "answer": "The PTO policy allows ...",
  "citations": [
    {
      "chunk_id": "uuid",
      "document_title": "Employee Handbook",
      "document_source": "hr/handbook.pdf",
      "confidence": 0.87,
      "excerpt": "Employees accrue 15 days of PTO per year..."
    }
  ],
  "low_confidence_context": false
}
```

`low_confidence_context: true` is a flag clients can use to show a UI warning or trigger a human-review hook.

#### EvalResult Extension

Add `confidence` tracking to offline evaluation:

```python
class EvalResult(BaseModel):
    ...
    # Confidence distribution over retrieved chunks
    mean_confidence: float | None       # mean post-rerank confidence across top-K
    min_confidence: float | None        # lowest confidence chunk that was used
    low_confidence_flag: bool = False   # True if min_confidence < warn_threshold
```

This lets the Grafana dashboard correlate low-confidence retrieval with low faithfulness or poor user feedback — the primary signal for knowing when to improve the index or add more data to a corpus.

---

### Confidence-Aware Pipeline

The confidence-aware pipeline wraps the retriever, generator, and judge into a single orchestration function. At each of the three layers a hard gate either short-circuits to an abstention response or lets the request proceed. No answer reaches the user unless it clears all three gates.

Reference design (Microsoft Tech Community — "Confidence-Aware RAG: Teaching Your AI Pipeline to Acknowledge Uncertainty"):

```python
# NOTE: This is a synchronous reference sketch from an external source.
# The actual implementation — ConfidenceAwarePipeline — is fully async.
# See knowledge/agent/pipeline.py.
def confidence_aware_rag(user_query: str) -> dict:
    # Layer 1 — retrieve with confidence gating
    results = retrieve_with_confidence(user_query, threshold=1.5)
    if not results:
        return {"answer": "...", "status": "abstained_retrieval"}

    # Layer 2 — generate with citation requirements
    generation = generate_answer(user_query, context, results)
    if not generation["citation_check"]["is_trustworthy"]:
        return {"answer": "...", "status": "abstained_citation"}

    # Layer 3 — judge the answer
    judgement = judge_answer(user_query, context, generation["answer"])
    if judgement["verdict"] == "unsupported" or judgement["confidence"] < 0.6:
        return {"answer": "...", "status": "abstained_judge"}

    if judgement["verdict"] == "partial":
        generation["answer"] += "\n\nNote: This answer may be incomplete..."

    return {"answer": ..., "status": "answered", "confidence": ..., "sources": [...]}
```

#### Architecture Mapping

Each layer maps to a distinct component in the `knowledge/` module.

```
knowledge/agent/
├── pipeline.py        # ConfidenceAwarePipeline — top-level orchestrator
├── agent.py           # Layer 2: structured generation + citation check
├── judge.py           # Layer 3: LLM-as-judge (nano/small model)
└── model_router.py    # pre-pipeline: routes to correct model tier
```

#### Layer 1 — Retrieval Gate (`knowledge/retrieval/retriever.py`)

`retrieve_with_confidence` runs the standard hybrid retrieval + CrossEncoder rerank pipeline, then computes an **aggregate confidence score** over the top-K results. If the aggregate falls below `retrieval_confidence_threshold` the function returns an empty list and the pipeline short-circuits immediately — no LLM call is made.

**Aggregate score** — sum of `SearchResult.confidence` for the top-K reranked results:

```python
aggregate_confidence = sum(r.confidence for r in reranked_results[:k])
```

Why a sum rather than a mean: a single high-confidence chunk is insufficient if the query spans multiple topics; the sum rewards coverage. With K=5 and threshold=1.5 the system requires an average per-chunk confidence of 0.30 — a deliberately low floor that only blocks truly empty retrieval. Tighten `retrieval_confidence_threshold` per corpus as quality improves.

```python
# knowledge/config/settings.py additions
retrieval_confidence_threshold: float = 1.5   # aggregate sum of top-K confidences
judge_confidence_threshold: float = 0.60      # per judge_answer() call
judge_k: int = 5                              # top-K chunks fed to judge + generator
```

#### Layer 2 — Citation Gate (`knowledge/agent/agent.py`)

The Pydantic AI agent generates the answer as a structured output that includes an inline citation check. The LLM is required to ground every factual claim in a `chunk_id`; if it cannot, `is_trustworthy` is `False`.

```python
class CitationCheck(BaseModel):
    is_trustworthy: bool
    uncited_claims: list[str]   # claims the model couldn't attribute to a chunk

class GenerationResult(BaseModel):
    answer: str
    citations: list[Citation]       # Citation model from Retrieval Pipeline section
    citation_check: CitationCheck
```

System prompt constraint (always included):
> "Every factual statement in your answer MUST be supported by one of the provided source chunks, cited inline as [chunk_id]. If you cannot find a supporting chunk for a claim, omit that claim entirely. Do not invent information."

`is_trustworthy = len(uncited_claims) == 0`. If any claim is uncited, the pipeline returns `abstained_citation` without showing the answer.

This gate catches the failure mode where the LLM has memorised a plausible-sounding answer that happens to contradict or go beyond the retrieved context — independent of whether the retrieval score was high.

#### Layer 3 — Judge Gate (`knowledge/agent/judge.py`)

A separate LLM call (nano or small model tier, cheaper than the generation model) evaluates the answer against the context. The judge is deliberately independent: it receives only the query, context, and answer — not the citation metadata — so it cannot be fooled by a well-formatted but hallucinated citation.

```python
class JudgeResult(BaseModel):
    verdict: Literal["supported", "partial", "unsupported"]
    confidence: float           # 0.0–1.0; judge's own confidence in its verdict
    reasoning: str              # short explanation (logged, not returned to user)

# Judge prompt (system):
# "You are an impartial evaluator. Given a question, a set of source passages,
#  and a generated answer, determine whether the answer is:
#  - supported: fully grounded in the passages
#  - partial: mostly grounded but missing or hedging on some aspects
#  - unsupported: contains claims not found in or contradicted by the passages
#  Return a JSON object with verdict, confidence (0-1), and reasoning."
```

Gate logic:
- `verdict == "unsupported"` OR `confidence < judge_confidence_threshold` → `abstained_judge`
- `verdict == "partial"` → answer proceeds but uncertainty note is appended
- `verdict == "supported"` AND `confidence >= judge_confidence_threshold` → `answered`

The judge uses the `nano` model tier by default. If the nano model's own `confidence` on the verdict is low (< 0.5), escalate the judge call to `small` — one level up. This avoids incorrect abstentions on ambiguous but answerable queries.

#### Pipeline Orchestrator (`knowledge/agent/pipeline.py`)

```python
class PipelineStatus(str, Enum):
    ANSWERED            = "answered"
    ABSTAINED_RETRIEVAL = "abstained_retrieval"   # Layer 1 gate
    ABSTAINED_CITATION  = "abstained_citation"    # Layer 2 gate
    ABSTAINED_JUDGE     = "abstained_judge"       # Layer 3 gate

class RAGResponse(BaseModel):
    answer: str
    status: PipelineStatus
    confidence: float | None           # judge confidence; None on abstentions
    citations: list[Citation] | None   # None on abstentions
    low_confidence_warning: bool       # True when verdict == "partial"
    pipeline_latency_ms: dict[str, int]  # {"retrieval": 120, "generation": 450, "judge": 80}
    # Cost fields — always populated; 0.0 for local Ollama models
    estimated_cost_usd: float          # total estimated cost for this request
    model_tier_used: str               # "nano" | "small" | "large" — what the router selected
    prompt_tokens: int                 # total input tokens across all LLM calls in pipeline
    completion_tokens: int             # total output tokens
    cache_hit: str | None              # "l2" | "l3" | None — which cache served this response
    # Observability
    request_id: str                    # UUID — correlates with logs, Langfuse trace, audit_events
    trace_url: str | None              # Langfuse trace URL (None when langfuse_enabled=False)
    # abstention fields (populated only on abstain)
    abstention_layer: int | None       # 1, 2, or 3
    abstention_reason: str | None
```

Abstention responses use fixed, corpus-configurable strings (not LLM-generated) — fast, deterministic, and safe from hallucination in the error path itself.

#### Hook Integration

Every gate fires its own hook point so observers and custom policies can intercept without touching pipeline logic:

| Gate outcome | Hook fired | HookContext additions |
|---|---|---|
| Layer 1 pass | `POST_RETRIEVE` | `aggregate_confidence`, `results` |
| Layer 1 abstain | `ON_VALIDATION_FAIL` | `abstention_layer=1`, `aggregate_confidence` |
| Layer 2 pass | `POST_LLM` | `generation_result`, `citation_check` |
| Layer 2 abstain | `ON_VALIDATION_FAIL` | `abstention_layer=2`, `uncited_claims` |
| Layer 3 pass | `POST_LLM` | `judge_result` |
| Layer 3 abstain | `ON_VALIDATION_FAIL` | `abstention_layer=3`, `judge_verdict`, `judge_confidence` |
| Partial answer | `POST_LLM` | `judge_verdict="partial"`, note appended |

#### Evaluation Extension

Add to `EvalResult`:

```python
# Pipeline status tracking
pipeline_status: PipelineStatus
abstention_layer: int | None        # which layer gated (1/2/3)

# Per-layer confidence values (for tuning thresholds)
retrieval_aggregate_confidence: float
citation_trustworthy: bool | None
judge_verdict: str | None
judge_confidence: float | None

# Derived quality flags
false_abstention: bool   # pipeline abstained on a gold query that has a known GT answer
                         # = the system should have answered but didn't
```

Key eval metrics to track per corpus:

| Metric | Formula | Target |
|--------|---------|--------|
| Abstention rate | `abstained / total` | < 15% on gold dataset |
| False abstention rate | `abstained_on_answerable / answerable` | < 5% |
| Layer 1 abstention share | `abstained_layer1 / abstained` | diagnoses retrieval gaps |
| Layer 2 abstention share | `abstained_layer2 / abstained` | diagnoses citation/hallucination pressure |
| Layer 3 abstention share | `abstained_layer3 / abstained` | diagnoses judge threshold calibration |
| Partial answer rate | `partial / answered` | < 20% on gold dataset |

If `false_abstention_rate > 5%` → lower `retrieval_confidence_threshold` or `judge_confidence_threshold`. If `abstention_rate > 20%` on live traffic → likely a corpus coverage problem, not a threshold problem.

#### Threshold Calibration Workflow

Thresholds are not set once and forgotten. Calibrate per corpus using the gold dataset:

1. Run eval with `retrieval_confidence_threshold = 0` (disable Layer 1 gate) to get a baseline hit rate.
2. Sweep `retrieval_confidence_threshold` from 0.5 → 3.0; plot abstention rate vs. false abstention rate. Pick the knee point.
3. Repeat for `judge_confidence_threshold` from 0.4 → 0.8.
4. Re-run after every significant ingestion batch (new docs shift the confidence distribution).

Store calibration results alongside `eval_runs` in the `eval_runs.report_json` column.

---

### Model Tiering

Route queries to the cheapest model that can answer them. Saves VRAM, reduces latency, cuts cost.

#### Tier Definitions

All tiers default to local Ollama models. Cloud model IDs are listed for reference only and are gated behind `cloud_models_enabled = True` in settings (off by default).

| Tier | Local model (default) | Cloud model (opt-in) | Use cases |
|------|----------------------|----------------------|-----------|
| `nano` | `qwen2.5:0.5b` | `claude-haiku-4-5` | Routing/intent classification, content policy check, LLM-judge first pass on citation faithfulness |
| `small` | `llama3.2:3b` | `claude-sonnet-4-6` | **Default tier** — standard RAG chat, document Q&A, summarisation, KG entity extraction (simple ontologies), LLM-judge escalation target |
| `large` | `llama3.1:70b` (q4) | `claude-opus-4-8` | Multi-hop reasoning, long/dense context windows, complex analysis, KG extraction on dense domains |

These three tiers cover **generation** — every LLM call in the pipeline (routing, chat, summarisation, judging) is one of nano/small/large. Retrieval quality itself does not depend on these tiers at all; it depends on the two encoder models below, which are separate from the LLM tiers and never swapped by the router.

#### Encoder Roles — Bi-Encoder vs. Cross-Encoder

Retrieval uses two distinct encoder architectures, not LLMs, at two different stages of the pipeline:

| Role | Type | Default model | Where it runs | Used for |
|------|------|---------------|----------------|----------|
| Embedding | **Bi-encoder** | `nomic-embed-text` (768-dim) | Ollama | Ingestion — embeds every chunk once at write time. Retrieval — embeds the query once per request. |
| Reranker | **Cross-encoder** | `BAAI/bge-reranker-base` | Local, via `sentence-transformers` (not Ollama) | Retrieval only — re-scores the top-K candidates returned by hybrid search. |

**Why both exist, not just one:**
- A bi-encoder encodes the query and each chunk **independently** into the same 768-dim space. That independence is what makes the pgvector HNSW index possible — chunk vectors are precomputed once at ingestion and the ANN index scans them in `O(log n)`. This is the only architecture that scales to millions of chunks; a cross-encoder cannot be indexed because it has no per-item vector to index.
- A cross-encoder scores a `(query, chunk)` pair **jointly** — the query and chunk attend to each other inside the model — which is far more accurate but must run once per pair at query time. It cannot be precomputed and cannot scale to a full corpus scan.
- The pipeline gets both properties by using the bi-encoder for first-pass ANN retrieval over the whole corpus (ingestion-time embedding via `nomic-embed-text`, indexed by HNSW), then handing only the small `top-K` candidate set to the cross-encoder for accurate re-scoring (`CrossEncoder rerank` step — see [Retrieval Pipeline](#retrieval-pipeline) above). Swapping either model is independent — see `docs/LOCAL_LLM_GUIDE.md` §11 for embedding-model alternatives (`bge-m3`, `mxbai-embed-large`) by GPU tier.

#### LLM-as-Judge (Citation Faithfulness)

The judge is **not** a separate model — it's the `nano`/`small` LLM tiers used in an evaluation role instead of a generation role. Full mechanics (prompt, verdict schema, escalation, gate logic) live in [Layer 3 — Judge Gate](#layer-3--judge-gate-knowledgeagentjudgepy) above; summary for the tiering table:

- Runs on `nano` by default (cheapest tier, since it only classifies `supported`/`partial`/`unsupported`, not generates prose).
- Escalates to `small` when the nano verdict's own `confidence < 0.5` — one retry, not a chain.
- Independent of the citation-gate check in Layer 2: Layer 2 is deterministic string matching against `[chunk_id]` anchors; the Layer 3 judge is a semantic check that the answer's *claims* are actually supported by the context, which catches well-formatted but hallucinated citations that Layer 2 alone would miss.

#### Routing Logic (`knowledge/agent/model_router.py`)

The router runs on the `nano` model so routing overhead is < 50 ms.

```
incoming query
    │
    ▼
QueryRouter (nano model, structured output)
    │  → complexity: "simple" | "moderate" | "complex"
    │  → requires_graph: bool
    │  → estimated_context_tokens: int
    │
    ├── "simple"   + context_tokens < 512  → Tier nano   (pure retrieval, no LLM rewrite)
    ├── "moderate" + context_tokens < 4096 → Tier small
    └── "complex"  OR requires_graph       → Tier large
```

**`QueryRouter` output schema**:
```python
class RoutingDecision(BaseModel):
    complexity: Literal["simple", "moderate", "complex"]
    requires_graph: bool
    requires_multipass: bool   # triggers staged retrieval if True
    estimated_context_tokens: int
    rejected: bool             # True → query blocked before routing
    rejection_reason: str | None
```

**Forcing a tier**: clients may pass `model_tier: "small" | "large"` in the request body; the API honours it only if the JWT role includes `tier_override`. This lets power users or test harnesses bypass auto-routing.

**Fallback**: if the `nano` router call exceeds 3 s, default to `small`.

#### Configuration (`settings.py` additions)

```python
model_tier_nano: str = "qwen2.5:0.5b"
model_tier_small: str = "llama3.2:3b"
model_tier_large: str = "llama3.1:70b"
model_routing_enabled: bool = True
model_routing_timeout_s: float = 3.0
```

#### Observability

- `model_tier_selected_total{tier}` Prometheus counter — track tier distribution.
- `model_router_latency_seconds` histogram — ensure routing overhead stays < 100 ms P99.
- Log `routing_decision` as a structured field on every request trace.

---


# Integration Test Q&A Reference

Documents every integration test question, the expected answer behavior, and how to record actual answers.

All tests in this file are marked `@pytest.mark.integration` and require live services:
- **PostgreSQL** (Neon or local) with ingested NeuralFlow AI + CUAD documents
- **Ollama** running locally with `nomic-embed-text` (or configured embedding model)

```bash
# Run integration tests (requires live services)
python -m pytest rag/tests/ -m integration -v

# Run unit tests only (no live services needed)
python -m pytest rag/tests/ -m "not integration" -q
```

---

## 1. NeuralFlow AI Retrieval Tests

**File**: `rag/tests/test_rag_agent.py` — `TestRetrieverQueries`

These verify that the hybrid retriever (BM25 + vector + RRF) surfaces the correct documents.

| Test | Query | Expected source file(s) | Expected content terms |
|------|-------|------------------------|------------------------|
| `test_company_overview_query` | "What does NeuralFlow AI do?" | `company-overview`, `mission-and-goals` | "neuralflow", "ai", "automation", "enterprise", "workflow" |
| `test_team_structure_query` | "How many engineers work at the company?" | any | "engineer", "team", "employee", "staff" — ideally numbers 47 or 18 |
| `test_benefits_query` | "What is the PTO policy?" | `team-handbook` | "pto", "time off", "vacation", "leave", "days" |
| `test_technology_stack_query` | "What technologies and tools does the company use?" | any | ≥2 of: "slack", "notion", "linear", "api", "ai", "automation", "cloud" |

---

## 2. NeuralFlow AI Gold Dataset — Retrieval Metrics

**File**: `rag/tests/test_retrieval_metrics.py` — `TestRetrievalMetrics`

Measures Hit Rate, MRR, Precision, Recall, NDCG at K∈{1,3,5} against a 10-query gold set.

### Gold Dataset

| # | Query | Relevant source filename stems |
|---|-------|-------------------------------|
| 1 | "What does NeuralFlow AI do?" | `company-overview`, `mission-and-goals` |
| 2 | "What is the PTO policy?" | `team-handbook` |
| 3 | "What is the learning budget for employees?" | `team-handbook` |
| 4 | "What technologies and architecture does the platform use?" | `technical-architecture-guide` |
| 5 | "What is the company mission and vision?" | `mission-and-goals` |
| 6 | "GlobalFinance Corp loan processing success story" | `client-review-globalfinance`, `Recording4` |
| 7 | "How many employees work at NeuralFlow AI?" | `company-overview`, `team-handbook` |
| 8 | "What is DocFlow AI and how does it process documents?" | `Recording2` |
| 9 | "Q4 2024 business results and performance review" | `q4-2024-business-review` |
| 10 | "implementation approach and playbook" | `implementation-playbook` |

### Minimum Pass Thresholds (K=5, hybrid search)

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| Hit Rate@5 | ≥ 0.60 | ≥60% of queries find a relevant doc in top-5 |
| MRR@5 | ≥ 0.40 | Mean reciprocal rank ≥ 0.40 |
| Precision@5 | ≥ 0.15 | ~1 relevant result per 5 returned |
| Recall@5 | ≥ 0.40 | ≥40% of all relevant docs surfaced |
| NDCG@5 | ≥ 0.40 | Normalised discounted cumulative gain |

---

## 3. RAG Agent Integration Tests

**File**: `rag/tests/test_rag_agent.py` — `TestRAGAgentIntegration`

These run the full Pydantic AI agent (LLM + tool calls) and verify the response content.

| Test | Query | Expected response contains |
|------|-------|---------------------------|
| `test_agent_run_simple_query` | "What does NeuralFlow AI specialize in? Keep your answer brief." | "ai", "automation", "enterprise", "workflow", or "intelligence" |
| `test_agent_run_specific_query` | "How many employees does NeuralFlow AI have? Just give me the number." | A number (ideally 47), or acknowledgement it wasn't found |
| `test_agent_run_benefits_query` | "What is the learning budget for employees at NeuralFlow AI?" | "$", "budget", "learning", "development", or "training"; ideally "$2,500" |
| `test_agent_run_pto_query` | "How many PTO days do employees get at NeuralFlow AI?" | "pto", "time off", "vacation", "days", "unlimited", or "leave" |

**Known fact**: NeuralFlow AI has **47 employees**, **18 engineers**, **$2,500 learning budget**, **unlimited PTO**.

---

## 4. Search Result Quality Tests

**File**: `rag/tests/test_rag_agent.py` — `TestSearchResultQuality`

| Test | Query | Expected behavior |
|------|-------|------------------|
| `test_results_have_required_fields` | "company overview" | ≥1 result; every result has non-null `chunk_id`, `document_id`, `content`, `similarity > 0`, `document_title`, `document_source` |
| `test_relevance_scoring` | exact="NeuralFlow AI automation" vs vague="company business" | Both queries return ≥1 result |

---

## 5. Audio Transcription Retrieval Tests

**File**: `rag/tests/test_rag_agent.py` — `TestAudioTranscription`

Tests verify that Whisper-transcribed audio recordings are indexed and retrievable.

| Test | Query | Expected source | Expected content terms |
|------|-------|-----------------|------------------------|
| `test_audio_docflow_query` | "What is DocFlow AI and how does it process documents?" | `Recording2` | "docflow", "document processing", "ocr", "extract" |
| `test_audio_globalfinance_story` | "Tell me about the Global Finance Corp success story from the recording" | `Recording4` | "global finance", "globalfinance", "loan", "processing" |
| `test_audio_platform_technology` | "What LLMs and technology does the platform use?" | `Recording3` | ≥1 of: "openai", "anthropic", "llm", "enterprise", "platform" |
| `test_audio_company_intro` | "NeuralFlow AI intelligent automation introduction" | `Recording1` | "neuralflow", "automation", "transform", "business" |

---

## 6. Legal Corpus — Contract Type Retrieval

**File**: `rag/tests/test_legal_retrieval.py` — `TestLegalRetrievalMetrics`, `TestLegalSpotChecks`

Tests against 509 CUAD contracts. Relevance is determined by whether the result's `document_source` path contains the expected contract-type string.

### Legal Gold Dataset

| # | Query | Relevant contract type | Corpus size |
|---|-------|----------------------|-------------|
| 1 | "exclusive distributor rights in territory minimum purchase obligations" | `Distributor` | 31 contracts |
| 2 | "co-branding marketing and distribution agreement brand license" | `Co_Branding` | 21 contracts |
| 3 | "franchise fee royalty payments territory license obligations" | `Franchise` | 15 contracts |
| 4 | "IT outsourcing services data processing operations" | `Outsourcing` | 16 contracts |
| 5 | "supply agreement purchase orders product specifications delivery" | `Supply` | 24 contracts |
| 6 | "software license grant non-exclusive perpetual right to use" | `License` | 40 contracts |
| 7 | "consulting services independent contractor professional fees statement of work" | `Consulting` | 11 contracts |
| 8 | "authorized reseller agreement sales territory commission" | `Reseller` | 8 contracts |
| 9 | "strategic alliance partnership joint marketing collaboration" | `Alliance`, `Collaboration` | 13 contracts |
| 10 | "service agreement professional services scope of work SLA" | `Service` | 37 contracts |

### Minimum Pass Thresholds (K=5)

| Metric | Threshold | Note |
|--------|-----------|------|
| Hit Rate@5 | ≥ 0.70 | Lower precision than NeuralFlow due to terminology overlap across 509 contracts |
| MRR@5 | ≥ 0.45 | |
| Precision@5 | ≥ 0.10 | |

### Spot-Check Tests

| Test | Query | Expected: ≥1 result in top-5 matches |
|------|-------|--------------------------------------|
| `test_distributor_query_returns_distributor_contract` | "exclusive distributor rights in territory minimum purchase obligations" | source contains "distributor" |
| `test_franchise_query_returns_franchise_contract` | "franchise fee royalty payments territory license obligations" | source contains "franchise" |
| `test_software_license_query_returns_license_contract` | "software license grant non-exclusive perpetual right to use" | source contains "license" |
| `test_supply_query_returns_supply_contract` | "supply agreement purchase orders product specifications delivery" | source contains "supply" |
| `test_top_result_is_legal_document` | "indemnification clause liability limitation consequential damages" | source contains "legal" path |

---

## 7. Corpus Isolation Tests

**File**: `rag/tests/test_legal_retrieval.py` — `TestCorpusIsolation`

Verify that queries surface the right document corpus, not cross-contaminate.

| Test | Query | Expected |
|------|-------|----------|
| `test_legal_query_returns_legal_docs` | "governing law indemnification termination clause" | ≥3 of top-5 are legal docs |
| `test_legal_parties_query_returns_legal_docs` | "parties to the agreement licensor licensee" | ≥60% of top-10 are legal docs |
| `test_technical_query_skews_neuralflow` | "AI platform enterprise workflow automation SaaS" | ≥1 NeuralFlow doc in top-10 |

---

## 8. Search Type Comparison Tests

**File**: `rag/tests/test_legal_retrieval.py` — `TestLegalSearchTypes`

| Test | What it checks |
|------|---------------|
| `test_all_search_types_achieve_minimum_hit_rate` | hybrid, semantic, text each achieve Hit Rate@5 ≥ 0.4 on the legal gold set |
| `test_hybrid_not_worse_than_semantic` | hybrid Hit Rate@5 within 15 percentage points of semantic |

---

## 9. Agent Flow Tests

**File**: `rag/tests/test_agent_flow.py` — `TestAgentFlow`

End-to-end tests that stream through the full Pydantic AI agent flow (UserPromptNode → ModelRequestNode → CallToolsNode).

| Test | Query | Expected response |
|------|-------|------------------|
| `test_agent_flow_with_tool_call` | "What is the PTO policy at NeuralFlow?" | Contains "pto", "time off", "vacation", "leave", "days", or "policy" |
| `test_agent_flow_no_verbose` | "Hello, how are you?" | Non-empty response string |

---

## Recording Actual Answers

### Hybrid KG Q&A results
```bash
python -m pytest rag/tests/test_hybrid_kg_retrieval.py -m integration \
    --record-answers -v
# Writes: docs/qa_results/hybrid_kg_results.json
```

### General integration test output
```bash
# Run with -s to see full log output including queries and results
python -m pytest rag/tests/test_rag_agent.py -m integration -v -s \
    --log-cli-level=INFO
```

### Last recorded metrics
Update this table after each integration test run:

| Date | Hit Rate@5 (NeuralFlow) | Hit Rate@5 (Legal) | MRR@5 (NeuralFlow) | Notes |
|------|------------------------|-------------------|-------------------|-------|
| 2026-04-27 | ≥0.60 (passing) | ≥0.70 (passing) | ≥0.40 (passing) | 297 passed, 12 skipped |
| 2026-05-02 | ≥0.60 (passing) | ≥0.70 (passing) | ≥0.40 (passing) | 270 unit + 31 integration passed; fixed: hybrid guardrail bug (RRF scores not cosine), context "Source" assertion, empty-response check for llama3.2:3b |

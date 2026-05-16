# Task Progress — Docling + LightRAG + RAG-Anything

Task list: `claude_instructions.txt`
Rule: A → B → C → D sequentially. One task at a time. Run tests after each task. Save progress after each task.

---

## Task A — Docling: COMPLETE (2026-05-16)

### A.0 — faq.md updates ✅
- Added TOC
- Added: Docling Architecture & Internals, Internal Workflow (ASCII pipeline diagram), Customization & Tuning, Image Handling & Local VLMs
- Added: Docling Chunking Failures section (5 failure types + quick test)

### A.1 — Download demo documents ✅
- Script: `download_demo_documents.py`
- Documents saved to `documents/`:
  - `attention_is_all_you_need.pdf` — two-column mixing (arxiv 1706.03762)
  - `bert_paper.pdf` — two-column mixing (arxiv 1810.04805)
  - `nist_sp800_53.pdf` — header/footer contamination (NIST SP 800-53 Rev 5)
  - `q4_financial_report.pdf` — table splitting (copied from rag/documents/)

### A.2 — Produce bad chunks ✅
- Script: `produce_bad_chunks.py`
- Output: `output/docling/bad_chunks/` (JSON per doc + summary.json)
- Results:

| Document | Chunks | Flagged |
|---|---|---|
| attention_is_all_you_need.pdf | 67 | 2 |
| bert_paper.pdf | 97 | 1 |
| nist_sp800_53.pdf | 1998 | 135 |
| q4_financial_report.pdf | 25 | 0 |

### A.3 — Produce good chunks ✅
- Script: `produce_good_chunks.py`
- Output: `output/docling/good_chunks/` (JSON per doc + summary.json)
- Fixes applied:
  - `TableFormerMode.ACCURATE` — ✅ works
  - `do_cell_matching=False` — ✅ works
  - `repeat_table_header=True` — ✅ works (table header injected into every continuation chunk)
  - Doc tree mutation to strip headers/footers — ❌ did not work (Docling 2.x internal ref structure differs from assumed API)
- TODO (revisit): post-hoc chunk filtering for header/footer contamination — filter chunks by short length + all-caps or page-number pattern after chunking

### A.4 — Document all failures + fixes in faq.md ✅
- All scenarios from `docling_failures.md` ported to `faq.md`
- Scripts, results table, and worked/didn't-work status all documented in `faq.md`

---

## Task B — Docling + LightRAG: COMPLETE (2026-05-16)

### B.1-7 — faq.md LightRAG section ✅
Documented in `faq.md` under "LightRAG Architecture and Internals":
- Internal pipeline (ASCII diagram: chunking → LLM extraction → graph + vector storage → query)
- Graph ontology: flat open-ended property graph, 11 default entity types, binary undirected relationships
- Full LLM prompts: entity extraction system/user prompt, gleaning pass, description summarisation, keyword extraction
- Local LLM recommendations: qwen2.5:14b (best), qwen2.5:7b, llama3.1:8b, mistral:7b
- All configurables: chunk_token_size, max_gleaning, max_async, ENTITY_TYPES, etc.
- Context window management: minimum 4096 tokens, formula for chunk_token_size limit, what happens on overflow
- PostgreSQL schema: all 11 LIGHTRAG_* tables with key DDL
- Apache AGE usage: nodes=entities, edges=relationships, pgvector for entry points, AGE for traversal
- Query modes: naive / local / global / hybrid
- Limitations: images lost, table quality poor, format drift on small models, hallucinated relations
- Scalability: ~50K chunks safe, 100K+ bottlenecks on AGE traversal + entity merging

### B.8 — Python script + tests ✅
- Script: `lightrag_demo.py`
- Tests: `test_lightrag_demo.py` (12 unit tests pass; 5 integration tests require live services)
- Backend: pgvector + AGE co-installed on PG16 (port 5433, legal_graph)
  - NOTE: pgvector was not on the AGE container — installed manually via apt on the container
- Output: `output/lightrag/demo_results_*.json`

**Demo results (qwen2.5:14b via Ollama):**

| Case | Expected | What happened |
|---|---|---|
| Clean prose | works | Transformer→BERT relationship correctly retrieved ✅ |
| Multi-hop relationships | works | LightRAG→PostgreSQL dependencies correctly traversed ✅ |
| Figure placeholder | fails | LLM answered from parametric knowledge, not document — hallucinated ✅ |
| Markdown table | fails | Wrong answer: said Deep-Att+PosUnk (39.2) best EN-FR; correct is Transformer big (41.0) ✅ |
| Column-mixed chunk | fails | Wrong affiliation: attributed Aidan Gomez to Google Brain (he's Univ of Toronto) ✅ |

---

## Task C — Docling + RAG-Anything: NOT STARTED

Same steps 1–7 as Task B.

---

## Task D — NL2SQL: NOT STARTED

1. Clean up `nl2sql_overall_system_design.md` into a clear concise system design doc
2. Add FAQs to `nls2sql_faq.md`

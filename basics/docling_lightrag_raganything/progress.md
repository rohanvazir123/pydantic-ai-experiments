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

## Task B — Docling + LightRAG: NOT STARTED

Steps to do (from `claude_instructions.txt`):
1. Clear concise architecture/internals document
2. Key internal pieces + graph ontology
3. KG generation from docling chunks — document the LLM prompt it uses
4. Which local LLM + configurables + context window management
5. Limitations and failure points, scalability
6. How it stores KG in postgres apache age tables
7. Document everything in `faq.md` (new section + update TOC)
8. Python script showing where it works and where it fails + tests

**NOTE:** All backend storage uses postgres only: pgvector + tsvector + pg apache age.

---

## Task C — Docling + RAG-Anything: NOT STARTED

Same steps 1–7 as Task B.

---

## Task D — NL2SQL: NOT STARTED

1. Clean up `nl2sql_overall_system_design.md` into a clear concise system design doc
2. Add FAQs to `nls2sql_faq.md`

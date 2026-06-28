# Chunking Strategy — Answers

## Q6. How do you decide chunk size? Failure modes of too small vs too large.

**Answer:**

Chunk size is the most consequential parameter in RAG and the one most teams get wrong by defaulting to a library default (typically 512 or 1024 tokens) without measuring the impact.

**The fundamental tension:**
- Small chunks → high retrieval precision (the retrieved chunk is relevant) but low context sufficiency (the chunk does not contain enough surrounding information for the LLM to generate a complete answer)
- Large chunks → high context sufficiency but low retrieval precision (the relevant sentence is buried in an irrelevant 2,000-token block, and the LLM's attention is diluted)

**Failure modes of chunks too small (< 128 tokens):**
- The retrieved chunk contains the answer but lacks the surrounding context needed to interpret it. Example: a chunk containing "The deadline is 30 days" with no reference to what the deadline applies to.
- Retrieval returns many chunks from the same section of a document, wasting k on redundant information.
- Table rows or list items get split mid-entry, producing malformed chunks that confuse both the embedding model and the LLM.

**Failure modes of chunks too large (> 1024 tokens):**
- The relevant sentence is a small fraction of the chunk. The embedding captures the overall topic of the chunk, not the specific sentence. Retrieval brings in the chunk but the LLM must search through it.
- Token budget consumption: 20 chunks × 1,024 tokens = 20,480 tokens of context. At $15/million input tokens, 10,000 queries/day costs $3,072/day just for context. Chunk size directly multiplies token cost.
- The lost-in-the-middle problem is amplified: a relevant sentence at position 800 of a 1024-token chunk may not receive adequate LLM attention.

**How to decide:**

*Step 1 — Measure retrieval precision at multiple chunk sizes:* Run your evaluation set at chunk sizes of 128, 256, 512, and 1024. For each, compute: what percentage of queries have the answer in the top-k retrieved chunks? This tells you the minimum chunk size where retrieval is reliable.

*Step 2 — Measure answer quality at the retrieval-sufficient chunk sizes:* For the chunk sizes that achieve acceptable retrieval precision, measure answer quality (faithfulness, completeness). Larger chunks often improve answer quality up to a point.

*Step 3 — Set per-document-type chunk sizes:* A single chunk size for all document types is a mistake:

| Document type | Recommended chunk size | Reason |
|--------------|----------------------|--------|
| FAQ / policy | 256–512 tokens | Self-contained Q&A units |
| Technical documentation | 512–768 tokens | Needs surrounding context |
| Legal contracts | 768–1024 tokens | Dense, cross-referential |
| News articles | 256–512 tokens | Self-contained paragraphs |
| Code files | Function/class boundary | Semantic unit is the function |
| Audio transcripts | 512–768 tokens (with overlap) | No natural paragraph breaks |

**Token cost implication of chunk size:**
Every token in retrieved chunks is a paid input token. With a 256-token chunk and k=10, you consume 2,560 tokens of context per query. With 1024-token chunks and k=10, you consume 10,240 tokens — 4× the cost. At scale, this difference is significant. Smaller chunks with a higher-precision retrieval pipeline are almost always cheaper and often more accurate.

---

## Q7. Semantic chunking vs fixed-size vs recursive character splitting — when does each win?

**Answer:**

**Fixed-size chunking (e.g., every 512 tokens with N-token overlap):**

*How it works:* Split the document at a fixed token count, with overlap between consecutive chunks to preserve boundary context.

*Wins when:* Documents are homogeneous and unstructured (transcripts, logs, plain text). Consistent chunk size simplifies index management and cost prediction. Fast to implement and computationally cheap.

*Fails when:* Splits occur mid-sentence, mid-table, or mid-code-block. A 512-token split on a legal document might cut a clause in half: "The company shall not be liable for damages..." gets split after "liable" and the obligation context is lost. The overlap mitigates but does not eliminate this.

**Recursive character splitting:**

*How it works:* Tries to split on natural separators in priority order: `\n\n` (paragraph), `\n` (line), `. ` (sentence), ` ` (word), `""` (character). Falls back to character splitting only when necessary.

*Wins when:* Documents have natural paragraph structure (most prose). Preserves semantic units better than fixed-size with very little added cost. This is the right default for most text corpora.

*Fails when:* Documents are structured (tables, code) — paragraph-level splits may be too coarse. PDF extraction produces inconsistent whitespace, making `\n\n` an unreliable separator.

**Semantic chunking:**

*How it works:* Embed each sentence. Find boundaries where adjacent sentence embeddings have a large cosine distance — these are topic boundaries. Group sentences within a topic into a single chunk.

*Wins when:* Documents mix multiple topics within a paragraph (common in business documents: a paragraph about Q3 revenue that pivots mid-way to discuss headcount). Semantic boundaries are more meaningful than paragraph breaks. Chunks are more topically coherent, which improves embedding quality.

*Costs:* Embed every sentence during chunking (not just at query time) — this is O(n_sentences) embedding calls for the entire corpus. For a 500,000-document corpus, this is expensive upfront but pays back in retrieval quality. Chunk sizes become variable — harder to predict token budget.

*Fails when:* Documents are already well-structured (clear headings, defined sections) — the sentence boundary detection adds cost with minimal gain.

**Recommendation by corpus type:**
- Plain text, transcripts: recursive character splitting with 512 tokens + 50 token overlap
- Business documents (PDFs, DOCX): semantic chunking
- Code: function/class boundary splitting (AST-based)
- FAQ / structured content: split on QA pairs explicitly

---

## Q8. Retrieved chunk has the answer but lacks context — heading is in the previous chunk, table is in the next.

**Answer:**

This is the context window boundary problem, and it has solutions at both chunking time and retrieval time.

**Solution 1 — Chunk overlap:**
Include the last N tokens of the previous chunk at the start of every chunk. Typical: 10–20% overlap (50–100 tokens for a 512-token chunk). The heading from the previous chunk appears at the start of the next chunk.

*Cost:* Overlap increases storage by 10–20% and increases the chance of retrieval returning two near-duplicate chunks (the same content appears at the end of chunk N and the start of chunk N+1). Use deduplication at context assembly time.

**Solution 2 — Sliding window indexing with parent-child retrieval:**
Index small child chunks for retrieval precision, but expand to larger parent chunks for context when a child chunk is retrieved.

```
Document → Split into large parent chunks (1,024 tokens)
         → Each parent split into small child chunks (128 tokens)
         → Index child embeddings for retrieval
         → When a child is retrieved, return its parent chunk to the LLM
```

The child chunk finds the exact relevant sentence. The parent chunk provides surrounding context — the heading, the preceding sentence, the table caption. This is one of the most effective patterns in production RAG.

*Cost:* Larger context (parent chunks are 1,024 tokens vs 128-token child). Higher input token cost. But higher answer quality, so often worthwhile.

**Solution 3 — Metadata injection at chunk time:**
For every chunk, extract and prepend: the document title, section heading, subsection heading, and page number. These are not part of the chunk's semantic content but provide essential context.

```
[Source: Employee Handbook 2024 | Section: Leave Policy | Subsection: Parental Leave]
Employees who have been with the company for at least 12 months are eligible...
```

Even if the chunk is retrieved in isolation, the heading metadata tells the LLM (and the user via citation) exactly where this passage belongs.

**Solution 4 — Table-aware retrieval:**
Tables are the hardest case. A table cell out of context is meaningless. Solutions:
- Convert tables to prose at ingestion time ("Row 1: Q1 2024, Revenue: $4.2M, Margin: 34%")
- Index each table as a single chunk regardless of size
- Store the table as structured JSON alongside the text chunk and inject it separately when a table-related query is detected

---

## Q9. Heterogeneous corpus — PDFs with tables, DOCX, HTML, audio transcripts, code. How does chunking differ?

**Answer:**

A uniform chunking strategy applied to heterogeneous documents is one of the most common RAG mistakes. Each document type has a different natural semantic unit.

**PDFs with embedded tables and figures:**

*Challenge:* PDF text extraction (via PyPDF2, pdfminer, pymupdf) often breaks tables into garbled text sequences, loses column alignment, and misses figure captions.

*Strategy:* Use a document intelligence library (Azure Document Intelligence, AWS Textract, Docling, Unstructured.io) that understands PDF structure. Extract tables separately as structured data (CSV or JSON representation), not as raw text. Index table data with a table-specific representation that preserves row/column semantics.

**DOCX files:**

*Challenge:* DOCX has rich structure (headings, styles, tables, numbered lists) that plain text extraction discards. python-docx can parse the structure, but most pipelines flatten it.

*Strategy:* Use DOCX structure as natural chunk boundaries: each heading-delimited section becomes a chunk. Tables are extracted separately. Numbered lists are kept together as a single chunk (do not split a 10-point list mid-list).

**HTML pages:**

*Challenge:* HTML contains navigation menus, footers, cookie banners, and sidebar content alongside the main article. Naive extraction includes all of it.

*Strategy:* Use a content extractor (trafilatura, newspaper3k, Readability) to extract the main article content, stripping boilerplate. Then apply semantic or recursive chunking to the clean text. Use HTML heading tags (`<h1>`, `<h2>`) as natural section boundaries.

**Audio transcripts:**

*Challenge:* Transcripts have no paragraph structure. Speaker changes and topic shifts are the natural boundaries. A word-boundary split at 512 tokens cuts mid-sentence routinely.

*Strategy:* Use speaker-turn boundaries as primary split points. If the transcript has timestamps, add them to chunk metadata. Apply a sliding window with 20% overlap to handle topic continuity across speaker turns.

**Code files:**

*Challenge:* A 512-token split of a Python file cuts mid-function. The function is the semantic unit, not the token count.

*Strategy:* AST-based splitting — split at function and class boundaries using the language's AST parser. Each function is a chunk. Include the class definition in the metadata of each method chunk. For very long functions (> 1,024 tokens), add a docstring summary as the chunk header.

**Token cost implication of heterogeneous processing:**
Document intelligence APIs (Azure, AWS Textract) cost $0.001–$0.015 per page. For 500,000 documents averaging 10 pages each, this is $5,000–$75,000 for initial ingestion. Apply document intelligence only to document types that require it (PDF with tables). Plain DOCX and HTML can be parsed with cheaper open-source libraries.

---

## Q10. How do you handle document hierarchy — headings, subsections, footnotes — in chunking?

**Answer:**

Losing document hierarchy is one of the most significant quality losses in naive chunking pipelines. Hierarchy carries meaning: a fact under "Limitations" means something different from the same fact under "Benefits."

**Hierarchical chunk representation:**

Store hierarchy as chunk metadata, not as part of the chunk text:

```json
{
  "chunk_id": "doc_042_chunk_017",
  "content": "Employees are entitled to 20 days of annual leave per calendar year.",
  "metadata": {
    "document_title": "Employee Handbook 2024",
    "document_id": "doc_042",
    "section_path": ["HR Policies", "Leave Policy", "Annual Leave"],
    "page_number": 34,
    "chunk_position": 3,
    "total_chunks_in_section": 8
  }
}
```

The `section_path` tells the LLM (and the user) the hierarchical context without consuming tokens in the chunk body.

**Hierarchical indexing — two-level retrieval:**

Index at two levels: the section level (coarse) and the chunk level (fine). At query time:
1. First retrieve relevant sections by section embedding
2. Within the matched sections, retrieve relevant chunks

This mirrors how a human would use a document: go to the right chapter, then find the right paragraph. Coarse-to-fine retrieval reduces false positives from chunks that are topically related but hierarchically wrong ("annual leave" under "Benefits" is different from "annual leave" under "Termination Policy").

**Footnotes:**

Footnotes are often the most precise source of technical detail (legal citations, technical specifications, exception conditions). Naive chunking drops them or attaches them to the wrong chunk.

*Strategy:* Index footnotes as separate chunks with metadata linking them to the parent chunk. When the parent chunk is retrieved, include associated footnotes in the context. A query about edge cases or exceptions will often need the footnote more than the body text.

**What breaking hierarchy costs:**
A retrieval system that ignores hierarchy will correctly retrieve a passage about "termination conditions" but not know whether it refers to termination of a contract or termination of employment. The hierarchy makes the disambiguation trivial. Without it, the LLM must infer from context — often incorrectly for ambiguous passages.

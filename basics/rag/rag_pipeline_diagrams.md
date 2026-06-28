# RAG Pipeline — Text Diagrams

## Table of Contents

- [Ingestion Pipeline](#ingestion-pipeline)
  - [Document Parsing Detail](#document-parsing-detail)
  - [Image-to-Text Routing](#image-to-text-routing)
  - [Chunking Strategy by Document Type](#chunking-strategy-by-document-type)
  - [Index Architecture](#index-architecture)
- [Retrieval Pipeline](#retrieval-pipeline)
  - [Query Understanding Detail](#query-understanding-detail)
  - [Hybrid Retrieval with RRF](#hybrid-retrieval-with-rrf)
  - [Context Assembly Detail](#context-assembly-detail)
  - [Generation and Post-Processing](#generation-and-post-processing)
- [End-to-End Flow (Ingestion + Retrieval)](#end-to-end-flow-ingestion--retrieval)

---

## Ingestion Pipeline

```
Raw Documents
(PDF · DOCX · HTML · audio · images · code)
                │
                ▼
┌───────────────────────────────────────────────┐
│  STAGE 1: PARSING                             │
│                                               │
│  PDF (simple)   → pdfminer / pypdf            │
│  PDF (complex)  → Docling / Azure DI          │
│  DOCX           → python-docx (preserves      │
│                   heading styles, tables)     │
│  HTML           → trafilatura                 │
│                   (strips nav/ads/footer)     │
│  Audio          → Whisper transcription       │
│  Code           → raw text (AST-aware split)  │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│  STAGE 2: DATA CLEANING                       │
│                                               │
│  Boilerplate removal                          │
│    → page headers / footers                   │
│    → copyright notices                        │
│    → nav menus (HTML)                         │
│    → repeated legal disclaimers               │
│                                               │
│  Encoding fixes (ftfy)                        │
│    → mojibake · ligatures (ﬁ→fi)              │
│    → hyphenated line breaks (infor-\n→inform) │
│    → zero-width chars                         │
│                                               │
│  OCR quality check (if scanned)               │
│    → score < 0.65 → re-OCR with cloud API     │
│                                               │
│  Document deduplication                       │
│    → SHA-256 content hash vs registry         │
│    → skip if hash matches existing doc        │
│                                               │
│  Chunk deduplication                          │
│    → MinHash LSH  (threshold: 0.98 sim)       │
│    → removes boilerplate repeated across docs │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│  STAGE 3: IMAGE-TO-TEXT CONVERSION            │
│                                               │
│  Classify image type                          │
│         │                                     │
│    ┌────┴──────────────────────────────┐      │
│    │                                   │      │
│  Scanned text              Charts/Graphs      │
│  → Tesseract OCR           → GPT-4o vision   │
│  → cloud OCR if            → structured desc  │
│    quality < 0.65            (all data points)│
│                                               │
│  Tables-as-images          Formulas           │
│  → GPT-4o vision           → Mathpix / GPT-4o │
│  → JSON + markdown         → LaTeX + plain    │
│    (both stored)             English          │
│                                               │
│  Diagrams / infographics   Logos / decorative │
│  → GPT-4o vision           → SKIP            │
│  → exhaustive prose          (do not index)  │
│    description                                │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│  STAGE 4: METADATA EXTRACTION                 │
│                                               │
│  Document level                               │
│    title · authors · date · version           │
│    document_type · access_level · tags        │
│                                               │
│  Section level                                │
│    chapter · section · subsection            │
│    section_path: ["HR", "Leave", "Parental"]  │
│    heading depth (H1/H2/H3)                   │
│                                               │
│  Paragraph level                              │
│    page_number · paragraph_index             │
│    char_offset_start · char_offset_end        │
│    (enables UI source highlighting)           │
│                                               │
│  Table level                                  │
│    table_id · caption · column headers        │
│    row_count · representation type            │
│                                               │
│  Image level                                  │
│    figure_id · figure_number · caption        │
│    image_type · raw_image_path                │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│  STAGE 5: CHUNKING                            │
│                                               │
│  Strategy per document type:                  │
│                                               │
│  FAQ / policy         256–512 tokens          │
│  Technical docs       512–768 tokens          │
│  Legal contracts      768–1,024 tokens        │
│  Code files           function / class boundary│
│  Audio transcripts    512–768 + 20% overlap   │
│                                               │
│  Parent-child indexing:                       │
│    large parent chunks (1,024 tokens)         │
│    small child chunks  (128 tokens)           │
│    child indexed for retrieval precision      │
│    parent returned for context sufficiency    │
│                                               │
│  Every chunk carries full metadata:           │
│    section_path · page · paragraph · offsets  │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│  STAGE 6: EMBEDDING                           │
│                                               │
│  Batch embed chunks (100–500 per API call)    │
│  Model: domain-appropriate                   │
│    general: text-embedding-3-small            │
│    scientific: nomic-embed-text               │
│    domain fine-tuned: if > 10K labeled pairs  │
│                                               │
│  Tables: embed prose representation           │
│          store markdown separately for LLM    │
│  Images: embed conversion_text                │
│          store raw image path for UI          │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│  STAGE 7: INDEXING                            │
│                                               │
│  Vector store  → HNSW index                   │
│  (Qdrant / pgvector / Pinecone)               │
│    namespace per tenant (zero cross-pollute)  │
│    metadata filters: access_level, doc_type   │
│                                               │
│  BM25 index  → sparse retrieval               │
│  (Elasticsearch / OpenSearch / built-in)      │
│                                               │
│  Metadata store → PostgreSQL                  │
│    full citation schema                       │
│    document registry (version + hash)         │
│    chunk → parent chunk mapping               │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
              Searchable, Attributable Index
```

---

### Document Parsing Detail

```
Incoming document
       │
       ├── Is it a PDF?
       │       │
       │       ├── Simple (text layer present, single column)
       │       │     → pdfminer / pypdf   (fast, cheap)
       │       │
       │       └── Complex (scanned, multi-column, tables)
       │             → Docling / Azure Document Intelligence
       │               (preserves layout, extracts tables)
       │
       ├── Is it DOCX?
       │     → python-docx
       │       extract heading hierarchy from styles
       │       extract tables with column headers intact
       │
       ├── Is it HTML?
       │     → trafilatura (main content extraction)
       │       strips: nav · ads · footers · cookie banners
       │       retains: article body · tables · headings
       │
       ├── Is it audio (MP3/WAV/M4A)?
       │     → Whisper transcription
       │       add speaker-turn metadata if diarization available
       │       chunk at speaker boundaries + 20% overlap
       │
       └── Is it code (py/js/ts/java)?
             → AST parser (language-specific)
               one chunk per function / class
               include docstring + signature in chunk header
```

---

### Image-to-Text Routing

```
Image extracted from document
            │
            ▼
    ┌───────────────────────┐
    │  Image Classifier     │
    │  (rule-based or ML)   │
    └───────┬───────────────┘
            │
    ┌───────┼────────────────────────────────────┐
    │       │                                    │
    ▼       ▼                                    ▼
SCANNED   TABLE                              CHART / GRAPH
  TEXT    AS IMG                             BAR/LINE/PIE
    │       │                                    │
    ▼       ▼                                    ▼
Tesseract  GPT-4o vision                    GPT-4o vision
  OCR     → JSON (headers,                  → Title, axes,
            rows, values)                     all data points,
score<0.65  + markdown                        key trend
→ cloud     (both stored)                     annotation
  OCR                                         
    │       │                                    │
    └───────┴────────────────────────────────────┘
            │
    ┌───────┼───────────────────┐
    │       │                   │
    ▼       ▼                   ▼
FORMULA  DIAGRAM /           LOGO /
 (LaTeX) INFOGRAPHIC         DECORATIVE
    │       │                   │
    ▼       ▼                   ▼
Mathpix /  GPT-4o vision      SKIP
GPT-4o   → exhaustive prose   (not indexed)
→ LaTeX    description of
  + plain  all elements,
  English  arrows, labels
```

---

### Chunking Strategy by Document Type

```
Document type → Strategy → Typical chunk size
─────────────────────────────────────────────
FAQ / Q&A      split on Q&A pairs       256–512 tokens
               (each Q+A = one chunk)

Policy / HR    recursive char split     256–512 tokens
               at paragraph breaks      + 10% overlap

Technical doc  semantic chunking        512–768 tokens
               at topic boundaries      + 15% overlap

Legal contract large overlap sliding   768–1,024 tokens
               window (dense           + 25% overlap
               cross-references)

Scientific     section-aware split      512–768 tokens
paper          at headings             + 10% overlap

Code file      AST-based               per function / class
               (never split mid-fn)    (variable size)

Audio          speaker-turn aware      512–768 tokens
transcript     + timestamp metadata    + 20% overlap

Tables         prose: for embedding    one chunk per table
               markdown: for LLM       (never split mid-table)
               (both stored)

Images         conversion text         one chunk per image
               from Stage 3            + figure metadata
```

---

### Index Architecture

```
┌─────────────────────────────────────────────────────┐
│                   INDEX LAYER                       │
│                                                     │
│  ┌──────────────────┐  ┌───────────────────────┐   │
│  │   VECTOR STORE   │  │    BM25 / SPARSE       │   │
│  │   (HNSW index)   │  │    (Elasticsearch)     │   │
│  │                  │  │                        │   │
│  │  chunk_id        │  │  chunk_id              │   │
│  │  embedding[768]  │  │  term frequencies      │   │
│  │  tenant_ns       │  │  IDF weights           │   │
│  │  access_level    │  │  tenant_id filter      │   │
│  │  doc_id          │  │                        │   │
│  └──────────────────┘  └───────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │         METADATA STORE (PostgreSQL)          │   │
│  │                                              │   │
│  │  documents table:                            │   │
│  │    id · title · authors · version            │   │
│  │    content_hash · access_level · tenant_id   │   │
│  │    ingested_at · source_url                  │   │
│  │                                              │   │
│  │  chunks table:                               │   │
│  │    id · document_id · content               │   │
│  │    section_path · page_number               │   │
│  │    paragraph_index · char_offsets           │   │
│  │    parent_chunk_id (for parent-child)        │   │
│  │    embedding_id (FK to vector store)         │   │
│  │                                              │   │
│  │  tables table:                               │   │
│  │    id · document_id · caption               │   │
│  │    column_headers · page_number             │   │
│  │    markdown_repr · prose_repr               │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## Retrieval Pipeline

```
User Query
      │
      ▼
┌─────────────────────────────────────────────────┐
│           QUERY UNDERSTANDING                   │
│                                                 │
│  Intent: ANALYTICAL / CONVERSATIONAL / OOS      │
│  Ambiguity scoring per dimension                │
│  Multi-turn context resolution                  │
│    ("that" → prior query result)                │
│    ("their" → entity from 3 turns ago)          │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┴────────────┐
        │                       │
        ▼                       ▼
┌───────────────┐     ┌──────────────────────────┐
│    CACHE      │     │   QUERY TRANSFORMATION    │
│    LOOKUP     │     │   (optional, in parallel) │
│               │     │                           │
│  key =        │     │  HyDE: generate a hypo-   │
│  normalize(q) │     │  thetical answer, embed it │
│  + schema_v   │     │                           │
│  + auth_scope │     │  Expansion: add synonyms  │
│               │     │  and related terms        │
│  HIT → done   │     │                           │
│  MISS → cont  │     │  Multi-query: N rephrase  │
└───────────────┘     │  variants, merge results  │
                      └────────────┬──────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────┐
│     HYBRID RETRIEVAL  (parallel execution)      │
│                                                 │
│  ┌──────────────────┐  ┌────────────────────┐   │
│  │  DENSE RETRIEVAL │  │  SPARSE RETRIEVAL  │   │
│  │  (semantic)      │  │  (BM25 keyword)    │   │
│  │                  │  │                    │   │
│  │  embed query     │  │  tokenize query    │   │
│  │  ANN search      │  │  TF-IDF scoring    │   │
│  │  → top-50        │  │  → top-50          │   │
│  │  handles:        │  │  handles:          │   │
│  │  synonymy        │  │  exact IDs         │   │
│  │  paraphrase      │  │  product codes     │   │
│  │  semantic sim    │  │  proper nouns      │   │
│  └────────┬─────────┘  └─────────┬──────────┘   │
│           │                      │               │
│           └──────────┬───────────┘               │
│                      ▼                           │
│         RECIPROCAL RANK FUSION (RRF)             │
│         score = 1/(60+rank_dense)                │
│               + 1/(60+rank_sparse)               │
│         → top-50 unified candidates              │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│      ACCESS CONTROL FILTER                      │
│      (enforced INSIDE vector store query)       │
│                                                 │
│  namespace = tenant_{tenant_id}                 │
│  filter:    document_id IN user_permissions     │
│                                                 │
│  NOT a post-retrieval filter —                  │
│  applied before results are returned            │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│           CROSS-ENCODER RERANKING               │
│                                                 │
│  Input: (query, chunk_content) pairs            │
│  Reads both simultaneously → fine-grained match │
│  Handles: negation · conditionals               │
│           precise factual alignment             │
│                                                 │
│  top-50 → top-10                                │
│  +5–15pp recall vs embedding similarity alone  │
│  Latency: 100–400ms                             │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│         RETRIEVAL FAILURE DETECTION             │
│                                                 │
│  max cosine similarity < 0.3?                   │
│    → no relevant chunk found                    │
│    → retry with query transformation            │
│    → if still fails: "I don't have enough info" │
│    → log to corpus gap queue                    │
│                                                 │
│  empty result set?                              │
│    → unambiguous retrieval failure              │
│    → do NOT generate from parametric knowledge  │
└──────────────────────┬──────────────────────────┘
                       │ chunks retrieved
                       ▼
┌─────────────────────────────────────────────────┐
│           PARENT CHUNK EXPANSION                │
│           (if parent-child indexing used)       │
│                                                 │
│  child chunk retrieved (128 tokens)             │
│    → fetch parent chunk (1,024 tokens)          │
│    → use parent for context assembly            │
│  preserves surrounding headings,                │
│  table captions, preceding sentences            │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│           CONTEXT ASSEMBLY                      │
│                                                 │
│  1. Deduplication                               │
│     cosine sim > 0.95 → remove duplicate        │
│                                                 │
│  2. Book-end ordering                           │
│     rank 1 → position 1 (first in context)     │
│     rank 2 → position last                     │
│     ranks 3–9 → middle                         │
│     (mitigates lost-in-the-middle effect)       │
│                                                 │
│  3. Token budget enforcement                   │
│     k × avg_chunk_tokens ≤ budget              │
│     drop lowest-scored chunks if over budget    │
│                                                 │
│  4. Source attribution injection               │
│     [Source: {title} | {section} | Page {n}]   │
│     prepended to each chunk                     │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│           LLM GENERATION                        │
│                                                 │
│  System prompt:                                 │
│    "Answer using ONLY the provided context.    │
│     Cite every factual claim with [^n]."       │
│                                                 │
│  Input:  assembled context + user query         │
│  Output: answer with inline citations           │
│                                                 │
│  Settings:                                      │
│    temperature = 0  (grounds to context)        │
│    streaming = true (user sees tokens arrive)   │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│        POST-GENERATION  (async)                 │
│                                                 │
│  Faithfulness check                             │
│    NLI entailment per claim → context           │
│    score < 0.8 → flag for review                │
│                                                 │
│  Citation verification                          │
│    cited chunk actually supports claim?         │
│    cosine(claim, cited_chunk) < 0.5 → flag     │
│                                                 │
│  Cache write                                    │
│    key = (query_cluster, schema_version,        │
│            auth_scope)                          │
│                                                 │
│  Implicit feedback logging                      │
│    re-query within 60s? → negative signal       │
│    export / share? → positive signal            │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
                      User
```

---

### Query Understanding Detail

```
Raw user query
      │
      ├── Multi-turn? resolve context first
      │     "that" → prior SQL / result set
      │     "their" → entity from entity registry
      │     "and also" → extend prior query
      │     semantic sim < 0.4 → topic shift → reset
      │
      ▼
┌─────────────────────────────────┐
│  INTENT CLASSIFICATION          │
│                                 │
│  Stage A: rule-based (< 1ms)    │
│    greeting / one word → CONV   │
│    "what is GDP of France" → OOS│
│                                 │
│  Stage B: embedding (< 20ms)    │
│    cosine to known query types  │
│    ANALYTICAL / CONV / OOS      │
│    if all sim < 0.5 → Stage C   │
│                                 │
│  Stage C: LLM classifier (500ms)│
│    only for novel edge cases    │
│    JSON: {intent, confidence}   │
└─────────────────────────────────┘
      │ ANALYTICAL
      ▼
┌─────────────────────────────────┐
│  SCHEMA COVERAGE CHECK          │
│  (< 30ms, embedding-based)      │
│                                 │
│  Extract concepts from query    │
│  Find each in schema index      │
│  Missing concepts? → OOS        │
│  All found? → IN_SCOPE          │
│  Some found? → PARTIAL          │
└─────────────────────────────────┘
      │ IN_SCOPE
      ▼
┌─────────────────────────────────┐
│  DATA AVAILABILITY CHECK        │
│  (post-retrieval)               │
│                                 │
│  table exists but 0 rows?       │
│  → "data not ingested yet"      │
│  date range has no data?        │
│  → "data available from X to Y" │
└─────────────────────────────────┘
```

---

### Hybrid Retrieval with RRF

```
Query: "SOC 2 Type II audit report Q3 2023"

DENSE RETRIEVAL                     SPARSE (BM25) RETRIEVAL
(semantic similarity)               (keyword match)
───────────────────────             ────────────────────────
Rank 1: Security Summary      →     Rank 1: SOC 2 Q3 2023 Report  ←── correct doc
Rank 2: SOC 2 Overview        →     Rank 2: SOC 2 Q2 2023 Report
Rank 3: SOC 2 Q3 2023 Report  ─┐   Rank 3: Security Summary
Rank 4: Annual Security 2023   │    Rank 4: Audit Procedures
Rank 5: Compliance Checklist   │

        RRF score = 1/(60 + rank_dense) + 1/(60 + rank_sparse)

SOC 2 Q3 2023 Report: 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226  ← RANK 1
Security Summary:      1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226  ← tie
SOC 2 Overview:        1/(60+2) + 1/(60+5) = 0.01613 + 0.01538 = 0.03151

RRF correctly surfaces the exact-match document to rank 1
even though dense-only ranked it 3rd.
```

---

### Context Assembly Detail

```
Top-10 reranked chunks
          │
          ▼
┌──────────────────────────────────┐
│  STEP 1: DEDUPLICATION           │
│                                  │
│  chunk_3 ──cosine──► chunk_7     │
│  similarity = 0.97 > threshold   │
│  → remove lower-ranked duplicate │
│                                  │
│  Remaining: 8 unique chunks      │
└──────────────────────┬───────────┘
                       │
                       ▼
┌──────────────────────────────────┐
│  STEP 2: BOOK-END ORDERING       │
│                                  │
│  Reranker scores:                │
│  chunk_2: 0.91  → position 1     │
│  chunk_5: 0.88  → position 8     │(last)
│  chunk_1: 0.84  → position 2     │
│  chunk_8: 0.81  → position 3     │
│  ...                             │
│  (most relevant at start + end,  │
│   middle gets less attention)    │
└──────────────────────┬───────────┘
                       │
                       ▼
┌──────────────────────────────────┐
│  STEP 3: TOKEN BUDGET CHECK      │
│                                  │
│  8 chunks × avg 512 tokens       │
│  = 4,096 input tokens            │
│                                  │
│  Budget: 6,000 tokens for context│
│  4,096 < 6,000 → all fit         │
│  If over budget → drop lowest    │
│    scored chunks until fits      │
└──────────────────────┬───────────┘
                       │
                       ▼
┌──────────────────────────────────┐
│  STEP 4: SOURCE ATTRIBUTION      │
│                                  │
│  [Source: Employee Handbook 2024 │
│   | Section 3.4 | Page 34]      │
│  Employees are entitled to 20... │
│  ─────────────────────────────── │
│  [Source: HR Policy Update 2024  │
│   | Section 1.1 | Page 2]       │
│  Contract employees are not...   │
│  ─────────────────────────────── │
└──────────────────────┬───────────┘
                       │
                       ▼
              Context ready for LLM
```

---

### Generation and Post-Processing

```
Context (assembled, attributed)
+ User query
       │
       ▼
┌────────────────────────────────────────┐
│  LLM GENERATION (streaming)            │
│                                        │
│  "Employees are entitled to 20 days   │
│   of annual leave [^1]. Contract      │
│   employees are excluded from this    │
│   benefit [^2]."                      │
│                                        │
│  [^1] Employee Handbook 2024, §3.4    │
│  [^2] HR Policy Update 2024, §1.1    │
└──────────────────────┬─────────────────┘
                       │ stream to user
                       │ (async below)
                       ▼
┌─────────────────────────────────────────────────────┐
│                 POST-GENERATION                     │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  Faithfulness check (NLI-based, fast)         │   │
│  │                                              │   │
│  │  Decompose answer into atomic claims          │   │
│  │  Claim 1: "20 days annual leave"             │   │
│  │    → check against chunk_2 → ENTAILED ✓     │   │
│  │  Claim 2: "contract employees excluded"      │   │
│  │    → check against chunk_5 → ENTAILED ✓     │   │
│  │  score: 2/2 = 1.0 (fully grounded)          │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  Citation verification                        │   │
│  │                                              │   │
│  │  [^1] → chunk_2: does it support claim 1?   │   │
│  │    cosine(claim_1, chunk_2) = 0.91 ✓        │   │
│  │  [^2] → chunk_5: does it support claim 2?   │   │
│  │    cosine(claim_2, chunk_5) = 0.87 ✓        │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  Cache write + feedback logging               │   │
│  │    cache_key → answer (TTL: 15 min)          │   │
│  │    log: query · retrieved_ids · score        │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## End-to-End Flow (Ingestion + Retrieval)

```
INGEST TIME                              QUERY TIME
───────────                              ──────────

Raw Document                             User Question
     │                                        │
     ▼                                        ▼
  Parse                               Normalize + Classify
  Clean                                        │
  Image→Text                          Cache Lookup ──HIT──► Answer
  Metadata                                     │ MISS
  Chunk                               Query Transform
     │                                        │
     ▼                                        ▼
  Embed                            Dense + Sparse Retrieve
  chunks                                (parallel)
     │                                        │
     ▼                                   RRF Merge
  Store in ◄────────────────────────────────  │
  Vector DB    shared index                   │
  BM25 index                            Access Filter
  Metadata DB                                 │
                                        Rerank (cross-encoder)
                                              │
                                        Parent Expand
                                              │
                                        Context Assemble
                                              │
                                        Generate + Cite
                                              │
                                        Faithfulness Check
                                              │
                                         ► Answer
```

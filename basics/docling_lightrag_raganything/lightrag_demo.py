"""
LightRAG demo: where it works and where it fails.

Backend: PostgreSQL (pgvector on port 5434) + Apache AGE (port 5433)
LLM:     qwen2.5:14b via Ollama
Embed:   nomic-embed-text via Ollama (pulled automatically if missing)

Storage tables (all LIGHTRAG_* prefix) are created automatically in rag_db.
Graph is stored in Apache AGE on legal_graph database.

Run:
    python basics/docling_lightrag_raganything/lightrag_demo.py

Output saved to:
    basics/docling_lightrag_raganything/output/lightrag/
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from datetime import datetime

import httpx
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_BASE = "http://localhost:11434"
LLM_MODEL = "qwen2.5:14b"
EMBED_MODEL = "nomic-embed-text:latest"
EMBED_DIM = 768

# Both pgvector and AGE are on port 5433 (legal_graph).
# pgvector was installed on the AGE container so one PG instance handles everything.
PG_HOST = "localhost"
PG_PORT = "5433"
PG_USER = "age_user"
PG_PASS = "age_pass"
PG_DB = "legal_graph"
GRAPH_NAME = "lightrag_demo"

WORKING_DIR = Path(__file__).parent / "output" / "lightrag" / "working"
OUT_DIR = Path(__file__).parent / "output" / "lightrag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

async def ollama_embed(texts: list[str]) -> np.ndarray:
    """Batch embed via Ollama nomic-embed-text."""
    async with httpx.AsyncClient(timeout=60) as client:
        vecs = []
        for text in texts:
            r = await client.post(
                f"{OLLAMA_BASE}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
            )
            r.raise_for_status()
            vecs.append(r.json()["embedding"])
        return np.array(vecs, dtype=np.float32)


async def ollama_llm(prompt: str, system_prompt: str | None = None, **_) -> str:
    """Single LLM call via Ollama."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(
            f"{OLLAMA_BASE}/v1/chat/completions",
            json={"model": LLM_MODEL, "messages": messages},
            headers={"Authorization": "Bearer ollama"},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Pull embed model if needed
# ---------------------------------------------------------------------------

async def ensure_embed_model() -> bool:
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(f"{OLLAMA_BASE}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            if any(EMBED_MODEL.split(":")[0] in m for m in models):
                return True
            print(f"  Pulling {EMBED_MODEL} ...")
            r = await client.post(
                f"{OLLAMA_BASE}/api/pull",
                json={"name": EMBED_MODEL},
                timeout=300,
            )
            return r.status_code == 200
        except Exception as e:
            print(f"  WARNING: could not ensure embed model: {e}")
            return False


# ---------------------------------------------------------------------------
# Build LightRAG with PG backend
# ---------------------------------------------------------------------------

async def build_rag() -> object:
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from lightrag.kg.shared_storage import initialize_pipeline_status

    # Inject PG connection config via env vars (lightrag reads these)
    # Single PG instance with both pgvector and AGE (port 5433)
    os.environ["POSTGRES_HOST"] = PG_HOST
    os.environ["POSTGRES_PORT"] = PG_PORT
    os.environ["POSTGRES_USER"] = PG_USER
    os.environ["POSTGRES_PASSWORD"] = PG_PASS
    os.environ["POSTGRES_DATABASE"] = PG_DB
    os.environ["POSTGRES_WORKSPACE"] = "lightrag_demo"
    os.environ["AGE_GRAPH_NAME"] = GRAPH_NAME

    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    embedding_func = EmbeddingFunc(
        embedding_dim=EMBED_DIM,
        max_token_size=8192,
        func=ollama_embed,
    )

    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=ollama_llm,
        embedding_func=embedding_func,
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        graph_storage="PGGraphStorage",
        doc_status_storage="PGDocStatusStorage",
        chunk_token_size=800,
        chunk_overlap_token_size=100,
        entity_extract_max_gleaning=1,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CASES: list[dict] = [
    # ── WORKS ──────────────────────────────────────────────────────────────
    {
        "id": "works_clean_prose",
        "label": "WORKS — Clean prose with clear entities",
        "expected": "pass",
        "text": """
The Transformer architecture was introduced by Ashish Vaswani and colleagues at Google Brain in 2017,
in the paper "Attention Is All You Need". The model replaces recurrent layers entirely with
multi-head self-attention mechanisms, allowing parallel computation across sequence positions.
The Transformer achieved state-of-the-art results on English-to-German and English-to-French
machine translation benchmarks. BERT, developed by Google AI Language in 2018, extended the
Transformer by using bidirectional pre-training on large text corpora, enabling strong performance
on question answering, sentiment analysis, and named entity recognition tasks.
""".strip(),
        "query": "What is the relationship between the Transformer and BERT?",
        "query_mode": "hybrid",
    },
    {
        "id": "works_relationships",
        "label": "WORKS — Multi-hop entity relationships",
        "expected": "pass",
        "text": """
PostgreSQL is an open-source relational database system developed by the PostgreSQL Global
Development Group. The pgvector extension adds vector similarity search to PostgreSQL, enabling
storage and querying of high-dimensional embeddings. Apache AGE is another PostgreSQL extension
that adds openCypher graph query support, allowing graph traversal using Cypher syntax directly
inside PostgreSQL. LightRAG uses both pgvector for embedding-based retrieval and Apache AGE for
knowledge graph traversal, making PostgreSQL the only required infrastructure component.
""".strip(),
        "query": "How does LightRAG use PostgreSQL?",
        "query_mode": "local",
    },
    # ── FAILS ──────────────────────────────────────────────────────────────
    {
        "id": "fails_figure_placeholder",
        "label": "FAILS — Figure placeholder (image content lost by Docling)",
        "expected": "fail",
        "text": """
[Figure 3]
[Image content not available — Docling extracted figure as FigureItem with no VLM configured]
Caption: Multi-head attention mechanism showing queries Q, keys K, and values V being
projected into h parallel attention heads, each computing scaled dot-product attention.
""".strip(),
        "query": "How does multi-head attention work mechanically?",
        "query_mode": "local",
        "failure_reason": (
            "Docling extracts the caption but not the diagram. LightRAG sees only the caption "
            "text and placeholder markers. Extracted entities will be 'Figure 3', 'queries Q', "
            "'keys K', 'values V' — descriptive labels, not structural relationships. "
            "A query asking HOW it works mechanically cannot be answered from caption text alone."
        ),
    },
    {
        "id": "fails_markdown_table",
        "label": "FAILS — Markdown table (structure lost as prose)",
        "expected": "fail",
        "text": """
| Model | Parameters | BLEU EN-DE | BLEU EN-FR | Training Cost |
|---|---|---|---|---|
| Transformer (base) | 65M | 27.3 | 38.1 | $0.8K |
| Transformer (big) | 213M | 28.4 | 41.0 | $6.6K |
| ByteNet | - | 23.75 | - | - |
| Deep-Att + PosUnk | - | - | 39.2 | - |
| MoE | 2.0B | 26.03 | 40.56 | $2.0K |
""".strip(),
        "query": "Which model has the best BLEU score on English to French translation?",
        "query_mode": "local",
        "failure_reason": (
            "The LLM sees the markdown table as prose. It extracts entities like 'Transformer', "
            "'ByteNet', 'MoE' but treats numeric values (BLEU scores) as entity descriptions, "
            "not structured data. Comparative queries ('which model has the BEST score') require "
            "the LLM to reason over all rows simultaneously, which the graph cannot support — "
            "each entity is stored independently with no 'ranked above' relationship."
        ),
    },
    {
        "id": "fails_column_mixed",
        "label": "FAILS — Column-mixed chunk from bad Docling extraction",
        "expected": "fail",
        "text": (
            "Ashish Vaswani ∗ Google Brain avaswani@google.com Noam Shazeer ∗ "
            "Google Brain noam@google.com\nLlion Jones ∗ Google Research llion@google.com "
            "Niki Parmar ∗ Google Research nikip@google.com Aidan N. Gomez ∗ † "
            "University of Toronto aidan@cs.toronto.edu Jakob Uszkoreit ∗ Google Research "
            "usz@google.com Łukasz Kaiser ∗ Google Brain lukaszkaiser@google.com"
        ),
        "query": "Who are the authors from Google Brain?",
        "query_mode": "local",
        "failure_reason": (
            "This is the actual column-mixed author block from attention_is_all_you_need.pdf "
            "(from bad_chunks output). The LLM sees interleaved author names and emails from "
            "two columns. It may extract authors correctly but relationships between authors "
            "and their institutions will be garbled — email addresses and affiliation markers "
            "(the asterisk symbol) appear as entity names."
        ),
    },
]


# ---------------------------------------------------------------------------
# Run a single case
# ---------------------------------------------------------------------------

async def run_case(rag, case: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"{case['label']}")
    print(f"{'='*60}")

    if "failure_reason" in case:
        print(f"  Expected failure reason: {case['failure_reason'][:120]}...")

    # Ingest
    print(f"  Ingesting text ({len(case['text'])} chars) ...")
    try:
        await rag.ainsert(case["text"])
        ingested = True
        print("  Ingestion: OK")
    except Exception as e:
        ingested = False
        print(f"  Ingestion ERROR: {e}")

    # Query
    result_text = None
    if ingested:
        print(f"  Query ({case['query_mode']}): {case['query']!r}")
        try:
            from lightrag.base import QueryParam
            result = await rag.aquery(
                case["query"],
                param=QueryParam(mode=case["query_mode"]),
            )
            result_text = str(result)
            print(f"  Answer ({len(result_text)} chars): {result_text[:300]}...")
        except Exception as e:
            result_text = f"QUERY ERROR: {e}"
            print(f"  {result_text}")

    return {
        "id": case["id"],
        "label": case["label"],
        "expected": case["expected"],
        "failure_reason": case.get("failure_reason"),
        "text_length": len(case["text"]),
        "query": case["query"],
        "query_mode": case["query_mode"],
        "ingested": ingested,
        "answer": result_text,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("LightRAG Demo — PostgreSQL Backend")
    print(f"  pgvector + AGE: {PG_HOST}:{PG_PORT}/{PG_DB}  graph={GRAPH_NAME}")
    print(f"  (pgvector and AGE co-installed on same PG16 instance)")
    print(f"  LLM:      {LLM_MODEL}")
    print(f"  Embed:    {EMBED_MODEL}")

    print("\nChecking embed model ...")
    await ensure_embed_model()

    print("\nInitialising LightRAG with PG backend ...")
    rag = await build_rag()
    print("  Ready.")

    results = []
    for case in CASES:
        result = await run_case(rag, case)
        results.append(result)

    # Save output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"demo_results_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "OK" if r["ingested"] else "INGEST_FAIL"
        answer_preview = (r["answer"] or "")[:80].replace("\n", " ")
        print(f"  [{r['expected'].upper():4}] {r['id']:<35} {status}")
        print(f"         Answer: {answer_preview}...")
    print(f"\nFull results saved to: {out_path}")

    # Save a latest symlink
    latest = OUT_DIR / "demo_results_latest.json"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_path.name)


if __name__ == "__main__":
    asyncio.run(main())

"""Generate an evaluation testset for the RAG v2 retrieval pipeline.

Pulls chunks from the database, asks the configured LLM to produce diverse
evaluation questions from each chunk, then writes two output files:

  tests/retrieval/eval_questions.json  — machine-readable (loaded by test runner)
  tests/retrieval/eval_questions.md   — human-readable for review / editing

Usage:
  uv run python scripts/generate_eval_questions.py
  uv run python scripts/generate_eval_questions.py --limit 20 --per-chunk 3
  uv run python scripts/generate_eval_questions.py --corpus default --tenant default

The JSON schema matches GOLD_DATASET in test_retrieval_metrics.py so the
generated questions can be loaded directly by TestGeneratedEvalDataset.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import asyncpg
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(dotenv_path=ROOT / ".env", override=False)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

OUT_JSON = ROOT / "tests/retrieval/eval_questions.json"
OUT_MD   = ROOT / "tests/retrieval/eval_questions.md"

# Max concurrent LLM calls (Ollama is single-threaded per model)
_CONCURRENCY = 2
_LLM_TIMEOUT_S = 30.0


# ── Output schema ─────────────────────────────────────────────────────────────

class GeneratedQuestion(BaseModel):
    question:              str
    expected_answer:       str   # 1-3 sentences; correct answer the retriever should support
    ground_truth_keywords: list[str]   # 3-5 lowercase words that must appear in a correct chunk
    question_type:         Literal["simple", "reasoning", "multi_context", "conditional"]


class QuestionBatch(BaseModel):
    questions: list[GeneratedQuestion]


# ── LLM agent ─────────────────────────────────────────────────────────────────

_GENERATOR_PROMPT = """\
You are building an evaluation testset for a Retrieval-Augmented Generation (RAG) system.

You will be given an excerpt from a document. Your task is to generate {n} evaluation questions
based on that excerpt.

Rules:
- Every question must be clearly answerable from the excerpt alone.
- Cover different cognitive levels:
    simple       — direct fact lookup ("What is X?", "How many Y?")
    reasoning    — requires reading multiple sentences to synthesise ("Why does X lead to Y?")
    multi_context — spans several concepts in the passage
    conditional  — depends on a condition stated in the text ("If X, then what happens?")
- expected_answer: 1-3 concise sentences that correctly answer the question.
- ground_truth_keywords: 3-5 lowercase words or short phrases that MUST appear in
  any correctly retrieved chunk (used for hit-rate evaluation).
- Do NOT generate questions about topics not present in the excerpt.
- Do NOT generate trivially obvious yes/no questions.

Return valid JSON matching the schema exactly.\
"""


def _make_agent(settings: Any) -> Agent:  # type: ignore[type-arg]
    provider = OpenAIProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
    model = OpenAIChatModel(settings.model_tier_small, provider=provider)
    ms: dict[str, Any] = {}
    if settings.llm_provider == "ollama":
        ms = {"extra_body": {"num_ctx": 4096}}
    return Agent(  # type: ignore[call-overload]
        model,
        output_type=QuestionBatch,
        model_settings=ms,
    )


# ── DB helpers ────────────────────────────────────────────────────────────────

async def fetch_chunks(
    conn: asyncpg.Connection,
    corpus_id: str,
    tenant_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Return sampled chunks — at most `limit`, spread across documents."""
    rows = await conn.fetch(
        """
        SELECT
            c.content,
            d.source_path,
            d.title
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.corpus_id = $1
          AND d.tenant_id = $2
          AND length(c.content) > 100
        ORDER BY d.source_path, random()
        LIMIT $3
        """,
        corpus_id, tenant_id, limit,
    )
    return [dict(r) for r in rows]


def _source_stem(source_path: str) -> str:
    """Extract a short stem for relevant_sources matching, e.g. 'team-handbook'."""
    name = Path(source_path).stem
    # Strip UUIDs or numeric suffixes that come from ingestion
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return name


# ── Question generation ───────────────────────────────────────────────────────

async def generate_for_chunk(
    agent: Agent,  # type: ignore[type-arg]
    content: str,
    source_path: str,
    n: int,
    sem: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    prompt = _GENERATOR_PROMPT.replace("{n}", str(n))
    user_msg = f"Document excerpt from '{Path(source_path).name}':\n\n{content[:2000]}"

    async with sem:
        try:
            result = await asyncio.wait_for(
                agent.run(user_msg, instructions=prompt),
                timeout=_LLM_TIMEOUT_S,
            )
            batch: QuestionBatch = result.output
        except TimeoutError:
            logger.warning("Timeout generating questions for %s — skipping", source_path)
            return []
        except Exception as exc:
            logger.warning("LLM error for %s: %s — skipping", source_path, exc)
            return []

    stem = _source_stem(source_path)
    out = []
    for q in batch.questions:
        out.append({
            "query":            q.question,
            "relevant_sources": [stem],
            "ground_truth":     [kw.lower() for kw in q.ground_truth_keywords],
            "question_type":    q.question_type,
            "source_document":  Path(source_path).name,
            "expected_answer":  q.expected_answer,
        })
    return out


def _dedup(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for q in questions:
        key = q["query"].lower().strip()[:80]
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out


# ── Output writers ────────────────────────────────────────────────────────────

def write_json(questions: list[dict[str, Any]]) -> None:
    # Strip expected_answer from the JSON (test runner doesn't need it)
    rows = [{k: v for k, v in q.items()} for q in questions]
    OUT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    logger.info("Wrote %d questions → %s", len(questions), OUT_JSON)


def write_markdown(questions: list[dict[str, Any]], corpus_id: str, tenant_id: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        "# RAG v2 Evaluation Testset",
        "",
        f"Generated: {ts}  ",
        f"Corpus: `{corpus_id}` / Tenant: `{tenant_id}`  ",
        f"Questions: {len(questions)}",
        "",
        "> Auto-generated by `scripts/generate_eval_questions.py`.  ",
        "> Review and edit as needed. Re-run to regenerate from the current corpus.",
        "",
        "---",
        "",
    ]
    for i, q in enumerate(questions, 1):
        lines += [
            f"## {i}. {q['question_type']} — {q['source_document']}",
            "",
            f"**Q:** {q['query']}  ",
            f"**Expected answer:** {q['expected_answer']}  ",
            f"**Ground truth keywords:** {', '.join(q['ground_truth'])}  ",
            f"**Relevant sources:** `{', '.join(q['relevant_sources'])}`  ",
            "",
            "---",
            "",
        ]
    OUT_MD.write_text("\n".join(lines))
    logger.info("Wrote markdown → %s", OUT_MD)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    from knowledge.config.settings import load_settings
    settings = load_settings()

    logger.info("Connecting to database…")
    conn: asyncpg.Connection = await asyncpg.connect(settings.database_url)

    try:
        chunks = await fetch_chunks(conn, args.corpus, args.tenant, args.limit)
    finally:
        await conn.close()

    if not chunks:
        logger.error(
            "No chunks found for corpus=%s tenant=%s — run 'make seed' first.",
            args.corpus, args.tenant,
        )
        sys.exit(1)

    logger.info("Fetched %d chunks across corpus '%s'", len(chunks), args.corpus)

    agent = _make_agent(settings)
    sem = asyncio.Semaphore(_CONCURRENCY)

    tasks = [
        generate_for_chunk(agent, c["content"], c["source_path"], args.per_chunk, sem)
        for c in chunks
    ]

    all_batches = await asyncio.gather(*tasks)
    questions = _dedup([q for batch in all_batches for q in batch])

    if not questions:
        logger.error("No questions generated — check Ollama is running and models are pulled.")
        sys.exit(1)

    logger.info("Generated %d unique questions (before dedup: %d)", len(questions), sum(len(b) for b in all_batches))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    write_json(questions)
    write_markdown(questions, args.corpus, args.tenant)

    print(f"\n  ✓ {len(questions)} questions saved to:")
    print(f"    {OUT_JSON}")
    print(f"    {OUT_MD}")
    print(f"\n  Run:  uv run pytest tests/retrieval/test_retrieval_metrics.py::TestGeneratedEvalDataset -v")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAG eval testset from corpus chunks")
    parser.add_argument("--corpus",    default="default", help="Corpus ID (default: default)")
    parser.add_argument("--tenant",    default="default", help="Tenant ID (default: default)")
    parser.add_argument("--limit",     type=int, default=30, help="Max chunks to sample (default: 30)")
    parser.add_argument("--per-chunk", type=int, default=2,  help="Questions per chunk (default: 2)")
    asyncio.run(main(parser.parse_args()))

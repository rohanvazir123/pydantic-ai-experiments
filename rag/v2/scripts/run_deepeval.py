"""DeepEval RAG evaluation runner.

Evaluates the full RAG pipeline against predefined test cases using five
DeepEval metrics scored by a local Ollama LLM judge:

  Faithfulness        — is the answer grounded in the retrieved context?
  AnswerRelevancy     — does the answer address the question?
  ContextualRelevancy — are the retrieved chunks on-topic?
  ContextualPrecision — are the most relevant chunks ranked highest?  (needs expected_output)
  ContextualRecall    — does the context cover the expected answer?   (needs expected_output)

Requirements:
  uv sync --extra eval
  Running services: PostgreSQL + Redis + Ollama (llama3.2:3b + nomic-embed-text)
  Ingested corpus: make seed

Usage:
  uv run python scripts/run_deepeval.py
  uv run python scripts/run_deepeval.py --judge llama3.1:70b
  uv run python scripts/run_deepeval.py --top-k 10

Output:
  evals/deepeval_results.md          — always overwritten (latest run)
  evals/deepeval_YYYY-MM-DD_HHMMSS.md — timestamped archive
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("deepeval").setLevel(logging.WARNING)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

EVALS_DIR = ROOT / "evals"
EVALS_DIR.mkdir(exist_ok=True)

TENANT_ID = "default"
CORPUS_ID = "default"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test cases
# expected_output is optional — its presence enables Contextual Precision +
# Contextual Recall (ground-truth metrics). Leave None for queries where we
# only have partial knowledge of the ideal answer.
# ---------------------------------------------------------------------------
TEST_CASES: list[dict[str, Any]] = [
    {
        "input": "What does NeuralFlow AI do?",
        "expected_output": (
            "NeuralFlow AI builds AI-powered products and provides machine learning "
            "consulting services to enterprise clients."
        ),
        "geval_criteria": (
            "The actual output must accurately describe the company's core business. "
            "Omitting either 'AI-powered products' or 'consulting services' is a failure. "
            "Vague descriptions like 'technology company' without specifics are a failure."
        ),
        "tags": ["company", "overview"],
    },
    {
        "input": "What is the PTO and leave policy?",
        "expected_output": (
            "Employees receive paid time off that accrues over the year. "
            "The policy covers vacation days, sick leave, and public holidays."
        ),
        "geval_criteria": (
            "The actual output must preserve all specific leave categories from the expected output "
            "(vacation days, sick leave, public holidays). Dropping any category or replacing "
            "specific terms with vague phrases like 'various leave types' is a failure."
        ),
        "tags": ["hr", "benefits"],
    },
    {
        "input": "Which business units performed best in Q4?",
        "expected_output": None,
        "tags": ["finance", "q4"],
    },
    {
        "input": "What technologies and tools does the engineering team use?",
        "expected_output": None,
        "tags": ["tech", "engineering"],
    },
    {
        "input": "What are the onboarding steps for new employees?",
        "expected_output": None,
        "tags": ["hr", "onboarding"],
    },
    {
        "input": "What are the company's goals and objectives for this year?",
        "expected_output": None,
        "tags": ["strategy", "goals"],
    },
    {
        "input": "How does the performance review process work?",
        "expected_output": None,
        "tags": ["hr", "performance"],
    },
]

# ---------------------------------------------------------------------------
# Ollama judge — wraps DeepEvalBaseLLM around our local Ollama endpoint
# ---------------------------------------------------------------------------

def _build_judge(model: str, base_url: str) -> Any:
    """Return a DeepEvalBaseLLM backed by the given Ollama model."""
    from deepeval.models.base_model import DeepEvalBaseLLM
    from openai import OpenAI
    from pydantic import BaseModel as PydanticBaseModel

    class OllamaJudge(DeepEvalBaseLLM):
        def __init__(self) -> None:
            self._openai = OpenAI(base_url=base_url, api_key="ollama")
            super().__init__()

        def load_model(self) -> "OllamaJudge":
            return self

        def generate(
            self,
            prompt: str,
            schema: type[PydanticBaseModel] | None = None,
        ) -> str | PydanticBaseModel:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
            if schema is not None:
                kwargs["response_format"] = {"type": "json_object"}

            resp = self._openai.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""

            if schema is None:
                return text

            # Parse JSON → schema instance; fall back to extracting first {...}
            try:
                return schema.model_validate_json(text)
            except Exception:
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    try:
                        return schema.model_validate_json(m.group())
                    except Exception:
                        pass
                # Last resort: return a minimal valid instance by coercing fields
                raise ValueError(f"Could not parse judge response as {schema.__name__}: {text[:200]}")

        async def a_generate(
            self,
            prompt: str,
            schema: type[PydanticBaseModel] | None = None,
        ) -> str | PydanticBaseModel:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.generate(prompt, schema))

        def get_model_name(self) -> str:
            return model

    return OllamaJudge()


# ---------------------------------------------------------------------------
# Pipeline runner — retriever + agent called directly (no HTTP)
# ---------------------------------------------------------------------------

async def _run_query(
    query: str,
    *,
    retriever: Any,
    pipeline: Any,
    top_k: int,
) -> dict[str, Any]:
    """Return answer, retrieval_context (full chunk texts), and latency_ms.

    Uses run_stream() (output_type=str) rather than run() (output_type=GenerationResult)
    because llama3.2:3b cannot reliably produce the nested Citation+CitationCheck JSON
    schema required by structured output mode.  The streaming path bypasses that schema.

    Retrieval context is fetched directly from the retriever *before* the pipeline call
    so we get full chunk texts (not the ≤200-char excerpts in citations).
    """
    t0 = asyncio.get_event_loop().time()

    # Full chunk content for DeepEval (citation excerpts are too short)
    results = await retriever.retrieve(
        query=query,
        corpus_ids=[CORPUS_ID],
        tenant_id=TENANT_ID,
        k=top_k,
    )
    retrieval_context: list[str] = [r.content for r in results]

    # Collect streamed answer tokens
    answer_parts: list[str] = []
    final_status = "answered"

    async for event_str in pipeline.run_stream(
        query=query,
        corpus_ids=[CORPUS_ID],
        tenant_id=TENANT_ID,
        user_id="eval",
        session_id="eval",
        model_tier="small",
        message_history=[],
    ):
        # SSE format: "data: {...}\n\n"
        data_str = event_str.removeprefix("data:").strip()
        if not data_str:
            continue
        try:
            event = json.loads(data_str)
            if "delta" in event:
                answer_parts.append(event["delta"])
            elif event.get("abstained"):
                final_status = f"abstained_{event.get('reason', 'unknown')}"
            elif "error" in event:
                final_status = "error"
        except (json.JSONDecodeError, KeyError):
            pass

    latency_ms = int((asyncio.get_event_loop().time() - t0) * 1000)
    return {
        "answer": "".join(answer_parts),
        "status": final_status,
        "retrieval_context": retrieval_context,
        "latency_ms": latency_ms,
        "confidence": None,
    }


# ---------------------------------------------------------------------------
# DeepEval metric runner
# ---------------------------------------------------------------------------

def _score_test_case(
    tc: dict[str, Any],
    pipeline_output: dict[str, Any],
    judge: Any,
) -> dict[str, Any]:
    """Run all applicable DeepEval metrics for one test case. Returns scores dict."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    answer = pipeline_output["answer"]
    retrieval_context = pipeline_output["retrieval_context"]
    expected = tc.get("expected_output")

    test_case = LLMTestCase(
        input=tc["input"],
        actual_output=answer,
        expected_output=expected,
        retrieval_context=retrieval_context,
    )

    scores: dict[str, float | None] = {}
    reasons: dict[str, str] = {}

    # Metrics that always run (no ground truth needed)
    always_metrics = [
        ("faithfulness",         FaithfulnessMetric(threshold=0.7,  model=judge, include_reason=True)),
        ("answer_relevancy",     AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=True)),
        ("contextual_relevancy", ContextualRelevancyMetric(threshold=0.6, model=judge, include_reason=True)),
    ]
    # Metrics that need expected_output
    grounded_metrics = [
        ("contextual_precision", ContextualPrecisionMetric(threshold=0.6, model=judge, include_reason=True)),
        ("contextual_recall",    ContextualRecallMetric(threshold=0.6,    model=judge, include_reason=True)),
    ] if expected else []

    for name, metric in always_metrics + grounded_metrics:
        try:
            metric.measure(test_case)
            scores[name] = round(metric.score, 3)
            reasons[name] = getattr(metric, "reason", "") or ""
        except Exception as exc:
            logger.warning("Metric %s failed for %r: %s", name, tc["input"][:60], exc)
            scores[name] = None
            reasons[name] = f"ERROR: {exc}"

    # GEval — custom correctness criteria defined per test case (opt-in)
    geval_criteria = tc.get("geval_criteria")
    if geval_criteria and expected:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCaseParams

        geval = GEval(
            name="Custom Correctness",
            criteria=geval_criteria,
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=0.7,
            model=judge,
        )
        try:
            geval.measure(test_case)
            scores["geval_correctness"] = round(geval.score, 3)
            reasons["geval_correctness"] = getattr(geval, "reason", "") or ""
        except Exception as exc:
            logger.warning("GEval failed for %r: %s", tc["input"][:60], exc)
            scores["geval_correctness"] = None
            reasons["geval_correctness"] = f"ERROR: {exc}"

    return {"scores": scores, "reasons": reasons}


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------

METRIC_THRESHOLDS = {
    "faithfulness":         0.7,
    "answer_relevancy":     0.7,
    "contextual_relevancy": 0.6,
    "contextual_precision": 0.6,
    "contextual_recall":    0.6,
    "geval_correctness":    0.7,
}

METRIC_LABELS = {
    "faithfulness":         "Faithfulness",
    "answer_relevancy":     "Answer Relevancy",
    "contextual_relevancy": "Contextual Relevancy",
    "contextual_precision": "Contextual Precision",
    "contextual_recall":    "Contextual Recall",
    "geval_correctness":    "GEval Correctness",
}


def _pass_icon(score: float | None, threshold: float) -> str:
    if score is None:
        return "—"
    return "✅" if score >= threshold else "❌"


def _build_report(
    results: list[dict[str, Any]],
    *,
    judge_model: str,
    rag_model: str,
    top_k: int,
    run_at: datetime,
) -> str:
    lines: list[str] = []

    lines.append("# DeepEval RAG Evaluation Report")
    lines.append("")
    lines.append(f"**Date:** {run_at.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Corpus:** {TENANT_ID}:{CORPUS_ID}")
    lines.append(f"**RAG model:** `{rag_model}`")
    lines.append(f"**Judge model:** `{judge_model}`")
    lines.append(f"**Top-K retrieved:** {top_k}")
    lines.append(f"**Test cases:** {len(results)}")
    lines.append("")

    # ── Aggregate summary ──
    all_metric_names = list(METRIC_LABELS)
    agg: dict[str, list[float]] = {m: [] for m in all_metric_names}
    for r in results:
        for m, v in r["scores"].items():
            if v is not None:
                agg[m].append(v)

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Avg Score | Threshold | Pass Rate | Status |")
    lines.append("|--------|-----------|-----------|-----------|--------|")
    for m in all_metric_names:
        vals = agg[m]
        if not vals:
            lines.append(f"| {METRIC_LABELS[m]} | — | {METRIC_THRESHOLDS[m]} | — | — |")
            continue
        avg = sum(vals) / len(vals)
        threshold = METRIC_THRESHOLDS[m]
        pass_rate = sum(1 for v in vals if v >= threshold) / len(vals)
        status = "✅ Pass" if avg >= threshold else "❌ Fail"
        lines.append(
            f"| {METRIC_LABELS[m]} | {avg:.3f} | {threshold} "
            f"| {pass_rate:.0%} ({sum(1 for v in vals if v >= threshold)}/{len(vals)}) "
            f"| {status} |"
        )
    lines.append("")

    # ── Per-query results ──
    lines.append("## Per-Query Results")
    lines.append("")

    for i, r in enumerate(results, 1):
        tc = r["tc"]
        po = r["pipeline_output"]
        scores = r["scores"]
        reasons = r["reasons"]

        lines.append(f"### Q{i}: {tc['input']}")
        lines.append("")

        tags = tc.get("tags", [])
        if tags:
            lines.append(f"**Tags:** {', '.join(f'`{t}`' for t in tags)}")

        status = po.get("status", "")
        confidence = po.get("confidence")
        latency = po.get("latency_ms", 0)
        lines.append(f"**Pipeline status:** `{status}`  |  **Latency:** {latency} ms  |  **Confidence:** {confidence or '—'}")
        lines.append("")

        answer = po.get("answer", "")
        lines.append(f"**Answer:**")
        lines.append(f"> {answer[:500]}{'…' if len(answer) > 500 else ''}")
        lines.append("")

        if tc.get("expected_output"):
            lines.append(f"**Expected (ground truth):**")
            lines.append(f"> {tc['expected_output']}")
            lines.append("")

        # Metric scores table
        lines.append("| Metric | Score | Threshold | Pass |")
        lines.append("|--------|-------|-----------|------|")
        for m in all_metric_names:
            if m not in scores:
                continue
            v = scores[m]
            t = METRIC_THRESHOLDS[m]
            score_str = f"{v:.3f}" if v is not None else "—"
            lines.append(f"| {METRIC_LABELS[m]} | {score_str} | {t} | {_pass_icon(v, t)} |")
        lines.append("")

        # Reasons (collapsed)
        if any(reasons.values()):
            lines.append("<details>")
            lines.append("<summary>Judge reasoning</summary>")
            lines.append("")
            for m, reason in reasons.items():
                if reason:
                    lines.append(f"**{METRIC_LABELS.get(m, m)}:** {reason}")
                    lines.append("")
            lines.append("</details>")
            lines.append("")

        # Retrieved context titles
        ctx = po.get("retrieval_context", [])
        if ctx:
            lines.append(f"**Retrieved context:** {len(ctx)} chunk(s)")
            for j, chunk in enumerate(ctx[:3], 1):
                preview = chunk[:120].replace("\n", " ")
                lines.append(f"  - Chunk {j}: {preview}…")
            if len(ctx) > 3:
                lines.append(f"  - _(+{len(ctx) - 3} more)_")
            lines.append("")

        lines.append("---")
        lines.append("")

    # ── Raw JSON (for programmatic consumption) ──
    lines.append("## Raw Scores (JSON)")
    lines.append("")
    lines.append("```json")
    raw = [
        {
            "query": r["tc"]["input"],
            "tags":  r["tc"].get("tags", []),
            "status": r["pipeline_output"].get("status"),
            "latency_ms": r["pipeline_output"].get("latency_ms"),
            "scores": r["scores"],
        }
        for r in results
    ]
    lines.append(json.dumps(raw, indent=2))
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    from knowledge.agent.pipeline import ConfidenceAwarePipeline
    from knowledge.config.settings import load_settings
    from knowledge.ingestion.embedder import Embedder
    from knowledge.retrieval.retriever import Retriever
    from knowledge.store.cache import RedisCache
    from knowledge.store.vector import PostgresHybridStore

    settings = load_settings()
    judge_model = args.judge or settings.llm_model
    rag_model   = settings.llm_model

    logger.info("Initialising stores…")
    vs    = PostgresHybridStore(settings=settings)
    cache = RedisCache(settings=settings)
    await vs.initialize()
    await cache.connect()

    embedder  = Embedder(settings=settings)
    retriever = Retriever(vector_store=vs, embedder=embedder, cache=cache, settings=settings)
    pipeline  = ConfidenceAwarePipeline(retriever=retriever, settings=settings)

    judge = _build_judge(judge_model, settings.llm_base_url)
    logger.info("Judge: %s  |  RAG model: %s  |  top_k: %d", judge_model, rag_model, args.top_k)

    results: list[dict[str, Any]] = []

    for i, tc in enumerate(TEST_CASES, 1):
        logger.info("[%d/%d] %s", i, len(TEST_CASES), tc["input"])
        try:
            pipeline_output = await _run_query(
                tc["input"],
                retriever=retriever,
                pipeline=pipeline,
                top_k=args.top_k,
            )
        except Exception as exc:
            logger.error("Pipeline failed for %r: %s", tc["input"], exc)
            pipeline_output = {
                "answer": "",
                "status": "error",
                "retrieval_context": [],
                "latency_ms": 0,
                "confidence": None,
            }

        scores_and_reasons = _score_test_case(tc, pipeline_output, judge)
        results.append({"tc": tc, "pipeline_output": pipeline_output, **scores_and_reasons})

        for metric, score in scores_and_reasons["scores"].items():
            icon = _pass_icon(score, METRIC_THRESHOLDS[metric])
            logger.info("  %-25s %s  %s", METRIC_LABELS[metric], f"{score:.3f}" if score is not None else " — ", icon)

    await vs.close()
    await cache.close()

    run_at = datetime.now(UTC)
    report = _build_report(
        results,
        judge_model=judge_model,
        rag_model=rag_model,
        top_k=args.top_k,
        run_at=run_at,
    )

    # Write latest + timestamped archive
    latest = EVALS_DIR / "deepeval_results.md"
    archive = EVALS_DIR / f"deepeval_{run_at.strftime('%Y-%m-%d_%H%M%S')}.md"

    latest.write_text(report, encoding="utf-8")
    archive.write_text(report, encoding="utf-8")

    logger.info("Report written → %s", latest)
    logger.info("Archive        → %s", archive)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepEval RAG evaluation")
    parser.add_argument(
        "--judge",
        default=None,
        help="Ollama model to use as LLM judge (default: settings.llm_model)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5)",
    )
    asyncio.run(main(parser.parse_args()))

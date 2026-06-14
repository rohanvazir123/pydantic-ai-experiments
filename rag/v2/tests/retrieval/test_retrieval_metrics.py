"""Retrieval quality evaluation for RAG v2.

Two test classes:

TestMetricFunctions  — pure unit tests for IR metric helpers, no services.
TestRetrievalMetrics — integration tests against a live PostgreSQL + Ollama
                       stack (marked @pytest.mark.integration).  Skipped
                       automatically when the database is unreachable.

Metrics computed for K in {1, 3, 5}:
  Hit Rate@K   — fraction of queries where ≥1 relevant doc appears in top-K
  MRR@K        — mean reciprocal rank of the first relevant result
  Precision@K  — mean fraction of top-K results that are relevant
  Recall@K     — mean fraction of relevant docs retrieved in top-K
  NDCG@K       — normalised discounted cumulative gain

Ground-truth answer tests verify that the retriever surfaces chunks that
actually contain the expected answer text, not just the right file.
"""

import logging
import math
import time

import pytest
import pytest_asyncio

from knowledge.config.settings import load_settings
from knowledge.ingestion.embedder import Embedder
from knowledge.ingestion.models import SearchResult
from knowledge.retrieval.retriever import Retriever
from knowledge.store.vector import PostgresHybridStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gold dataset — (query, relevant source stems, ground-truth keywords)
# ---------------------------------------------------------------------------
# relevant_sources: substring match against SearchResult.document_source
# ground_truth:     words/phrases that should appear in the retrieved content
#                   for at least one top-5 result (case-insensitive)
GOLD_DATASET: list[dict] = [
    {
        "query": "What does NeuralFlow AI do?",
        "relevant_sources": ["company-overview", "mission-and-goals"],
        "ground_truth": ["ai", "neural", "platform"],
    },
    {
        "query": "What is the PTO policy?",
        "relevant_sources": ["team-handbook"],
        "ground_truth": ["pto", "vacation", "time off"],
    },
    {
        "query": "What is the learning budget for employees?",
        "relevant_sources": ["team-handbook"],
        "ground_truth": ["learning", "budget", "training"],
    },
    {
        "query": "What technologies and architecture does the platform use?",
        "relevant_sources": ["technical-architecture-guide"],
        "ground_truth": ["architecture", "technology", "system"],
    },
    {
        "query": "What is the company mission and vision?",
        "relevant_sources": ["mission-and-goals"],
        "ground_truth": ["mission", "vision", "goal"],
    },
    {
        "query": "GlobalFinance Corp loan processing success story",
        "relevant_sources": ["client-review-globalfinance", "Recording4"],
        "ground_truth": ["globalfinance", "loan", "processing"],
    },
    {
        "query": "How many employees work at NeuralFlow AI?",
        "relevant_sources": ["company-overview", "team-handbook"],
        "ground_truth": ["employee", "team", "staff", "people"],
    },
    {
        "query": "Q4 2024 business results and performance review",
        "relevant_sources": ["q4-2024-business-review"],
        "ground_truth": ["q4", "2024", "revenue", "result", "performance"],
    },
    {
        "query": "Implementation approach and playbook",
        "relevant_sources": ["implementation-playbook"],
        "ground_truth": ["implementation", "playbook", "approach"],
    },
]

# Corpus / tenant used for all integration tests
DEFAULT_CORPUS_ID = "default"
DEFAULT_TENANT_ID = "default"

K_VALUES = [1, 3, 5]

# Minimum acceptable thresholds at K=5
THRESHOLDS_K5 = {
    "hit_rate": 0.60,
    "mrr":      0.40,
    "precision": 0.15,
    "recall":   0.40,
    "ndcg":     0.40,
}


# ---------------------------------------------------------------------------
# Pure metric helpers
# ---------------------------------------------------------------------------

def is_relevant(document_source: str, relevant_sources: list[str]) -> bool:
    src = document_source.lower()
    return any(stem.lower() in src for stem in relevant_sources)


def build_relevance_list(results: list[SearchResult], relevant_sources: list[str]) -> list[int]:
    return [int(is_relevant(r.document_source, relevant_sources)) for r in results]


def hit_rate(relevance_list: list[int]) -> float:
    return 1.0 if any(relevance_list) else 0.0


def reciprocal_rank(relevance_list: list[int]) -> float:
    for i, rel in enumerate(relevance_list):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(relevance_list: list[int], k: int) -> float:
    return sum(relevance_list[:k]) / k if k else 0.0


def recall_at_k(relevance_list: list[int], k: int, total_relevant: int) -> float:
    return sum(relevance_list[:k]) / total_relevant if total_relevant else 0.0


def ndcg_at_k(relevance_list: list[int], k: int) -> float:
    top_k = relevance_list[:k]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(top_k))
    n_relevant = sum(relevance_list)
    ideal_k = min(n_relevant, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))
    return dcg / idcg if idcg > 0 else 0.0


def compute_all_metrics(
    per_query_relevance: list[list[int]],
    per_query_total_relevant: list[int],
    k: int,
) -> dict[str, float]:
    n = len(per_query_relevance)
    if n == 0:
        return {}
    return {
        "hit_rate":  sum(hit_rate(r) for r in per_query_relevance) / n,
        "mrr":       sum(reciprocal_rank(r[:k]) for r in per_query_relevance) / n,
        "precision": sum(precision_at_k(r, k) for r in per_query_relevance) / n,
        "recall":    sum(
            recall_at_k(r, k, t)
            for r, t in zip(per_query_relevance, per_query_total_relevant)
        ) / n,
        "ndcg":      sum(ndcg_at_k(r, k) for r in per_query_relevance) / n,
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = (p / 100) * (len(sv) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (idx - lo)


def ground_truth_hit(results: list[SearchResult], keywords: list[str]) -> bool:
    """Return True if any keyword appears in any of the top-5 result chunks."""
    combined = " ".join(r.content.lower() for r in results[:5])
    return any(kw.lower() in combined for kw in keywords)


# ---------------------------------------------------------------------------
# Unit tests — no services
# ---------------------------------------------------------------------------

class TestMetricFunctions:
    def test_is_relevant_match(self) -> None:
        assert is_relevant("/path/team-handbook.md", ["team-handbook"]) is True

    def test_is_relevant_no_match(self) -> None:
        assert is_relevant("/path/team-handbook.md", ["company-overview"]) is False

    def test_is_relevant_case_insensitive(self) -> None:
        assert is_relevant("/docs/Recording4.mp3", ["recording4"]) is True

    def test_is_relevant_multiple_sources(self) -> None:
        assert is_relevant("/docs/company-overview.md", ["team-handbook", "company-overview"]) is True

    def test_hit_rate_positive(self) -> None:
        assert hit_rate([0, 1, 0]) == 1.0

    def test_hit_rate_negative(self) -> None:
        assert hit_rate([0, 0, 0]) == 0.0

    def test_reciprocal_rank_first(self) -> None:
        assert reciprocal_rank([1, 0, 0]) == pytest.approx(1.0)

    def test_reciprocal_rank_second(self) -> None:
        assert reciprocal_rank([0, 1, 0]) == pytest.approx(0.5)

    def test_reciprocal_rank_none(self) -> None:
        assert reciprocal_rank([0, 0, 0]) == 0.0

    def test_precision_all_relevant(self) -> None:
        assert precision_at_k([1, 1, 1, 0, 0], k=3) == pytest.approx(1.0)

    def test_precision_partial(self) -> None:
        assert precision_at_k([1, 0, 1, 0, 0], k=4) == pytest.approx(0.5)

    def test_recall_full(self) -> None:
        assert recall_at_k([1, 1, 0, 0], k=3, total_relevant=2) == pytest.approx(1.0)

    def test_recall_partial(self) -> None:
        assert recall_at_k([1, 0, 0, 1], k=3, total_relevant=2) == pytest.approx(0.5)

    def test_recall_zero_relevant(self) -> None:
        assert recall_at_k([0, 0, 0], k=3, total_relevant=0) == 0.0

    def test_ndcg_perfect(self) -> None:
        assert ndcg_at_k([1, 1, 0, 0], k=3) == pytest.approx(1.0)

    def test_ndcg_worst(self) -> None:
        expected = (1.0 / math.log2(4)) / (1.0 / math.log2(2))
        assert ndcg_at_k([0, 0, 1], k=3) == pytest.approx(expected)

    def test_ndcg_no_relevant(self) -> None:
        assert ndcg_at_k([0, 0, 0], k=3) == 0.0

    def test_compute_all_metrics_shape(self) -> None:
        rels = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
        metrics = compute_all_metrics(rels, [1, 1], k=5)
        assert set(metrics) == {"hit_rate", "mrr", "precision", "recall", "ndcg"}

    def test_compute_all_metrics_empty(self) -> None:
        assert compute_all_metrics([], [], k=5) == {}

    def test_percentile_median(self) -> None:
        assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == pytest.approx(3.0)

    def test_percentile_p95(self) -> None:
        vals = list(range(1, 101))
        assert percentile(vals, 95) == pytest.approx(95.05)


# ---------------------------------------------------------------------------
# Integration tests — require PostgreSQL + Ollama + ingested corpus
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRetrievalMetrics:
    """IR metric evaluation against the NeuralFlow AI gold dataset.

    Run:  pytest tests/retrieval/ -v --log-cli-level=INFO -m integration
    Skip: pytest tests/retrieval/ -v -m "not integration"
    """

    @pytest_asyncio.fixture
    async def retriever(self) -> Retriever:  # type: ignore[override]
        settings = load_settings()
        store = PostgresHybridStore(settings=settings)
        try:
            await store.initialize()
        except Exception as exc:
            pytest.skip(f"PostgreSQL unreachable: {exc}")
        count = await store.get_chunk_count(DEFAULT_CORPUS_ID, DEFAULT_TENANT_ID)
        if count == 0:
            await store.close()
            pytest.skip(
                f"No documents in corpus '{DEFAULT_CORPUS_ID}' — "
                "run 'make seed' to ingest sample data before retrieval tests"
            )
        embedder = Embedder(settings=settings)
        r = Retriever(vector_store=store, embedder=embedder, settings=settings)
        yield r
        await store.close()

    # ── Shared helpers ────────────────────────────────────────────────────────

    async def _run_gold(
        self,
        retriever: Retriever,
        k: int,
        search_type: str = "hybrid",
    ) -> tuple[list[list[int]], list[int], list[float], list[list[SearchResult]]]:
        per_query_relevance: list[list[int]] = []
        per_query_totals:    list[int]        = []
        latencies:           list[float]      = []
        per_query_results:   list[list[SearchResult]] = []

        for entry in GOLD_DATASET:
            t0 = time.perf_counter()
            results: list[SearchResult] = await retriever.retrieve(
                query=entry["query"],
                corpus_ids=[DEFAULT_CORPUS_ID],
                tenant_id=DEFAULT_TENANT_ID,
                k=k,
                search_type=search_type,
            )
            latencies.append((time.perf_counter() - t0) * 1000)

            rel_list = build_relevance_list(results, entry["relevant_sources"])
            per_query_relevance.append(rel_list)
            per_query_totals.append(len(entry["relevant_sources"]))
            per_query_results.append(results)

        return per_query_relevance, per_query_totals, latencies, per_query_results

    def _log_table(
        self,
        metrics_by_k: dict[int, dict[str, float]],
        latencies: list[float],
    ) -> None:
        logger.info("")
        logger.info("=" * 65)
        logger.info("  RETRIEVAL METRICS — RAG v2, hybrid, NeuralFlow AI corpus")
        logger.info("=" * 65)
        logger.info(f"  {'Metric':<18}{'K=1':>10}{'K=3':>10}{'K=5':>10}")
        logger.info("-" * 55)
        for metric in ["hit_rate", "mrr", "precision", "recall", "ndcg"]:
            label = metric.upper()
            row = f"  {label + '@K':<18}"
            for k in K_VALUES:
                row += f"{metrics_by_k[k][metric]:>10.3f}"
            logger.info(row)
        logger.info("-" * 55)
        logger.info(f"  {'Mean latency':<18}{sum(latencies)/len(latencies):>9.0f}ms")
        logger.info(f"  {'P95  latency':<18}{percentile(latencies, 95):>9.0f}ms")
        logger.info("=" * 65)

    def _log_per_query(
        self,
        per_query_relevance: list[list[int]],
        per_query_results: list[list[SearchResult]],
        latencies: list[float],
    ) -> None:
        logger.info("  Per-query breakdown (K=5):")
        logger.info(f"  {'Query':<52} {'Hit':>4} {'RR':>6} {'Lat':>7}")
        logger.info("  " + "-" * 72)
        for entry, rel, results, lat in zip(
            GOLD_DATASET, per_query_relevance, per_query_results, latencies
        ):
            q = entry["query"][:50]
            h = "✓" if hit_rate(rel) else "✗"
            rr = reciprocal_rank(rel)
            top_src = results[0].document_source.split("/")[-1] if results else "—"
            logger.info(f"  {q:<52} {h:>4} {rr:>6.2f} {lat:>6.0f}ms  → {top_src}")

    # ── IR metric tests ───────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_hit_rate_at_5(self, retriever: Retriever) -> None:
        """Hit Rate@5 ≥ 0.60 — every query should surface ≥1 relevant doc."""
        rels, totals, lats, results = await self._run_gold(retriever, k=5)
        by_k = {k: compute_all_metrics([r[:k] for r in rels], totals, k) for k in K_VALUES}
        self._log_table(by_k, lats)
        self._log_per_query(rels, results, lats)
        score = by_k[5]["hit_rate"]
        assert score >= THRESHOLDS_K5["hit_rate"], (
            f"Hit Rate@5 {score:.3f} < threshold {THRESHOLDS_K5['hit_rate']}"
        )

    @pytest.mark.asyncio
    async def test_mrr_at_5(self, retriever: Retriever) -> None:
        """MRR@5 ≥ 0.40 — first relevant result should appear near the top."""
        rels, totals, _, _ = await self._run_gold(retriever, k=5)
        score = compute_all_metrics(rels, totals, k=5)["mrr"]
        logger.info(f"MRR@5 = {score:.3f}  (threshold ≥ {THRESHOLDS_K5['mrr']})")
        assert score >= THRESHOLDS_K5["mrr"], (
            f"MRR@5 {score:.3f} < threshold {THRESHOLDS_K5['mrr']}"
        )

    @pytest.mark.asyncio
    async def test_ndcg_at_5(self, retriever: Retriever) -> None:
        """NDCG@5 ≥ 0.40 — relevant docs should rank above irrelevant ones."""
        rels, totals, _, _ = await self._run_gold(retriever, k=5)
        score = compute_all_metrics(rels, totals, k=5)["ndcg"]
        logger.info(f"NDCG@5 = {score:.3f}  (threshold ≥ {THRESHOLDS_K5['ndcg']})")
        assert score >= THRESHOLDS_K5["ndcg"], (
            f"NDCG@5 {score:.3f} < threshold {THRESHOLDS_K5['ndcg']}"
        )

    @pytest.mark.asyncio
    async def test_precision_at_5(self, retriever: Retriever) -> None:
        """Precision@5 ≥ 0.15 — ≥1 in 5 returned results should be relevant."""
        rels, totals, _, _ = await self._run_gold(retriever, k=5)
        score = compute_all_metrics(rels, totals, k=5)["precision"]
        logger.info(f"Precision@5 = {score:.3f}  (threshold ≥ {THRESHOLDS_K5['precision']})")
        assert score >= THRESHOLDS_K5["precision"], (
            f"Precision@5 {score:.3f} < threshold {THRESHOLDS_K5['precision']}"
        )

    @pytest.mark.asyncio
    async def test_recall_at_5(self, retriever: Retriever) -> None:
        """Recall@5 ≥ 0.40 — ≥40 % of relevant docs should appear in top-5."""
        rels, totals, _, _ = await self._run_gold(retriever, k=5)
        score = compute_all_metrics(rels, totals, k=5)["recall"]
        logger.info(f"Recall@5 = {score:.3f}  (threshold ≥ {THRESHOLDS_K5['recall']})")
        assert score >= THRESHOLDS_K5["recall"], (
            f"Recall@5 {score:.3f} < threshold {THRESHOLDS_K5['recall']}"
        )

    @pytest.mark.asyncio
    async def test_p95_latency_under_10s(self, retriever: Retriever) -> None:
        """P95 query latency must be under 10 s."""
        _, _, lats, _ = await self._run_gold(retriever, k=5)
        mean_ms = sum(lats) / len(lats)
        p95_ms  = percentile(lats, 95)
        logger.info(f"Mean = {mean_ms:.0f}ms  P95 = {p95_ms:.0f}ms")
        assert p95_ms < 10_000, f"P95 latency {p95_ms:.0f}ms > 10 000ms"

    @pytest.mark.asyncio
    async def test_semantic_hit_rate_baseline(self, retriever: Retriever) -> None:
        """Semantic-only Hit Rate@5 ≥ 0.40 — embedding search must find something."""
        rels, totals, _, _ = await self._run_gold(retriever, k=5, search_type="semantic")
        score = compute_all_metrics(rels, totals, k=5)["hit_rate"]
        logger.info(f"Semantic Hit Rate@5 = {score:.3f}")
        assert score >= 0.40, f"Semantic Hit Rate@5 {score:.3f} < 0.40"

    @pytest.mark.asyncio
    async def test_text_hit_rate_baseline(self, retriever: Retriever) -> None:
        """Text-only Hit Rate@5 ≥ 0.40 — keyword search must find something."""
        rels, totals, _, _ = await self._run_gold(retriever, k=5, search_type="text")
        score = compute_all_metrics(rels, totals, k=5)["hit_rate"]
        logger.info(f"Text Hit Rate@5 = {score:.3f}")
        assert score >= 0.40, f"Text Hit Rate@5 {score:.3f} < 0.40"

    @pytest.mark.asyncio
    async def test_hybrid_not_worse_than_semantic(self, retriever: Retriever) -> None:
        """Hybrid Hit Rate@5 must not fall >10 pp below semantic-only."""
        scores: dict[str, float] = {}
        for st in ("hybrid", "semantic"):
            rels, totals, _, _ = await self._run_gold(retriever, k=5, search_type=st)
            scores[st] = compute_all_metrics(rels, totals, k=5)["hit_rate"]
        logger.info(f"Hybrid {scores['hybrid']:.3f}  Semantic {scores['semantic']:.3f}")
        assert scores["hybrid"] >= scores["semantic"] - 0.10, (
            f"Hybrid ({scores['hybrid']:.3f}) > 10pp below semantic ({scores['semantic']:.3f})"
        )

    # ── Corpus isolation ──────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_unknown_corpus_returns_empty(self, retriever: Retriever) -> None:
        """A corpus that has never been ingested returns no results."""
        results = await retriever.retrieve(
            query="What does NeuralFlow AI do?",
            corpus_ids=["corpus-that-does-not-exist-xyz"],
            tenant_id=DEFAULT_TENANT_ID,
            k=5,
        )
        assert results == [], (
            f"Expected empty results for unknown corpus, got {len(results)} results"
        )

    @pytest.mark.asyncio
    async def test_results_scoped_to_corpus(self, retriever: Retriever) -> None:
        """Results for a known corpus contain content; a different corpus returns none."""
        results_default = await retriever.retrieve(
            query="What does NeuralFlow AI do?",
            corpus_ids=[DEFAULT_CORPUS_ID],
            tenant_id=DEFAULT_TENANT_ID,
            k=5,
        )
        results_other = await retriever.retrieve(
            query="What does NeuralFlow AI do?",
            corpus_ids=["other-corpus-xyz"],
            tenant_id=DEFAULT_TENANT_ID,
            k=5,
        )
        assert results_default, "Expected results for the default corpus"
        assert results_other == [], (
            f"Expected empty results for unknown corpus, got {len(results_other)}"
        )

    # ── Ground-truth content tests ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_pto_query_returns_handbook_content(self, retriever: Retriever) -> None:
        """PTO query must surface content mentioning leave / time off."""
        results = await retriever.retrieve(
            query="What is the PTO policy?",
            corpus_ids=[DEFAULT_CORPUS_ID],
            tenant_id=DEFAULT_TENANT_ID,
            k=5,
        )
        assert results, "No results returned"
        assert ground_truth_hit(results, ["pto", "vacation", "leave", "time off"]), (
            f"None of the top-5 chunks mention PTO/leave. Top chunk: {results[0].content[:200]}"
        )

    @pytest.mark.asyncio
    async def test_mission_query_returns_goals_content(self, retriever: Retriever) -> None:
        """Mission query must surface content mentioning mission or vision."""
        results = await retriever.retrieve(
            query="What is the company mission and vision?",
            corpus_ids=[DEFAULT_CORPUS_ID],
            tenant_id=DEFAULT_TENANT_ID,
            k=5,
        )
        assert results, "No results returned"
        assert ground_truth_hit(results, ["mission", "vision", "goal"]), (
            f"Top chunk: {results[0].content[:200]}"
        )

    @pytest.mark.asyncio
    async def test_q4_query_returns_business_review_content(self, retriever: Retriever) -> None:
        """Q4 results query must surface business review content."""
        results = await retriever.retrieve(
            query="Q4 2024 business results and performance review",
            corpus_ids=[DEFAULT_CORPUS_ID],
            tenant_id=DEFAULT_TENANT_ID,
            k=5,
        )
        assert results, "No results returned"
        assert ground_truth_hit(results, ["q4", "2024", "revenue", "quarter", "performance"]), (
            f"Top chunk: {results[0].content[:200]}"
        )

    @pytest.mark.asyncio
    async def test_implementation_query_returns_playbook_content(self, retriever: Retriever) -> None:
        """Implementation playbook query must surface playbook content."""
        results = await retriever.retrieve(
            query="Implementation approach and playbook steps",
            corpus_ids=[DEFAULT_CORPUS_ID],
            tenant_id=DEFAULT_TENANT_ID,
            k=5,
        )
        assert results, "No results returned"
        assert ground_truth_hit(results, ["implementation", "playbook", "phase", "step"]), (
            f"Top chunk: {results[0].content[:200]}"
        )

    @pytest.mark.asyncio
    async def test_company_overview_query_returns_company_content(self, retriever: Retriever) -> None:
        """Company overview query must surface company description content."""
        results = await retriever.retrieve(
            query="What does NeuralFlow AI do?",
            corpus_ids=[DEFAULT_CORPUS_ID],
            tenant_id=DEFAULT_TENANT_ID,
            k=5,
        )
        assert results, "No results returned"
        assert ground_truth_hit(results, ["neuralflow", "ai", "company", "platform"]), (
            f"Top chunk: {results[0].content[:200]}"
        )

    @pytest.mark.asyncio
    async def test_result_fields_populated(self, retriever: Retriever) -> None:
        """Every returned SearchResult has the required fields set."""
        results = await retriever.retrieve(
            query="What does NeuralFlow AI do?",
            corpus_ids=[DEFAULT_CORPUS_ID],
            tenant_id=DEFAULT_TENANT_ID,
            k=3,
        )
        assert results, "No results returned"
        for r in results:
            assert r.content,          f"Empty content in {r.chunk_id}"
            assert r.document_title,   f"Empty title in {r.chunk_id}"
            assert r.document_source,  f"Empty source in {r.chunk_id}"
            assert r.raw_score > 0,    f"Zero raw_score in {r.chunk_id}"
            assert r.raw_score_type,   f"Empty score type in {r.chunk_id}"

    @pytest.mark.asyncio
    async def test_top_result_outscores_bottom(self, retriever: Retriever) -> None:
        """Results must be sorted by score descending."""
        results = await retriever.retrieve(
            query="What does NeuralFlow AI do?",
            corpus_ids=[DEFAULT_CORPUS_ID],
            tenant_id=DEFAULT_TENANT_ID,
            k=5,
        )
        if len(results) < 2:
            pytest.skip("Too few results to compare ordering")
        scores = [r.raw_score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted by score: {scores}"
        )

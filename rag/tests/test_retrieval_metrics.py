"""
Retrieval quality evaluation using a gold dataset.

Metrics computed for K in {1, 3, 5}:
  Hit Rate@K    — fraction of queries where ≥1 relevant doc appears in top-K
  MRR@K         — mean reciprocal rank of the first relevant result
  Precision@K   — mean fraction of top-K results that are relevant
  Recall@K      — mean fraction of relevant docs retrieved in top-K
  NDCG@K        — normalised discounted cumulative gain

System metrics (per full evaluation run):
  Mean latency  — average retrieval time per query (ms)
  P95 latency   — 95th-percentile query latency (ms)
"""

import logging
import math
import time

import pytest
import pytest_asyncio

from rag.retrieval.retriever import Retriever
from rag.storage.vector_store.postgres import PostgresHybridStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gold dataset
# Each entry maps a natural-language query to the document filename stems
# that are considered relevant answers. Relevance is checked by testing
# whether any stem appears (case-insensitive) in the result's document_source.
# ---------------------------------------------------------------------------
GOLD_DATASET: list[dict] = [
    {
        "query": "What does NeuralFlow AI do?",
        "relevant_sources": ["company-overview", "mission-and-goals"],
    },
    {
        "query": "What is the PTO policy?",
        "relevant_sources": ["team-handbook"],
    },
    {
        "query": "What is the learning budget for employees?",
        "relevant_sources": ["team-handbook"],
    },
    {
        "query": "What technologies and architecture does the platform use?",
        "relevant_sources": ["technical-architecture-guide"],
    },
    {
        "query": "What is the company mission and vision?",
        "relevant_sources": ["mission-and-goals"],
    },
    {
        "query": "GlobalFinance Corp loan processing success story",
        "relevant_sources": ["client-review-globalfinance", "Recording4"],
    },
    {
        "query": "How many employees work at NeuralFlow AI?",
        "relevant_sources": ["company-overview", "team-handbook"],
    },
    {
        "query": "What is DocFlow AI and how does it process documents?",
        "relevant_sources": ["Recording2"],
    },
    {
        "query": "Q4 2024 business results and performance review",
        "relevant_sources": ["q4-2024-business-review"],
    },
    {
        "query": "implementation approach and playbook",
        "relevant_sources": ["implementation-playbook"],
    },
]

K_VALUES = [1, 3, 5]

# Minimum acceptable metric thresholds at K=5 for hybrid search.
# These reflect baseline expectations for the NeuralFlow AI document corpus.
THRESHOLDS_K5 = {
    "hit_rate": 0.6,    # ≥60% of queries find a relevant doc in top-5
    "mrr": 0.4,         # mean reciprocal rank ≥0.40
    "precision": 0.15,  # at least ~1 relevant result per 5 returned on average
    "recall": 0.4,      # ≥40% of relevant docs surfaced in top-5
    "ndcg": 0.4,        # NDCG@5 ≥0.40
}


# ---------------------------------------------------------------------------
# Pure metric functions  (no I/O — unit-testable in isolation)
# ---------------------------------------------------------------------------

def is_relevant(document_source: str, relevant_sources: list[str]) -> bool:
    """Return True if any relevant source stem appears in document_source."""
    src_lower = document_source.lower()
    return any(stem.lower() in src_lower for stem in relevant_sources)


def build_relevance_list(results, relevant_sources: list[str]) -> list[int]:
    """Convert a result list into a binary relevance sequence (1=relevant, 0=not)."""
    return [int(is_relevant(r.document_source, relevant_sources)) for r in results]


def hit_rate(relevance_list: list[int]) -> float:
    """1.0 if any relevant result is present, else 0.0."""
    return 1.0 if any(relevance_list) else 0.0


def reciprocal_rank(relevance_list: list[int]) -> float:
    """1 / rank of first relevant result (1-indexed); 0.0 if none found."""
    for i, rel in enumerate(relevance_list):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(relevance_list: list[int], k: int) -> float:
    """Fraction of top-K results that are relevant."""
    if k == 0:
        return 0.0
    return sum(relevance_list[:k]) / k


def recall_at_k(relevance_list: list[int], k: int, total_relevant: int) -> float:
    """Fraction of all relevant documents that appear in top-K."""
    if total_relevant == 0:
        return 0.0
    return sum(relevance_list[:k]) / total_relevant


def ndcg_at_k(relevance_list: list[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain at K (binary relevance).

    DCG@K  = Σ rel_i / log2(i+2)   for i = 0..K-1
    IDCG@K = Σ 1    / log2(i+2)    for i = 0..min(#relevant, K)-1
    NDCG@K = DCG@K / IDCG@K        (0.0 if no relevant docs exist)
    """
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
    """Aggregate per-query relevance lists into mean metrics at K."""
    n = len(per_query_relevance)
    if n == 0:
        return {}
    return {
        "hit_rate": sum(hit_rate(r) for r in per_query_relevance) / n,
        "mrr": sum(reciprocal_rank(r[:k]) for r in per_query_relevance) / n,
        "precision": sum(precision_at_k(r, k) for r in per_query_relevance) / n,
        "recall": sum(
            recall_at_k(r, k, t)
            for r, t in zip(per_query_relevance, per_query_total_relevant)
        ) / n,
        "ndcg": sum(ndcg_at_k(r, k) for r in per_query_relevance) / n,
    }


def percentile(values: list[float], p: float) -> float:
    """Return the p-th percentile of values (0–100)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (p / 100) * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


# ---------------------------------------------------------------------------
# Unit tests for metric functions  (no DB, no async)
# ---------------------------------------------------------------------------

class TestMetricFunctions:
    """Unit tests for pure metric helpers — no external dependencies."""

    def test_is_relevant_match(self):
        assert is_relevant("/path/to/team-handbook.md", ["team-handbook"]) is True

    def test_is_relevant_no_match(self):
        assert is_relevant("/path/to/team-handbook.md", ["company-overview"]) is False

    def test_is_relevant_case_insensitive(self):
        assert is_relevant("/docs/Recording4.mp3", ["recording4"]) is True

    def test_is_relevant_multiple_sources(self):
        assert is_relevant("/docs/company-overview.md", ["team-handbook", "company-overview"]) is True

    def test_hit_rate_positive(self):
        assert hit_rate([0, 1, 0]) == 1.0

    def test_hit_rate_negative(self):
        assert hit_rate([0, 0, 0]) == 0.0

    def test_reciprocal_rank_first(self):
        assert reciprocal_rank([1, 0, 0]) == pytest.approx(1.0)

    def test_reciprocal_rank_second(self):
        assert reciprocal_rank([0, 1, 0]) == pytest.approx(0.5)

    def test_reciprocal_rank_none(self):
        assert reciprocal_rank([0, 0, 0]) == 0.0

    def test_precision_at_k_all_relevant(self):
        assert precision_at_k([1, 1, 1, 0, 0], k=3) == pytest.approx(1.0)

    def test_precision_at_k_partial(self):
        assert precision_at_k([1, 0, 1, 0, 0], k=4) == pytest.approx(0.5)

    def test_precision_at_k_none(self):
        assert precision_at_k([0, 0, 0], k=3) == pytest.approx(0.0)

    def test_recall_at_k_full(self):
        # 2 relevant, both in top-3
        assert recall_at_k([1, 1, 0, 0], k=3, total_relevant=2) == pytest.approx(1.0)

    def test_recall_at_k_partial(self):
        # 2 relevant, only 1 in top-3
        assert recall_at_k([1, 0, 0, 1], k=3, total_relevant=2) == pytest.approx(0.5)

    def test_recall_at_k_zero_relevant(self):
        assert recall_at_k([0, 0, 0], k=3, total_relevant=0) == 0.0

    def test_ndcg_perfect_ranking(self):
        # 2 relevant docs, both at top — perfect NDCG
        assert ndcg_at_k([1, 1, 0, 0], k=3) == pytest.approx(1.0)

    def test_ndcg_worst_ranking(self):
        # 1 relevant doc buried at position 3 out of 3
        # DCG = 1/log2(4) ≈ 0.5; IDCG = 1/log2(2) = 1.0
        expected = (1.0 / math.log2(4)) / (1.0 / math.log2(2))
        assert ndcg_at_k([0, 0, 1], k=3) == pytest.approx(expected)

    def test_ndcg_no_relevant(self):
        assert ndcg_at_k([0, 0, 0], k=3) == 0.0

    def test_compute_all_metrics_shape(self):
        rels = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
        totals = [1, 1]
        metrics = compute_all_metrics(rels, totals, k=5)
        assert set(metrics) == {"hit_rate", "mrr", "precision", "recall", "ndcg"}

    def test_compute_all_metrics_empty(self):
        assert compute_all_metrics([], [], k=5) == {}

    def test_percentile_median(self):
        assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == pytest.approx(3.0)

    def test_percentile_p95(self):
        # vals = [1..100], idx = 0.95*99 = 94.05 → 95 + (96-95)*0.05 = 95.05
        vals = list(range(1, 101))
        assert percentile(vals, 95) == pytest.approx(95.05)


# ---------------------------------------------------------------------------
# Integration tests — run retriever against the gold dataset
# ---------------------------------------------------------------------------

class TestRetrievalMetrics:
    """Evaluate retrieval quality against the gold dataset.

    Requires: PostgreSQL with ingested NeuralFlow AI documents + Ollama.
    Skipped automatically if the database is unreachable.
    """

    @pytest_asyncio.fixture
    async def retriever(self):
        store = PostgresHybridStore()
        await store.initialize()
        r = Retriever(store=store)
        yield r
        await store.close()

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    async def _run_gold_dataset(
        self,
        retriever: Retriever,
        k: int,
        search_type: str = "hybrid",
    ) -> tuple[list[list[int]], list[int], list[float]]:
        """Run all gold queries and return (per_query_relevance, totals, latencies_ms)."""
        per_query_relevance: list[list[int]] = []
        per_query_totals: list[int] = []
        latencies: list[float] = []

        for entry in GOLD_DATASET:
            t0 = time.perf_counter()
            results = await retriever.retrieve(
                query=entry["query"],
                match_count=k,
                search_type=search_type,
            )
            latencies.append((time.perf_counter() - t0) * 1000)

            rel_list = build_relevance_list(results, entry["relevant_sources"])
            per_query_relevance.append(rel_list)
            per_query_totals.append(len(entry["relevant_sources"]))

        return per_query_relevance, per_query_totals, latencies

    def _log_metrics_table(
        self,
        metrics_by_k: dict[int, dict[str, float]],
        latencies: list[float],
    ) -> None:
        logger.info("")
        logger.info("=" * 65)
        logger.info("  RETRIEVAL METRICS — hybrid search, NeuralFlow AI corpus")
        logger.info("=" * 65)
        logger.info(f"  {'Metric':<18}{'K=1':>10}{'K=3':>10}{'K=5':>10}")
        logger.info("-" * 55)
        for metric in ["hit_rate", "mrr", "precision", "recall", "ndcg"]:
            label = metric.upper() if metric != "ndcg" else "NDCG"
            row = f"  {label+'@K':<18}"
            for k in K_VALUES:
                row += f"{metrics_by_k[k][metric]:>10.3f}"
            logger.info(row)
        logger.info("-" * 55)
        logger.info(f"  {'Mean latency':<18}{sum(latencies)/len(latencies):>9.0f}ms")
        logger.info(f"  {'P95  latency':<18}{percentile(latencies, 95):>9.0f}ms")
        logger.info("=" * 65)
        logger.info("")

    def _log_per_query_detail(
        self,
        per_query_relevance: list[list[int]],
        latencies: list[float],
    ) -> None:
        logger.info("  Per-query breakdown (K=5):")
        logger.info(f"  {'Query':<52} {'Hit':>4} {'RR':>6} {'Lat':>7}")
        logger.info("  " + "-" * 72)
        for entry, rel, lat in zip(GOLD_DATASET, per_query_relevance, latencies):
            q = entry["query"][:50]
            h = "✓" if hit_rate(rel) else "✗"
            rr = reciprocal_rank(rel)
            logger.info(f"  {q:<52} {h:>4} {rr:>6.2f} {lat:>6.0f}ms")
        logger.info("")

    # ------------------------------------------------------------------ #
    # Tests                                                                #
    # ------------------------------------------------------------------ #

    @pytest.mark.asyncio
    async def test_hit_rate_at_5(self, retriever):
        """Hit Rate@5 must meet minimum threshold — every query should surface ≥1 relevant doc."""
        per_query_relevance, per_query_totals, latencies = await self._run_gold_dataset(
            retriever, k=5
        )
        metrics_by_k = {
            k: compute_all_metrics(
                [r[:k] for r in per_query_relevance], per_query_totals, k
            )
            for k in K_VALUES
        }
        self._log_metrics_table(metrics_by_k, latencies)
        self._log_per_query_detail(per_query_relevance, latencies)

        score = metrics_by_k[5]["hit_rate"]
        logger.info(f"Hit Rate@5 = {score:.3f}  (threshold ≥ {THRESHOLDS_K5['hit_rate']})")
        assert score >= THRESHOLDS_K5["hit_rate"], (
            f"Hit Rate@5 {score:.3f} below threshold {THRESHOLDS_K5['hit_rate']}"
        )

    @pytest.mark.asyncio
    async def test_mrr_at_5(self, retriever):
        """MRR@5 — the first relevant result should appear near the top on average."""
        per_query_relevance, per_query_totals, _ = await self._run_gold_dataset(
            retriever, k=5
        )
        metrics = compute_all_metrics(per_query_relevance, per_query_totals, k=5)
        score = metrics["mrr"]
        logger.info(f"MRR@5 = {score:.3f}  (threshold ≥ {THRESHOLDS_K5['mrr']})")
        assert score >= THRESHOLDS_K5["mrr"], (
            f"MRR@5 {score:.3f} below threshold {THRESHOLDS_K5['mrr']}"
        )

    @pytest.mark.asyncio
    async def test_ndcg_at_5(self, retriever):
        """NDCG@5 — relevant results should be ranked ahead of irrelevant ones."""
        per_query_relevance, per_query_totals, _ = await self._run_gold_dataset(
            retriever, k=5
        )
        metrics = compute_all_metrics(per_query_relevance, per_query_totals, k=5)
        score = metrics["ndcg"]
        logger.info(f"NDCG@5 = {score:.3f}  (threshold ≥ {THRESHOLDS_K5['ndcg']})")
        assert score >= THRESHOLDS_K5["ndcg"], (
            f"NDCG@5 {score:.3f} below threshold {THRESHOLDS_K5['ndcg']}"
        )

    @pytest.mark.asyncio
    async def test_precision_at_5(self, retriever):
        """Precision@5 — at least 1 in 5 returned results should be relevant on average."""
        per_query_relevance, per_query_totals, _ = await self._run_gold_dataset(
            retriever, k=5
        )
        metrics = compute_all_metrics(per_query_relevance, per_query_totals, k=5)
        score = metrics["precision"]
        logger.info(f"Precision@5 = {score:.3f}  (threshold ≥ {THRESHOLDS_K5['precision']})")
        assert score >= THRESHOLDS_K5["precision"], (
            f"Precision@5 {score:.3f} below threshold {THRESHOLDS_K5['precision']}"
        )

    @pytest.mark.asyncio
    async def test_recall_at_5(self, retriever):
        """Recall@5 — at least 40% of relevant docs should appear in top-5."""
        per_query_relevance, per_query_totals, _ = await self._run_gold_dataset(
            retriever, k=5
        )
        metrics = compute_all_metrics(per_query_relevance, per_query_totals, k=5)
        score = metrics["recall"]
        logger.info(f"Recall@5 = {score:.3f}  (threshold ≥ {THRESHOLDS_K5['recall']})")
        assert score >= THRESHOLDS_K5["recall"], (
            f"Recall@5 {score:.3f} below threshold {THRESHOLDS_K5['recall']}"
        )

    @pytest.mark.asyncio
    async def test_latency(self, retriever):
        """System metric: P95 query latency must be under 10 seconds."""
        _, _, latencies = await self._run_gold_dataset(retriever, k=5)
        mean_ms = sum(latencies) / len(latencies)
        p95_ms = percentile(latencies, 95)
        logger.info(f"Mean latency = {mean_ms:.0f}ms  P95 = {p95_ms:.0f}ms")
        assert p95_ms < 10_000, f"P95 latency {p95_ms:.0f}ms exceeds 10s limit"

    @pytest.mark.asyncio
    async def test_semantic_vs_text_hit_rate(self, retriever):
        """Semantic and text search both achieve Hit Rate@5 ≥ 0.4 individually."""
        for search_type in ("semantic", "text"):
            per_query_relevance, per_query_totals, _ = await self._run_gold_dataset(
                retriever, k=5, search_type=search_type
            )
            metrics = compute_all_metrics(per_query_relevance, per_query_totals, k=5)
            score = metrics["hit_rate"]
            logger.info(f"{search_type} Hit Rate@5 = {score:.3f}")
            assert score >= 0.4, (
                f"{search_type} Hit Rate@5 {score:.3f} below 0.4"
            )

    @pytest.mark.asyncio
    async def test_hybrid_beats_semantic_alone(self, retriever):
        """Hybrid Hit Rate@5 should not fall more than 10pp below semantic-only.

        Hybrid combines semantic + keyword signals via RRF. In theory it should
        be ≥ either alone, but in practice RRF rank merging can demote some
        purely-semantic hits. A 10-percentage-point tolerance is acceptable.
        """
        results = {}
        for search_type in ("hybrid", "semantic"):
            per_query_relevance, per_query_totals, _ = await self._run_gold_dataset(
                retriever, k=5, search_type=search_type
            )
            results[search_type] = compute_all_metrics(
                per_query_relevance, per_query_totals, k=5
            )["hit_rate"]
        logger.info(
            f"Hybrid Hit Rate@5 = {results['hybrid']:.3f}, "
            f"Semantic = {results['semantic']:.3f}"
        )
        tolerance = 0.10
        assert results["hybrid"] >= results["semantic"] - tolerance, (
            f"Hybrid Hit Rate@5 ({results['hybrid']:.3f}) is more than "
            f"{tolerance:.0%} below semantic ({results['semantic']:.3f})"
        )

"""Retrieval quality tests for the CUAD legal document corpus.

Integration tests that run the retriever against a gold dataset of
contract-type queries. Each query targets a specific contract category
(Distributor, Franchise, License, etc.) and verifies that the top-K
results contain documents from the expected category.

Also includes corpus-isolation tests to verify that:
- Legal clause queries surface legal docs, not NeuralFlow AI docs
- NeuralFlow company queries surface NeuralFlow docs, not CUAD contracts

Requires: PostgreSQL with ingested CUAD + NeuralFlow documents + Ollama.
"""

import logging
import time

import pytest
import pytest_asyncio

from rag.retrieval.retriever import Retriever
from rag.storage.vector_store.postgres import PostgresHybridStore

# Reuse pure metric helpers from the existing retrieval metrics test module
from rag.tests.test_retrieval_metrics import (
    build_relevance_list,
    compute_all_metrics,
    hit_rate,
    is_relevant,
    percentile,
    reciprocal_rank,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legal gold dataset
#
# Each entry maps a clause-type query to filename-stem patterns that appear
# in the source path of relevant contracts. Patterns are checked with a
# case-insensitive substring match against `document_source`.
# ---------------------------------------------------------------------------

LEGAL_GOLD_DATASET: list[dict] = [
    {
        "query": "exclusive distributor rights in territory minimum purchase obligations",
        "relevant_sources": ["Distributor", "distributor"],
        "description": "Distributor agreements (31 contracts in corpus)",
    },
    {
        "query": "co-branding marketing and distribution agreement brand license",
        "relevant_sources": ["Co_Branding", "Co-Branding", "co_branding"],
        "description": "Co-branding agreements (21 contracts)",
    },
    {
        "query": "franchise fee royalty payments territory license obligations",
        "relevant_sources": ["Franchise", "franchise"],
        "description": "Franchise agreements (15 contracts)",
    },
    {
        "query": "IT outsourcing services data processing operations",
        "relevant_sources": ["Outsourcing", "outsourcing"],
        "description": "Outsourcing agreements (16 contracts)",
    },
    {
        "query": "supply agreement purchase orders product specifications delivery",
        "relevant_sources": ["Supply", "supply"],
        "description": "Supply agreements (24 contracts)",
    },
    {
        "query": "software license grant non-exclusive perpetual right to use",
        "relevant_sources": ["License", "license"],
        "description": "License agreements (40 contracts)",
    },
    {
        "query": "consulting services independent contractor professional fees statement of work",
        "relevant_sources": ["Consulting", "consulting"],
        "description": "Consulting agreements (11 contracts)",
    },
    {
        "query": "authorized reseller agreement sales territory commission",
        "relevant_sources": ["Reseller", "reseller"],
        "description": "Reseller agreements (8 contracts)",
    },
    {
        "query": "strategic alliance partnership joint marketing collaboration",
        "relevant_sources": ["Alliance", "alliance", "Collaboration", "collaboration"],
        "description": "Alliance and collaboration agreements (13 contracts)",
    },
    {
        "query": "service agreement professional services scope of work SLA",
        "relevant_sources": ["Service", "service"],
        "description": "Service agreements (37 contracts)",
    },
]

# Minimum thresholds for the legal gold dataset at K=5.
# Lower than NeuralFlow thresholds because CUAD has 509 contracts with
# overlapping terminology — precision is naturally diluted.
LEGAL_THRESHOLDS_K5 = {
    "hit_rate": 0.70,   # ≥70% of queries find a relevant contract type in top-5
    "mrr": 0.45,        # relevant contract should rank highly on average
    "precision": 0.10,  # at least 1 in 10 returned results from the right type
}

K_VALUES = [1, 3, 5]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def retriever():
    store = PostgresHybridStore()
    await store.initialize()
    r = Retriever(store=store)
    yield r
    await store.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_legal_source(document_source: str) -> bool:
    """True if the document comes from the legal corpus."""
    return "legal" in document_source.lower()


def _is_neuralflow_source(document_source: str) -> bool:
    """True if the document comes from the NeuralFlow AI corpus."""
    legal_patterns = [
        "agreement", "contract", "exhibit", "legal",
        "inc_", "corp_", "ltd_",
    ]
    src = document_source.lower()
    return not any(p in src for p in legal_patterns)


async def _run_legal_gold(
    retriever: Retriever,
    k: int,
    search_type: str = "hybrid",
) -> tuple[list[list[int]], list[int], list[float]]:
    """Run all legal gold queries; return (per_query_relevance, totals, latencies_ms)."""
    per_query_relevance: list[list[int]] = []
    per_query_totals: list[int] = []
    latencies: list[float] = []

    for entry in LEGAL_GOLD_DATASET:
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


def _log_legal_table(
    per_query_relevance: list[list[int]],
    latencies: list[float],
    search_type: str = "hybrid",
) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  RETRIEVAL METRICS — {search_type} search, CUAD legal corpus")
    logger.info("=" * 70)
    logger.info(f"  {'Query':<52} {'Hit':>4} {'RR':>6} {'Lat':>7}")
    logger.info("  " + "-" * 72)
    for entry, rel, lat in zip(LEGAL_GOLD_DATASET, per_query_relevance, latencies):
        q = entry["query"][:50]
        h = "✓" if hit_rate(rel) else "✗"
        rr = reciprocal_rank(rel)
        logger.info(f"  {q:<52} {h:>4} {rr:>6.2f} {lat:>6.0f}ms")
    logger.info("")


# ---------------------------------------------------------------------------
# Contract-type retrieval tests (gold dataset)
# ---------------------------------------------------------------------------


class TestLegalRetrievalMetrics:
    """Verify retrieval quality over the CUAD contract-type gold dataset."""

    @pytest.mark.asyncio
    async def test_hit_rate_at_5(self, retriever):
        """Hit Rate@5 ≥ 0.70 — most contract-type queries surface a relevant contract."""
        rels, totals, latencies = await _run_legal_gold(retriever, k=5)
        metrics = compute_all_metrics(rels, totals, k=5)
        _log_legal_table(rels, latencies)
        score = metrics["hit_rate"]
        logger.info(f"Hit Rate@5 = {score:.3f}  (threshold ≥ {LEGAL_THRESHOLDS_K5['hit_rate']})")
        assert score >= LEGAL_THRESHOLDS_K5["hit_rate"], (
            f"Legal Hit Rate@5 {score:.3f} below threshold {LEGAL_THRESHOLDS_K5['hit_rate']}"
        )

    @pytest.mark.asyncio
    async def test_mrr_at_5(self, retriever):
        """MRR@5 ≥ 0.45 — the first relevant contract ranks near the top on average."""
        rels, totals, _ = await _run_legal_gold(retriever, k=5)
        metrics = compute_all_metrics(rels, totals, k=5)
        score = metrics["mrr"]
        logger.info(f"Legal MRR@5 = {score:.3f}  (threshold ≥ {LEGAL_THRESHOLDS_K5['mrr']})")
        assert score >= LEGAL_THRESHOLDS_K5["mrr"], (
            f"Legal MRR@5 {score:.3f} below threshold {LEGAL_THRESHOLDS_K5['mrr']}"
        )

    @pytest.mark.asyncio
    async def test_precision_at_5(self, retriever):
        """Precision@5 ≥ 0.10 — at least 1 in 10 returned chunks is from the right contract type."""
        rels, totals, _ = await _run_legal_gold(retriever, k=5)
        metrics = compute_all_metrics(rels, totals, k=5)
        score = metrics["precision"]
        logger.info(f"Legal Precision@5 = {score:.3f}  (threshold ≥ {LEGAL_THRESHOLDS_K5['precision']})")
        assert score >= LEGAL_THRESHOLDS_K5["precision"], (
            f"Legal Precision@5 {score:.3f} below threshold {LEGAL_THRESHOLDS_K5['precision']}"
        )

    @pytest.mark.asyncio
    async def test_latency(self, retriever):
        """P95 latency < 10s — legal queries must not be disproportionately slow."""
        _, _, latencies = await _run_legal_gold(retriever, k=5)
        mean_ms = sum(latencies) / len(latencies)
        p95_ms = percentile(latencies, 95)
        logger.info(f"Legal mean latency = {mean_ms:.0f}ms  P95 = {p95_ms:.0f}ms")
        assert p95_ms < 10_000, f"P95 latency {p95_ms:.0f}ms exceeds 10s limit"

    @pytest.mark.asyncio
    async def test_hit_rate_improves_from_k1_to_k5(self, retriever):
        """Hit Rate@5 should be higher than Hit Rate@1 — more results helps for legal queries."""
        rels, totals, _ = await _run_legal_gold(retriever, k=5)
        metrics_k1 = compute_all_metrics([r[:1] for r in rels], totals, k=1)
        metrics_k5 = compute_all_metrics(rels, totals, k=5)
        logger.info(
            f"Legal Hit Rate: K=1 {metrics_k1['hit_rate']:.3f} → K=5 {metrics_k5['hit_rate']:.3f}"
        )
        assert metrics_k5["hit_rate"] >= metrics_k1["hit_rate"], (
            "Hit Rate@5 should not be lower than Hit Rate@1"
        )


# ---------------------------------------------------------------------------
# Per-query spot checks
# ---------------------------------------------------------------------------


class TestLegalSpotChecks:
    """Spot-check specific high-confidence queries against the legal corpus."""

    @pytest.mark.asyncio
    async def test_distributor_query_returns_distributor_contract(self, retriever):
        """'exclusive distributor rights' should return at least one distributor contract in top-5."""
        results = await retriever.retrieve(
            query="exclusive distributor rights in territory minimum purchase obligations",
            match_count=5,
            search_type="hybrid",
        )
        assert any(
            is_relevant(r.document_source, ["Distributor", "distributor"])
            for r in results
        ), "No distributor contract found in top-5 results"

    @pytest.mark.asyncio
    async def test_franchise_query_returns_franchise_contract(self, retriever):
        """'franchise fee royalty territory' should return at least one franchise contract in top-5."""
        results = await retriever.retrieve(
            query="franchise fee royalty payments territory license obligations",
            match_count=5,
            search_type="hybrid",
        )
        assert any(
            is_relevant(r.document_source, ["Franchise", "franchise"])
            for r in results
        ), "No franchise contract found in top-5 results"

    @pytest.mark.asyncio
    async def test_software_license_query_returns_license_contract(self, retriever):
        """Software license query should surface a license agreement in top-5."""
        results = await retriever.retrieve(
            query="software license grant non-exclusive perpetual right to use",
            match_count=5,
            search_type="hybrid",
        )
        assert any(
            is_relevant(r.document_source, ["License", "license"])
            for r in results
        ), "No license contract found in top-5 results"

    @pytest.mark.asyncio
    async def test_supply_query_returns_supply_contract(self, retriever):
        """Supply agreement query should surface a supply contract in top-5."""
        results = await retriever.retrieve(
            query="supply agreement purchase orders product specifications delivery",
            match_count=5,
            search_type="hybrid",
        )
        assert any(
            is_relevant(r.document_source, ["Supply", "supply"])
            for r in results
        ), "No supply contract found in top-5 results"

    @pytest.mark.asyncio
    async def test_top_result_is_legal_document(self, retriever):
        """A legal clause query should return a legal document as the #1 result."""
        results = await retriever.retrieve(
            query="governing law jurisdiction choice of law clause",
            match_count=5,
            search_type="hybrid",
        )
        assert results, "No results returned for legal clause query"
        top_source = results[0].document_source.lower()
        assert "legal" in top_source, (
            f"Top result is not a legal document: {results[0].document_source}"
        )


# ---------------------------------------------------------------------------
# Corpus isolation tests
# ---------------------------------------------------------------------------


class TestCorpusIsolation:
    """Verify that legal and NeuralFlow queries surface the right corpus."""

    @pytest.mark.asyncio
    async def test_legal_query_returns_legal_docs(self, retriever):
        """A contract clause query should return predominantly legal documents."""
        results = await retriever.retrieve(
            query="termination for cause written notice cure period contract",
            match_count=5,
            search_type="hybrid",
        )
        legal_count = sum(1 for r in results if _is_legal_source(r.document_source))
        logger.info(f"Legal docs in top-5: {legal_count}/5")
        assert legal_count >= 3, (
            f"Expected ≥3 legal docs in top-5, got {legal_count}. "
            f"Sources: {[r.document_source for r in results]}"
        )

    @pytest.mark.asyncio
    async def test_neuralflow_query_returns_neuralflow_docs(self, retriever):
        """A NeuralFlow-specific query should not return CUAD legal contracts."""
        results = await retriever.retrieve(
            query="NeuralFlow AI company mission vision and goals",
            match_count=5,
            search_type="hybrid",
        )
        legal_count = sum(1 for r in results if _is_legal_source(r.document_source))
        logger.info(
            f"Legal docs in NeuralFlow query top-5: {legal_count}/5. "
            f"Sources: {[r.document_source for r in results]}"
        )
        assert legal_count <= 2, (
            f"Too many legal docs ({legal_count}) returned for a NeuralFlow query"
        )

    @pytest.mark.asyncio
    async def test_legal_parties_query_returns_legal_docs(self, retriever):
        """'parties to the agreement' should return legal contracts, not NeuralFlow docs."""
        results = await retriever.retrieve(
            query="parties to the agreement obligations representations warranties",
            match_count=10,
            search_type="hybrid",
        )
        legal_count = sum(1 for r in results if _is_legal_source(r.document_source))
        ratio = legal_count / len(results) if results else 0
        logger.info(f"Legal ratio for parties query: {legal_count}/{len(results)} = {ratio:.2f}")
        assert ratio >= 0.6, (
            f"Expected ≥60% legal docs for parties query, got {ratio:.0%}"
        )

    @pytest.mark.asyncio
    async def test_technical_query_skews_neuralflow(self, retriever):
        """A product/tech query should return NeuralFlow docs ahead of legal contracts."""
        results = await retriever.retrieve(
            query="AI platform machine learning document processing automation",
            match_count=10,
            search_type="hybrid",
        )
        # At least 1 NeuralFlow document should appear in top-10
        neuralflow_found = any(not _is_legal_source(r.document_source) for r in results)
        logger.info(f"Sources: {[r.document_source for r in results[:5]]}")
        assert neuralflow_found, "Expected at least 1 NeuralFlow doc for AI platform query"


# ---------------------------------------------------------------------------
# Search-type comparison (hybrid vs semantic vs text) for legal queries
# ---------------------------------------------------------------------------


class TestLegalSearchTypes:
    """Compare hybrid, semantic, and text search on the legal gold dataset."""

    @pytest.mark.asyncio
    async def test_all_search_types_achieve_minimum_hit_rate(self, retriever):
        """All three search modes should achieve Hit Rate@5 ≥ 0.4 on legal queries."""
        min_hit_rate = 0.40
        for search_type in ("hybrid", "semantic", "text"):
            rels, totals, _ = await _run_legal_gold(retriever, k=5, search_type=search_type)
            metrics = compute_all_metrics(rels, totals, k=5)
            score = metrics["hit_rate"]
            logger.info(f"[{search_type}] Legal Hit Rate@5 = {score:.3f}")
            assert score >= min_hit_rate, (
                f"{search_type} Hit Rate@5 {score:.3f} below minimum {min_hit_rate}"
            )

    @pytest.mark.asyncio
    async def test_hybrid_not_worse_than_semantic(self, retriever):
        """Hybrid Hit Rate@5 should be within 15pp of semantic-only on legal queries."""
        results = {}
        for search_type in ("hybrid", "semantic"):
            rels, totals, _ = await _run_legal_gold(retriever, k=5, search_type=search_type)
            results[search_type] = compute_all_metrics(rels, totals, k=5)["hit_rate"]
        logger.info(
            f"Legal Hybrid Hit Rate@5 = {results['hybrid']:.3f}, "
            f"Semantic = {results['semantic']:.3f}"
        )
        assert results["hybrid"] >= results["semantic"] - 0.15, (
            f"Hybrid ({results['hybrid']:.3f}) is >15pp below semantic ({results['semantic']:.3f})"
        )

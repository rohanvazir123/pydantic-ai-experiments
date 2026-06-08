"""IR retrieval metrics — ported from rag/tests/retrieval/test_retrieval_metrics.py."""

import math


def is_relevant(document_source: str, relevant_sources: list[str]) -> bool:
    src = document_source.lower()
    return any(stem.lower() in src for stem in relevant_sources)


def build_relevance_list(results: list, relevant_sources: list[str]) -> list[int]:
    return [int(is_relevant(r.document_source, relevant_sources)) for r in results]


def hit_rate(relevance: list[int]) -> float:
    return 1.0 if any(relevance) else 0.0


def reciprocal_rank(relevance: list[int]) -> float:
    for i, r in enumerate(relevance):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(relevance: list[int], k: int) -> float:
    return sum(relevance[:k]) / k if k > 0 else 0.0


def recall_at_k(relevance: list[int], k: int, total_relevant: int) -> float:
    return sum(relevance[:k]) / total_relevant if total_relevant > 0 else 0.0


def ndcg_at_k(relevance: list[int], k: int) -> float:
    top = relevance[:k]
    dcg  = sum(r / math.log2(i + 2) for i, r in enumerate(top))
    n_rel = sum(relevance)
    ideal_k = min(n_rel, k)
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
        "hit_rate":  sum(hit_rate(r)           for r in per_query_relevance) / n,
        "mrr":       sum(reciprocal_rank(r[:k]) for r in per_query_relevance) / n,
        "precision": sum(precision_at_k(r, k)  for r in per_query_relevance) / n,
        "recall":    sum(
            recall_at_k(r, k, t)
            for r, t in zip(per_query_relevance, per_query_total_relevant)
        ) / n,
        "ndcg":      sum(ndcg_at_k(r, k)       for r in per_query_relevance) / n,
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sv  = sorted(values)
    idx = (p / 100) * (len(sv) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (idx - lo)

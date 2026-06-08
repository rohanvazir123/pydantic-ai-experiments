"""Per-stage timeout budget for a single API request.

Each stage carves a sub-deadline from the remaining parent budget.
If any stage exceeds its budget, asyncio.wait_for raises TimeoutError.

Usage:
    budget = TimeoutBudget()
    result = await asyncio.wait_for(embed_query(q), timeout=budget.embedding_s)
"""

from dataclasses import dataclass


@dataclass
class TimeoutBudget:
    total_s:          float = 30.0   # API hard deadline

    validation_s:     float = 0.2
    routing_s:        float = 3.0
    embedding_s:      float = 5.0
    retrieval_s:      float = 8.0
    rerank_s:         float = 3.0
    semantic_cache_s: float = 1.0
    generation_s:     float = 15.0
    judge_s:          float = 5.0

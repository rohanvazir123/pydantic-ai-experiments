"""Evaluation Pydantic models — EvalRun, EvalResult, GoldSample."""

from datetime import datetime, UTC
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class GoldSample(BaseModel):
    id:                   UUID      = Field(default_factory=uuid4)
    corpus_id:            str
    query:                str
    relevant_doc_sources: list[str]
    ground_truth_answer:  str | None = None
    difficulty:           Literal["easy", "medium", "hard"] = "medium"
    tags:                 list[str] = Field(default_factory=list)


class EvalRun(BaseModel):
    id:              UUID      = Field(default_factory=uuid4)
    corpus_id:       str
    git_commit:      str       = ""
    model_tier:      str       = "small"
    search_type:     str       = "hybrid"
    k:               int       = 5
    started_at:      datetime  = Field(default_factory=lambda: datetime.now(UTC))
    completed_at:    datetime | None = None
    status:          Literal["queued", "running", "completed", "failed"] = "queued"
    sample_count:    int       = 0
    baseline_run_id: UUID | None = None
    report_json:     dict[str, Any] | None = None


class EvalResult(BaseModel):
    id:          UUID = Field(default_factory=uuid4)
    run_id:      UUID
    sample_id:   UUID
    # Retrieval metrics
    hit_rate:    float | None = None
    mrr:         float | None = None
    ndcg:        float | None = None
    precision_at_k: float | None = None
    recall_at_k:    float | None = None
    # Generation metrics
    faithfulness:    float | None = None
    answer_relevance: float | None = None
    # Correctness (requires ground truth)
    bleu_4:           float | None = None
    rouge_l_f:        float | None = None
    semantic_similarity: float | None = None
    # Performance
    retrieval_ms:     int | None = None
    generation_ms:    int | None = None
    total_ms:         int | None = None
    prompt_tokens:    int | None = None
    completion_tokens: int | None = None
    estimated_cost_usd: float | None = None
    cache_tier_hit:   str | None = None
    # Confidence
    mean_confidence:  float | None = None
    min_confidence:   float | None = None
    low_confidence_flag: bool = False
    # Pipeline status
    pipeline_status:  str | None = None
    abstention_layer: int | None = None
    judge_verdict:    str | None = None
    judge_confidence: float | None = None
    false_abstention: bool = False

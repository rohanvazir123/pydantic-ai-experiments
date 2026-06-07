-- Migration 004: evaluation system
-- gold_samples, eval_runs (with report_json), eval_results (with confidence fields)

CREATE TABLE IF NOT EXISTS gold_samples (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_id               TEXT NOT NULL,
    query                   TEXT NOT NULL,
    relevant_doc_sources    TEXT[] NOT NULL,
    ground_truth_answer     TEXT,
    difficulty              TEXT NOT NULL DEFAULT 'medium'
                                CHECK (difficulty IN ('easy', 'medium', 'hard')),
    tags                    TEXT[] NOT NULL DEFAULT '{}',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS gold_samples_corpus_idx ON gold_samples (corpus_id);

CREATE TABLE IF NOT EXISTS eval_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_id       TEXT NOT NULL,
    git_commit      TEXT NOT NULL,
    model_tier      TEXT NOT NULL,
    search_type     TEXT NOT NULL,
    k               INTEGER NOT NULL DEFAULT 5,
    started_at      TIMESTAMPTZ NOT NULL,
    completed_at    TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'queued'
                        CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    sample_count    INTEGER NOT NULL DEFAULT 0,
    baseline_run_id UUID REFERENCES eval_runs(id),
    report_json     JSONB       -- regression diff written by reporter.py
);

CREATE INDEX IF NOT EXISTS eval_runs_corpus_started ON eval_runs (corpus_id, started_at DESC);

CREATE TABLE IF NOT EXISTS eval_results (
    id                              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id                          UUID NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    sample_id                       UUID NOT NULL REFERENCES gold_samples(id),
    -- Retrieval metrics
    hit_rate                        FLOAT,
    mrr                             FLOAT,
    ndcg                            FLOAT,
    precision_at_k                  FLOAT,
    recall_at_k                     FLOAT,
    -- Generation metrics
    faithfulness                    FLOAT,
    answer_relevance                FLOAT,
    -- Correctness (requires ground truth)
    bleu_4                          FLOAT,
    rouge_1_f                       FLOAT,
    rouge_2_f                       FLOAT,
    rouge_l_f                       FLOAT,
    meteor                          FLOAT,
    bert_score_f                    FLOAT,
    semantic_similarity             FLOAT,
    -- Performance
    retrieval_ms                    INTEGER,
    llm_first_token_ms              INTEGER,
    generation_ms                   INTEGER,
    total_ms                        INTEGER,
    prompt_tokens                   INTEGER,
    completion_tokens               INTEGER,
    total_tokens                    INTEGER,
    estimated_cost_usd              FLOAT,
    cache_tier_hit                  TEXT,
    -- Confidence scoring
    mean_confidence                 FLOAT,
    min_confidence                  FLOAT,
    low_confidence_flag             BOOLEAN NOT NULL DEFAULT FALSE,
    -- Confidence-aware pipeline
    pipeline_status                 TEXT,
    abstention_layer                INTEGER,
    retrieval_aggregate_confidence  FLOAT,
    citation_trustworthy            BOOLEAN,
    judge_verdict                   TEXT,
    judge_confidence                FLOAT,
    false_abstention                BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS eval_results_run_id_idx ON eval_results (run_id);

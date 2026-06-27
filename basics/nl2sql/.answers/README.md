# NL2SQL Deep-Dive Answers

Detailed answers to every topic in [nl2sql_system_design.md](../nl2sql_system_design.md). One file per section.

## Files

| File | Section | Topics |
|------|---------|--------|
| [01_pipeline_architecture.md](01_pipeline_architecture.md) | Pipeline Architecture | End-to-end design, LLM vs deterministic boundary, schema linking, ambiguity handling |
| [02_schema_representation.md](02_schema_representation.md) | Schema Representation | Large schema retrieval, metadata enrichment, FK representation, warehouse vs OLTP |
| [03_accuracy_evaluation.md](03_accuracy_evaluation.md) | Accuracy and Evaluation | Ground truth construction, evaluation metrics, semantic error detection, shipping criteria |
| [04_multi_turn_context.md](04_multi_turn_context.md) | Multi-Turn Context | Co-reference resolution, conversation state, context reset triggers |
| [05_execution_safety_security.md](05_execution_safety_security.md) | Execution and Security | Prompt injection defence, RLS/CLS enforcement, pre-execution validation, cost guardrails |
| [06_latency_performance.md](06_latency_performance.md) | Latency and Performance | Latency levers, caching strategy, streaming trade-offs |
| [07_model_prompt_strategy.md](07_model_prompt_strategy.md) | Model and Prompt Strategy | Fine-tuning vs RAG vs few-shot, example selection, prompt compression, dialect handling |
| [08_feedback_loops.md](08_feedback_loops.md) | Feedback Loops | Implicit signals, edit signal extraction, safe model updates |
| [09_hard_tradeoffs.md](09_hard_tradeoffs.md) | Hard Trade-offs | Accuracy vs explainability, uncertainty surfacing, multi-persona design |
| [10_latency_slas.md](10_latency_slas.md) | Latency SLAs | SLA decomposition, retrieval tail latency, fast mode design |
| [11_vague_queries.md](11_vague_queries.md) | Vague Queries | Ambiguity scoring, assumption surfacing, correctness SLA, rejection threshold |
| [12_schema_drift.md](12_schema_drift.md) | Schema Drift | Migration impact, schema versioning, cache invalidation, backwards compatibility |
| [13_schema_subset_llm.md](13_schema_subset_llm.md) | Schema Subset Selection | Wrong table detection, retrieval pipeline failure modes, embedding strategy, dynamic k, multi-tenancy |
| [14_eval_pipeline.md](14_eval_pipeline.md) | Eval Pipeline Design | Component-level evaluation, adversarial test cases, CI/CD integration, regression detection, human review, sandbox + pilot testing |
| [15_query_intent_and_scope.md](15_query_intent_and_scope.md) | Query Intent and Scope Detection | Ambiguity detection (dimension scoring, schema-grounded), out-of-scope detection (unrelated, entity mismatch, data unavailability), pre-generation gate architecture |

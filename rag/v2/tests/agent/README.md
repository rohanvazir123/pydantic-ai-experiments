# tests/agent/

## Table of Contents

- [What This Is](#what-this-is)
- [Requirements](#requirements)

---

## What This Is

Agent and confidence-aware pipeline tests. Verifies the three-layer gate (retrieval → citation → judge), streaming SSE events, abstention on empty corpus, and multi-turn conversation context.

---

## Requirements

Full stack: PostgreSQL + Redis + Ollama with `llama3.2:3b`, `nomic-embed-text:latest`, and `qwen2.5:0.5b` (nano tier). NeuralFlow AI corpus must be ingested first.

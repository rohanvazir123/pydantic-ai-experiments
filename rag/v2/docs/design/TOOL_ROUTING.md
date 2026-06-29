# RAG v2 — LLM Tool Routing

The RAG agent exposes two retrieval tools. The LLM decides at runtime which to call (or both) based on the question type. This doc covers what the tools do, what signals trigger each one, and how the pre-call routing pipeline feeds into that decision.

---

## The Two Tools

Both are registered on the agent via `@agent.tool` in `knowledge/agent/agent.py`.

| Tool | Backing store | Returns |
|------|--------------|---------|
| `search_knowledge_base` | pgvector hybrid search (RRF) | Passage chunks with `[chunk_id]` anchors |
| `search_knowledge_graph` | Apache AGE (Cypher) | Entity/relationship list |

### `search_knowledge_base`

```python
async def search_knowledge_base(
    ctx: RunContext[RAGState],
    query: str,
    match_count: int | None = 5,
    search_type: str | None = "hybrid",
) -> str:
```

Calls `retriever.retrieve()` with the given query scoped to `corpus_ids` + `tenant_id`. Returns formatted chunks:

```
[chunk_id: abc123] Team Handbook (handbook.pdf)
PTO policy allows 15 days per year...
```

### `search_knowledge_graph`

```python
async def search_knowledge_graph(
    ctx: RunContext[RAGState],
    query: str,
    entity_type: str | None = None,
    limit: int | None = 15,
) -> str:
```

Calls `retriever._graph.query()`. Returns:

```
## Knowledge Graph — Entities
- [Person] Alice Smith — VP of Engineering, reports to CEO
- [Team] Platform — owns the infrastructure corpus
```

Returns `"Knowledge graph is not available for this corpus."` if no AGE store is configured.

---

## Routing Logic (System Prompt)

The `MAIN_SYSTEM_PROMPT` in `knowledge/agent/prompts.py` contains the routing rules the LLM follows:

```
3. If the provided context does not contain a chunk that supports a claim:
   a. Call search_knowledge_base with a more targeted or decomposed query FIRST.
   b. For questions about entities, relationships, or connections between things,
      call search_knowledge_graph instead.
   c. Only omit the claim if additional retrieval also returns nothing relevant.
```

In plain terms:

| Question type | Tool called |
|--------------|-------------|
| "What is the PTO policy?" | `search_knowledge_base` (passage lookup) |
| "Who is Alice's manager?" | `search_knowledge_graph` (relationship traversal) |
| "What does GLBA require and which team owns compliance?" | Both — passage for GLBA text, graph for team ownership |
| Context already answers the question | Neither — no tool call |

The LLM may call each tool multiple times per request (up to `retries=3` on the agent) using progressively decomposed queries if the first retrieval doesn't satisfy a claim.

---

## Pre-Call Routing Pipeline

Two lighter models run *before* the agent to route the request:

### 1. Model Router (`ROUTER_SYSTEM_PROMPT`)

Classifies the query to select the LLM tier and flags graph/multi-pass requirements:

```
simple   — factual, single-entity, single-hop → nano model (qwen2.5:0.5b)
moderate — multi-part, synthesis across sources → small model (llama3.2:3b)
complex  — multi-hop, reasoning chains, graph traversal → large model (llama3.1:70b)

requires_graph: true if the query asks about relationships or entity connections.
requires_multipass: true if the query spans multiple sub-questions.
```

`requires_graph: true` doesn't directly call `search_knowledge_graph` — it raises the model tier to `large` so a more capable model handles the agentic loop. The graph tool is still chosen by that model at runtime.

### 2. Intent Classifier (`INTENT_CLASSIFIER_PROMPT`)

Maps the query to one of five intents used to tune retrieval parameters:

| Intent | Description | Typical tool path |
|--------|-------------|-------------------|
| `factual` | Single fact or definition | `search_knowledge_base` |
| `comparison` | Two or more entities/options | `search_knowledge_base` (multiple queries) |
| `summarization` | High-level overview | `search_knowledge_base` (higher `match_count`) |
| `procedural` | How-to, step-by-step | `search_knowledge_base` |
| `relational` | Org structure, entity connections | `search_knowledge_graph` |

The intent result may also set a higher `k_multiplier` (more chunks) or `include_graph` flag that the pipeline passes into the agent's initial context before the LLM starts calling tools.

---

## Turn Sequence Examples

### Simple factual (no tool call)

Initial context already contains the answer:

| # | Role | Content |
|---|------|---------|
| 1 | system | `MAIN_SYSTEM_PROMPT` |
| 2 | user | `"What is NeuralFlow's mission?"` + pre-retrieved context chunks |
| 3 | assistant | `GenerationResult(answer="NeuralFlow AI builds… [chunk_id: abc]", ...)` |

### Passage retrieval (one tool call)

Context is thin; LLM retrieves more:

| # | Role | Content |
|---|------|---------|
| 1 | system | `MAIN_SYSTEM_PROMPT` |
| 2 | user | `"What are the GLBA data retention requirements?"` |
| 3 | assistant (tool call) | `search_knowledge_base("GLBA data retention requirements")` |
| 4 | tool result | `[chunk_id: xyz] GLBA Safeguards Rule…` |
| 5 | assistant | `GenerationResult(answer="GLBA requires… [xyz]", ...)` |

### Relationship query (graph tool)

| # | Role | Content |
|---|------|---------|
| 1 | system | `MAIN_SYSTEM_PROMPT` |
| 2 | user | `"Who manages the Platform team?"` |
| 3 | assistant (tool call) | `search_knowledge_graph("Platform team manager")` |
| 4 | tool result | `## Knowledge Graph — Entities\n- [Person] Bob Chen — Platform Lead…` |
| 5 | assistant | `GenerationResult(answer="Bob Chen leads the Platform team [chunk_id: …]", ...)` |

### Multi-hop (both tools)

| # | Role | Content |
|---|------|---------|
| 1 | system | `MAIN_SYSTEM_PROMPT` |
| 2 | user | `"What does GLBA require and which team at NeuralFlow owns compliance?"` |
| 3 | assistant (tool call) | `search_knowledge_base("GLBA requirements financial institutions")` |
| 4 | tool result | passage chunks |
| 5 | assistant (tool call) | `search_knowledge_graph("compliance team owner")` |
| 6 | tool result | entity list |
| 7 | assistant | `GenerationResult(answer="GLBA requires… [chunk1]. The Legal team owns compliance [chunk2].", ...)` |

---

## Safety Rails

| Rail | Mechanism |
|------|-----------|
| Max tool calls | `retries=3` on `PydanticAgent` — agent stops after 3 agentic iterations |
| Graph unavailable | `search_knowledge_graph` returns a string error; LLM falls back to `search_knowledge_base` |
| Nothing found | Both tools return `"No relevant information found."` — LLM must omit the unsupported claim |
| Citation check | `citation_check.is_trustworthy = False` if any claim in `GenerationResult` lacks a `[chunk_id]` |

---

## Key Files

| File | Role |
|------|------|
| `knowledge/agent/agent.py` | Tool definitions, `_build_agent()`, `RAGState` |
| `knowledge/agent/prompts.py` | `MAIN_SYSTEM_PROMPT` (tool routing rules), `ROUTER_SYSTEM_PROMPT`, `INTENT_CLASSIFIER_PROMPT` |
| `knowledge/retrieval/retriever.py` | `retrieve()` called by `search_knowledge_base` |
| `knowledge/store/graph.py` | AGE graph store called by `search_knowledge_graph` |

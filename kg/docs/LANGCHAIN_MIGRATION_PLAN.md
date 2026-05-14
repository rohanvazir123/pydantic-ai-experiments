# Migration Plan: pydantic-ai → LangChain (Ollama) in `kg/extraction_pipeline.py`

## What's actually being replaced

The pydantic-ai `Agent` here isn't doing agentic reasoning — it's a thin wrapper
around an Ollama LLM call with a system prompt. LangChain's native equivalent is
`ChatOllama.ainvoke(messages)`.

---

## Why `ChatOllama` over `ChatOpenAI`

The codebase already uses Ollama exclusively for KG extraction. LangChain ships a
dedicated `ChatOllama` class (`langchain-ollama` package) that is:

- **Explicit** — tied to Ollama, not a generic OpenAI-compatible shim
- **JSON mode** — `format="json"` forces structured output at the Ollama layer,
  eliminating the 21.5% JSON parse failure rate observed in the first real run
  (see `KG_FAQ.md`). `_parse_json` / `_clean_json` stay as a fallback but should
  rarely fire.
- **No API key** — Ollama is unauthenticated locally; no dummy key needed
- **`num_ctx` control** — explicit context window size, critical because Ollama
  defaults to 2048 tokens which is dangerously close to our worst-case prompt

---

## Context Length

### The problem

Ollama's default `num_ctx` is **2048 tokens**. Token budget for the most expensive
pass (pass 5 — validation, which sends chunk + entities + relationships + system prompt):

| Part | Chars | ~tokens (4 chars/token) |
|---|---|---|
| System prompt | ~900 | ~225 |
| Chunk text | 1500 | ~375 |
| Entities JSON | ~800 avg | ~200 |
| Relationships JSON | ~800 avg | ~200 |
| Expected output | ~600 | ~150 |
| **Total** | | **~1150** |

Looks safe — but spike cases (dense contracts, many entities, long evidence texts)
can push past 2048. `num_ctx=4096` gives comfortable headroom and must be set
explicitly — Ollama will not auto-expand.

### The fix — two layers

**Layer 1: set `num_ctx` explicitly on `ChatOllama`**

Add `KG_LLM_NUM_CTX` to settings (default 4096). Pass it to `ChatOllama`.
This controls how much KV-cache Ollama allocates per request.

**Layer 2: pre-call truncation guard in `_run_llm`**

Before invoking, estimate total prompt tokens. If over 80% of `num_ctx`,
truncate the *user prompt only* (never the system prompt — it carries the ontology).
Log a warning so the run log captures which chunks hit the limit.

```python
_CHARS_PER_TOKEN = 4
_OUTPUT_RESERVE  = 512   # tokens reserved for model output
_BUDGET_FRACTION = 0.8   # start truncating at 80% of num_ctx to leave headroom


def _truncate_to_budget(system_prompt: str, user_prompt: str, num_ctx: int) -> str:
    """Truncate user_prompt so total prompt stays within budget tokens."""
    budget_chars = int(num_ctx * _BUDGET_FRACTION - _OUTPUT_RESERVE) * _CHARS_PER_TOKEN
    available    = budget_chars - len(system_prompt)
    if len(user_prompt) <= available:
        return user_prompt
    logger.warning(
        "Prompt truncated: user_prompt %d → %d chars (num_ctx=%d)",
        len(user_prompt), available, num_ctx,
    )
    return user_prompt[:available]
```

`_run_llm` calls `_truncate_to_budget` before building the messages list.

---

## Scope of changes

### 1. `pyproject.toml`

Add `langchain-ollama`. Keep `pydantic-ai` — still used by `rag/agent/rag_agent.py`.

```toml
"langchain-ollama>=0.2.0",
```

### 2. `rag/config/settings.py`

Add one new setting:

```python
kg_llm_num_ctx: int = Field(
    default=4096,
    description="Ollama context window size (num_ctx). 4096 covers all 5 passes safely.",
)
```

### 3. `kg/extraction_pipeline.py`

#### Imports

| Remove | Add |
|---|---|
| `from pydantic_ai import Agent` | `from langchain_ollama import ChatOllama` |
| `from pydantic_ai.models.openai import OpenAIChatModel` | `from langchain_core.messages import HumanMessage, SystemMessage` |
| `from pydantic_ai.providers.openai import OpenAIProvider` | |

Also: replace the locally-defined `VALID_LABELS` / `VALID_REL_TYPES` frozensets
with imports from `kg.constants` (see Guardrails section — they're currently
duplicated and out of sync risk).

#### LLM factory

| | Old | New |
|---|---|---|
| Function | `_make_agent(system_prompt: str) -> Agent` | `_make_llm() -> ChatOllama` |
| Pattern | One agent per system prompt, baked in at construction | One shared client, system prompts passed at call time |
| Attributes | `self._entity_agent`, `self._rel_agent`, `self._hierarchy_agent`, `self._cross_ref_agent`, `self._validation_agent` | `self._llm` (single instance) |

```python
def _make_llm() -> ChatOllama:
    settings = load_settings()
    base_url  = (settings.kg_llm_base_url or settings.llm_base_url).rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return ChatOllama(
        model=settings.kg_llm_model or settings.llm_model,
        base_url=base_url,
        temperature=0,
        format="json",
        num_ctx=settings.kg_llm_num_ctx,
    )
```

#### Call site

| | Old | New |
|---|---|---|
| Function | `_run_agent(agent, user_prompt) -> dict` | `_run_llm(llm, system_prompt, user_prompt, num_ctx) -> dict` |
| Invoke | `await agent.run(prompt)` | `await llm.ainvoke([SystemMessage(...), HumanMessage(...)])` |
| Result | `result.output` | `response.content` |

```python
async def _run_llm(
    llm: ChatOllama, system_prompt: str, user_prompt: str, num_ctx: int
) -> dict:
    """Invoke Ollama, return parsed JSON dict. Returns {} on failure."""
    user_prompt = _truncate_to_budget(system_prompt, user_prompt, num_ctx)
    for attempt in range(4):
        try:
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            return _parse_json(response.content)
        except Exception as exc:
            if "429" in str(exc) or "rate_limit" in str(exc).lower():
                wait = min(2 ** attempt, 30)
                logger.info("Rate limited, retrying in %ds", wait)
                await asyncio.sleep(wait)
                continue
            logger.warning("LLM call failed: %s", exc)
            return {}
    return {}
```

#### `ExtractionPipeline.__init__`

```python
# Before
self._entity_agent     = _make_agent(_ENTITY_PROMPT)
self._rel_agent        = _make_agent(_RELATIONSHIP_PROMPT)
self._hierarchy_agent  = _make_agent(_HIERARCHY_PROMPT)
self._cross_ref_agent  = _make_agent(_CROSS_CONTRACT_PROMPT)
self._validation_agent = _make_agent(_VALIDATION_PROMPT)

# After
settings      = load_settings()
self._llm     = _make_llm()
self._num_ctx = settings.kg_llm_num_ctx
```

#### `_pass_*` methods

```python
# Before
data = await _run_agent(self._entity_agent, f"Contract text:\n\n{chunk}")

# After
data = await _run_llm(self._llm, _ENTITY_PROMPT, f"Contract text:\n\n{chunk}", self._num_ctx)
```

Same pattern for all five passes.

### 4. No other files change

`rag/agent/rag_agent.py`, all tests, all other docs — untouched.

---

## Guardrails audit

### Extraction path — what exists

| Guardrail | Where | What it does |
|---|---|---|
| Label whitelist | `ExtractedEntity.safe_label()` | Validates label against `VALID_LABELS`; falls back to `"Clause"` |
| Relationship whitelist | `ExtractedRelationship.safe_rel_type()` | Returns `None` for unknown types; filtered out before Bronze write |
| Confidence threshold | Silver staging + `_pass_*` methods | Drops entities/relationships below 0.7 |
| JSON repair | `_parse_json` / `_clean_json` | Handles 4 llama3 failure modes; returns `{}` on total failure |
| Pass 5 validation | `_pass_validate()` | Second LLM opinion — filters hallucinated/unsupported relationships |
| Bronze dedup | `UNIQUE (contract_id, chunk_index, model_version)` | Idempotent re-runs |
| Silver dedup | `DISTINCT ON (label, normalized_name)` | Merges duplicate entities across chunks |
| Pydantic validation | Try/except in all `_pass_*` methods | Silently drops malformed LLM output |

### Extraction path — gaps

| Gap | Risk | Fix |
|---|---|---|
| **Context overflow** | Ollama silently truncates at `num_ctx=2048`; output is garbled | Set `num_ctx=4096` + `_truncate_to_budget()` — covered in this plan |
| **Duplicate ontology** | `VALID_LABELS` / `VALID_REL_TYPES` defined in both `extraction_pipeline.py` and `kg/constants.py` — drift risk | Import from `kg.constants` only — fix in this migration |
| **No output size cap** | Abnormally large LLM output could cause memory pressure | Add `max_tokens` / `num_predict` to `ChatOllama` (e.g. 1024) |
| **Prompt injection** | Contract text goes into prompts unsanitized | Low practical risk (legal text, not user input), but worth noting |

### Retrieval path — what exists

| Guardrail | Where | What it does |
|---|---|---|
| Label whitelist | `AgeGraphStore._safe_label()` | Whitelist against `VALID_LABELS`; falls back to `"Clause"` before Cypher interpolation |
| Relationship whitelist | `AgeGraphStore._safe_rel_type()` | Returns `None` for unknown types; write is skipped |
| Read-only Cypher guard | `AgeGraphStore.run_cypher_query()` | Regex blocks `CREATE / MERGE / SET / DELETE / REMOVE / DROP / DETACH` |

### Retrieval path — gaps

| Gap | Risk | Fix |
|---|---|---|
| **Entity name in Cypher** | `normalized_name` and `name` are interpolated into Cypher strings via f-string — a contract party named `}) DETACH DELETE (` would break the query | Escape or parameterize the name value in `upsert_entity` |
| **No query result size cap** | `run_cypher_query` has no `LIMIT` enforcement; a MATCH with no filter returns entire graph | Inject `LIMIT 500` if no LIMIT clause is present |
| **Substring search not escaped** | `search_entities` / `search_as_context` pass the query into `CONTAINS {query_esc}` — verify escaping is applied | Confirm `_escape_for_cypher()` exists and covers apostrophes, backslashes |

---

## Configuration (`.env`)

```
KG_LLM_MODEL=llama3.1:8b
KG_LLM_BASE_URL=http://localhost:11434/v1   # /v1 stripped automatically
KG_LLM_NUM_CTX=4096                         # new — Ollama context window
KG_CONFIDENCE_THRESHOLD=0.7
```

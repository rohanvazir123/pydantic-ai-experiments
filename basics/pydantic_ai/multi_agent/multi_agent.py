"""
Multi-Agent RAG: Planner + Executor Pattern
============================================

This file demonstrates a two-tier multi-agent architecture for RAG, using the
same PostgreSQL/pgvector knowledge base as the main RAG system.

Architecture
------------

    User question
         │
         ▼
    ┌─────────────┐
    │   PLANNER   │  Decomposes the question into focused sub-queries.
    │   agent     │  Returns a typed QueryPlan (list of SubQuery objects).
    └──────┬──────┘
           │  asyncio.gather — all sub-queries run in parallel
           ▼
    ┌──────┬──────┬──────┐
    │ EXE  │ EXE  │ EXE  │  One Executor agent per sub-query.
    │  1   │  2   │  3   │  Each retrieves from the KB and answers its slice.
    └──────┴──────┴──────┘
           │  sub-answers collected
           ▼
    ┌─────────────┐
    │ SYNTHESIZER │  The Planner agent in a second turn — merges sub-answers
    │   (planner) │  into a single, cited final response.
    └─────────────┘

Why this beats a single agent for complex RAG questions
-------------------------------------------------------
A single agent making 4 sequential tool calls takes 4× the latency. Parallel
executors cut that to ~1× the latency of the slowest sub-query. More
importantly, each executor gets a tightly-scoped prompt and retrieval context
rather than a bloated single context window mixing unrelated chunks.

Run
---
    python basics/pydantic_ai/multi_agent.py
"""

import asyncio
import os
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

# ---------------------------------------------------------------------------
# Shared LLM — swap for "openai:gpt-4o" or any OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

_MODEL = os.getenv("LLM_MODEL", "ollama:llama3.1:8b")
_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
_API_KEY = os.getenv("LLM_API_KEY", "ollama")

# Pydantic AI uses "provider:model" strings.  For Ollama we pass provider
# settings explicitly via OpenAIModel so we can set base_url and api_key.
_llm = OpenAIModel(
    model_name=_MODEL.split(":", 1)[-1] if ":" in _MODEL else _MODEL,
    base_url=_BASE_URL,
    api_key=_API_KEY,
)


# ---------------------------------------------------------------------------
# Shared stub retriever — replace with rag.retrieval.retriever.Retriever
# ---------------------------------------------------------------------------

async def _retrieve(query: str, top_k: int = 3) -> list[dict[str, str]]:
    """Stub retriever that returns fake chunks.

    In production wire this to:
        from rag.retrieval.retriever import Retriever
        retriever = Retriever()
        results = await retriever.retrieve(query, match_count=top_k)
        return [{"source": r.document_source, "content": r.content} for r in results]
    """
    return [
        {"source": f"contract_{i}.md", "content": f"[stub] Relevant clause for '{query}' — chunk {i}"}
        for i in range(1, top_k + 1)
    ]


# ---------------------------------------------------------------------------
# Typed data models
# ---------------------------------------------------------------------------

class SubQuery(BaseModel):
    """A single focused retrieval unit produced by the Planner."""
    id: int = Field(description="Sequential identifier (1-based)")
    query: str = Field(description="Self-contained retrieval query, specific enough for vector search")
    focus: str = Field(description="What aspect of the user question this sub-query addresses")


class QueryPlan(BaseModel):
    """Structured output from the Planner agent."""
    original_question: str
    sub_queries: list[SubQuery] = Field(description="2-5 focused sub-queries that together cover the question")
    reasoning: str = Field(description="One sentence explaining the decomposition strategy")


class SubAnswer(BaseModel):
    """Result produced by one Executor agent run."""
    sub_query_id: int
    query: str
    answer: str
    sources: list[str]


# ---------------------------------------------------------------------------
# PLANNER agent
# Produces a QueryPlan — structured, typed, no free-form text
# ---------------------------------------------------------------------------

planner_agent: Agent[None, QueryPlan] = Agent(
    _llm,
    result_type=QueryPlan,
    system_prompt="""You are a query planner for a legal contract RAG system.

Your job: decompose a complex user question into 2-5 focused sub-queries that
can each be answered by a single vector search over contract chunks.

Rules:
- Each sub-query must be self-contained and specific (good for embedding)
- Together, the sub-queries must cover every aspect of the original question
- Prefer narrow sub-queries over broad ones — narrow retrieval = less noise
- Do NOT attempt to answer the question yourself
""",
)


# ---------------------------------------------------------------------------
# EXECUTOR agent
# Receives one sub-query + retrieved chunks, returns a focused answer
# ---------------------------------------------------------------------------

class ExecutorDeps(BaseModel):
    """Dependencies injected into each Executor agent run."""
    sub_query: SubQuery
    retrieved_chunks: list[dict[str, str]]


executor_agent: Agent[ExecutorDeps, str] = Agent(
    _llm,
    deps_type=ExecutorDeps,
    system_prompt="""You are a contract analysis agent.

You will be given:
  1. A focused sub-query
  2. Retrieved contract chunks relevant to that sub-query

Answer ONLY the sub-query using ONLY the provided chunks.
Cite each source as [Source: filename].
If the chunks do not contain the answer, say "Not found in retrieved context."
Be concise — 2-4 sentences maximum.
""",
)


@executor_agent.tool
async def get_relevant_chunks(ctx: RunContext[ExecutorDeps]) -> str:
    """Retrieve contract chunks for the sub-query.

    Returns the pre-fetched chunks injected via ExecutorDeps so the agent
    has access to the retrieved context without making an extra call.
    """
    chunks = ctx.deps.retrieved_chunks
    if not chunks:
        return "No relevant chunks found."
    parts = [f"[Source: {c['source']}]\n{c['content']}" for c in chunks]
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# SYNTHESIZER  (re-uses planner_agent in a second turn)
# Takes all sub-answers and produces the final cited response
# ---------------------------------------------------------------------------

synthesizer_agent: Agent[None, str] = Agent(
    _llm,
    system_prompt="""You are a synthesis agent for a legal contract RAG system.

You receive a set of focused sub-answers from specialist retrieval agents.
Your job: merge them into a single, coherent, well-cited answer to the
original question.

Rules:
- Preserve all [Source: ...] citations from sub-answers
- Do not add information that is not in the sub-answers
- Resolve any contradictions by noting them explicitly
- Keep the final answer under 200 words unless detail is requested
""",
)


# ---------------------------------------------------------------------------
# Multi-agent pipeline
# ---------------------------------------------------------------------------

async def run_multi_agent_rag(question: str) -> str:
    """
    Full planner → parallel executors → synthesizer pipeline.

    Steps
    -----
    1. Planner decomposes the question into a typed QueryPlan.
    2. Each sub-query is sent to the retriever (all in parallel via asyncio.gather).
    3. Each sub-query + its chunks is handled by a separate Executor agent
       (all in parallel via asyncio.gather).
    4. Synthesizer merges all sub-answers into the final response.

    Concurrency model
    -----------------
    Steps 2 and 3 use asyncio.gather — true async concurrency.  Each Executor
    agent is an independent coroutine: its own LLM call, its own context
    window, its own tool execution.  There is no shared mutable state between
    executors.
    """
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print('='*60)

    # ── Step 1: Plan ────────────────────────────────────────────────────────
    print("\n[1/4] Planner decomposing question...")
    plan_result = await planner_agent.run(question)
    plan: QueryPlan = plan_result.output

    print(f"  Reasoning : {plan.reasoning}")
    print(f"  Sub-queries ({len(plan.sub_queries)}):")
    for sq in plan.sub_queries:
        print(f"    [{sq.id}] {sq.query}  ({sq.focus})")

    # ── Step 2: Retrieve in parallel ────────────────────────────────────────
    print("\n[2/4] Retrieving chunks for all sub-queries in parallel...")
    chunk_lists: list[list[dict[str, str]]] = await asyncio.gather(
        *[_retrieve(sq.query) for sq in plan.sub_queries]
    )
    print(f"  Retrieved {sum(len(c) for c in chunk_lists)} chunks total "
          f"across {len(plan.sub_queries)} sub-queries")

    # ── Step 3: Execute in parallel ─────────────────────────────────────────
    print("\n[3/4] Running executor agents in parallel...")

    async def _run_executor(sq: SubQuery, chunks: list[dict[str, str]]) -> SubAnswer:
        deps = ExecutorDeps(sub_query=sq, retrieved_chunks=chunks)
        result = await executor_agent.run(
            f"Answer this sub-query using the chunks available via your tool:\n\n{sq.query}",
            deps=deps,
        )
        sources = [c["source"] for c in chunks]
        return SubAnswer(
            sub_query_id=sq.id,
            query=sq.query,
            answer=result.output,
            sources=sources,
        )

    sub_answers: list[SubAnswer] = await asyncio.gather(
        *[_run_executor(sq, chunks) for sq, chunks in zip(plan.sub_queries, chunk_lists)]
    )

    print("  Sub-answers received:")
    for sa in sub_answers:
        print(f"    [{sa.sub_query_id}] {sa.answer[:80]}...")

    # ── Step 4: Synthesize ──────────────────────────────────────────────────
    print("\n[4/4] Synthesizer merging sub-answers...")

    synthesis_prompt = f"""Original question: {question}

Sub-answers from specialist agents:

""" + "\n\n".join(
        f"[Sub-query {sa.sub_query_id}: {sa.query}]\n{sa.answer}"
        for sa in sub_answers
    )

    final_result = await synthesizer_agent.run(synthesis_prompt)
    final_answer: str = final_result.output

    print("\n" + "─"*60)
    print("FINAL ANSWER:")
    print("─"*60)
    print(final_answer)
    return final_answer


# ---------------------------------------------------------------------------
# Sequential (non-parallel) variant — for comparison
# ---------------------------------------------------------------------------

async def run_sequential_rag(question: str) -> str:
    """
    Same pipeline but executors run one after the other.

    Use this to benchmark latency difference vs parallel execution.
    With 4 sub-queries and 500ms LLM latency each:
      Sequential: ~4 × 500ms = ~2000ms
      Parallel:   ~1 × 500ms =  ~500ms  (limited by slowest sub-query)
    """
    plan_result = await planner_agent.run(question)
    plan: QueryPlan = plan_result.output

    sub_answers: list[SubAnswer] = []
    for sq in plan.sub_queries:
        chunks = await _retrieve(sq.query)
        deps = ExecutorDeps(sub_query=sq, retrieved_chunks=chunks)
        result = await executor_agent.run(sq.query, deps=deps)
        sub_answers.append(SubAnswer(
            sub_query_id=sq.id,
            query=sq.query,
            answer=result.output,
            sources=[c["source"] for c in chunks],
        ))

    synthesis_prompt = "\n\n".join(
        f"[{sa.sub_query_id}] {sa.query}\n{sa.answer}" for sa in sub_answers
    )
    result = await synthesizer_agent.run(
        f"Original: {question}\n\nSub-answers:\n{synthesis_prompt}"
    )
    return result.output


# ---------------------------------------------------------------------------
# Passing message history between agents (multi-turn composition)
# ---------------------------------------------------------------------------

async def demo_message_history_passthrough() -> None:
    """
    Demonstrates passing message_history from one agent run to another.

    Pattern: the Planner's full conversation history (including its reasoning)
    is passed to the Synthesizer so it retains context about the original plan.

    result.all_messages() returns the complete [user, assistant, tool, ...] list
    in Pydantic AI's message format — safe to pass directly to the next agent.
    """
    print("\n=== Message history passthrough ===")

    plan_result = await planner_agent.run(
        "What are the liability caps in the Amazon and Microsoft contracts?"
    )

    # The synthesizer receives the planner's full conversation as prior context
    synth_result = await synthesizer_agent.run(
        "Now synthesize the following sub-answers into a final response:\n"
        "[stub sub-answers for demo]",
        message_history=plan_result.all_messages(),
    )
    print(f"Synthesizer output (with planner history):\n{synth_result.output}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def main() -> Any:
    question = (
        "What are the termination clauses and payment terms "
        "in the Amazon and Google contracts, and how do they differ?"
    )
    await run_multi_agent_rag(question)
    print("\n\n--- Sequential variant (same question, for latency comparison) ---")
    await run_sequential_rag(question)
    print("\n")
    await demo_message_history_passthrough()


if __name__ == "__main__":
    asyncio.run(main())

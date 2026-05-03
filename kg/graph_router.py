"""
Rule-based graph router.

Maps a natural-language question to the subset of logical graph schemas
required to answer it.  Never calls an LLM — latency is near-zero and
routing is deterministic.

Why rule-based (not LLM):
  Schema selection is a small, well-defined decision with crisp linguistic
  signals.  An LLM would add ~1 s of latency, a model dependency, and
  non-determinism for no practical gain — identical reasoning to
  IntentClassifier in rag/retrieval/intent_classifier.py.

Usage:
    from kg.graph_router import GraphRouter, GraphType

    router = GraphRouter()
    types  = router.route("Which parties indemnify each other?")
    # → [GraphType.ENTITY]
"""

from __future__ import annotations

import re

from kg.schemas import GraphType


# ---------------------------------------------------------------------------
# Compiled regex patterns — one list per logical graph
# ---------------------------------------------------------------------------

_ENTITY: list[re.Pattern[str]] = [re.compile(p, re.I) for p in [
    r"\bpart(?:y|ies)\b",
    r"\bjurisdiction\b",
    r"\bgoverning\s+law\b",
    r"\bsigned\s+by\b",
    r"\bindemni(?:f|ty|fication)\b",
    r"\bterminat(?:e|ion|ing)\b",
    r"\bconfidential(?:ity)?\b",
    r"\bpayment\s+term\b",
    r"\bobligation\b",
    r"\blimit(?:s|ation)?\s+(?:of\s+)?liabilit",
    r"\brenewal\b",
    r"\bliabilit(?:y|ies)\b",
    r"\bcontract(?:s|ing|or|ual)?\b",
    r"\beffective\s+date\b",
    r"\bexpir(?:ation|y)\s+date\b",
    r"\bgovernin[g]?\s+law\b",
    r"\bdisclose[sd]?\b",
]]

_HIERARCHY: list[re.Pattern[str]] = [re.compile(p, re.I) for p in [
    r"\bsections?\b",
    r"\bparagraphs?\b",
    r"\bsubsections?\b",
    r"\bheadings?\b",
    r"\bdocument\s+structur",
    r"\bhierarch",
    r"\bclause\s+text\b",
    r"\bhas_(?:section|clause|chunk)\b",
    r"\bchunks?\s+of\b",
    r"\bstructur(?:e|al)\b",
]]

_LINEAGE: list[re.Pattern[str]] = [re.compile(p, re.I) for p in [
    r"\bamend(?:s|ment|ed|ing)?\b",
    r"\bsupersed(?:e|es|ed|ing)\b",
    r"\brepla(?:ce|ces|ced|cing)\b",
    r"\bincorporat(?:e|es|ed|ing)\s+by\s+reference\b",
    r"\blineage\b",
    r"\breference(?:s|d)?\s+document\b",
    r"\bprevious\s+(?:version|contract)\b",
    r"\boriginal\s+agreement\b",
    r"\battache[sd]?\b",
    r"\bsupersession\b",
]]

_RISK: list[re.Pattern[str]] = [re.compile(p, re.I) for p in [
    r"\brisk\b",
    r"\bcompliance\s+gap\b",
    r"\bmissing\s+clause\b",
    r"\bexposure\b",
    r"\bvulnerabilit",
    r"\bincreases?\s+risk\b",
    r"\bgap\s+(?:in|analysis)\b",
    r"\bnot\s+have\s+(?:an?\s+)?(?:indemnit|liabilit|terminat)",
    r"\bno\s+(?:indemnit|liabilit|terminat)",
    r"\bcauses?\s+risk\b",
]]

_ORDERED: list[tuple[GraphType, list[re.Pattern[str]]]] = [
    (GraphType.ENTITY,    _ENTITY),
    (GraphType.HIERARCHY, _HIERARCHY),
    (GraphType.LINEAGE,   _LINEAGE),
    (GraphType.RISK,      _RISK),
]


class GraphRouter:
    """
    Routes a natural-language question to the relevant graph type(s).

    Always returns at least ``[GraphType.ENTITY]`` — the most general subgraph
    and the safe default when no specific signals are detected.
    """

    def route(self, query: str) -> list[GraphType]:
        """
        Return the list of graph types relevant for *query*.

        Stable output order: ENTITY → HIERARCHY → LINEAGE → RISK.
        """
        matched: set[GraphType] = set()
        for graph_type, patterns in _ORDERED:
            if any(p.search(query) for p in patterns):
                matched.add(graph_type)

        matched.add(GraphType.ENTITY)  # always include the default subgraph

        return [gt for gt, _ in _ORDERED if gt in matched]

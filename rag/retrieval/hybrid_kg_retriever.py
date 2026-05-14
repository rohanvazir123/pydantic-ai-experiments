"""
Hybrid KG + text retriever.

Architecture:

    USER QUERY
         │
         ▼
    Intent Classification
         │
    +----+----+
    │         │
    ▼         ▼
  Semantic   Structured
  Retrieval  Reasoning
  (tsvector  (KG / AGE /
  + pgvector  SQL)
  + RRF)
    │         │
    +----+----+
         │
         ▼
    Context Fusion Layer
         │
         ▼
    Final LLM Reasoning

Semantic and Structured paths run in parallel via asyncio.gather.
Context Fusion concatenates KG facts (structured) above text passages
(semantic) so the LLM sees both in one context block.
"""

import asyncio
import logging
from dataclasses import dataclass, field

from rag.ingestion.models import SearchResult
from rag.retrieval.intent_classifier import IntentClassifier, QueryIntent

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    query: str
    intent: QueryIntent
    text_chunks: list[SearchResult] = field(default_factory=list)
    kg_facts: list[dict] = field(default_factory=list)
    fused_context: str = ""


class HybridKGRetriever:
    """
    Combines semantic retrieval (vector + BM25 + RRF) with KG structured
    reasoning (AGE Cypher entity/relationship lookup) and fuses the results
    into a single LLM context block.

    Parameters
    ----------
    retriever:
        Existing Retriever instance — handles vector + BM25 + RRF.
    kg_store:
        AgeGraphStore — provides search_entities,
        get_related_entities, search_as_context.
    classifier:
        IntentClassifier override (default: IntentClassifier()).
    """

    def __init__(self, retriever, kg_store, classifier: IntentClassifier | None = None):
        self._retriever = retriever
        self._kg = kg_store
        self._classifier = classifier or IntentClassifier()

    async def retrieve(
        self,
        query: str,
        match_count: int = 5,
        intent: QueryIntent | None = None,
    ) -> HybridResult:
        """
        Run intent classification, execute both paths in parallel, fuse results.

        Args:
            query:       Natural-language question.
            match_count: Number of text chunks to return from semantic path.
            intent:      Override intent classification (optional).

        Returns:
            HybridResult with .fused_context ready for LLM consumption.
        """
        if intent is None:
            intent = self._classifier.classify(query)

        logger.info("[HYBRID] query=%r intent=%s", query, intent)

        semantic_coro = (
            self._retriever.retrieve(query, match_count)
            if self._classifier.needs_semantic(intent)
            else None
        )
        structured_coro = (
            self._structured_retrieve(query)
            if self._classifier.needs_structured(intent)
            else None
        )

        text_chunks: list[SearchResult] = []
        kg_facts: list[dict] = []

        if semantic_coro and structured_coro:
            text_chunks, kg_facts = await asyncio.gather(semantic_coro, structured_coro)
        elif semantic_coro:
            text_chunks = await semantic_coro
        elif structured_coro:
            kg_facts = await structured_coro

        logger.info(
            "[HYBRID] done: %d text chunks, %d KG facts",
            len(text_chunks),
            len(kg_facts),
        )

        fused = _fuse(query, intent, text_chunks, kg_facts)
        return HybridResult(
            query=query,
            intent=intent,
            text_chunks=text_chunks,
            kg_facts=kg_facts,
            fused_context=fused,
        )

    async def _structured_retrieve(self, query: str) -> list[dict]:
        """Search KG entities + relationships and return as structured fact list."""
        try:
            entities = await self._kg.search_entities(query, limit=10)
        except Exception as e:
            logger.warning("[HYBRID] KG entity search failed: %s", e)
            return []

        facts: list[dict] = []
        for ent in entities:
            facts.append(
                {
                    "type": "entity",
                    "id": ent.get("id", ""),
                    "name": ent.get("name", ""),
                    "entity_type": ent.get("entity_type", ""),
                    "document_id": ent.get("document_id", ""),
                    "document_title": ent.get("document_title", ""),
                }
            )

        # Fetch relationships for each matched entity (cap to avoid fan-out)
        for ent_fact in facts[:5]:
            eid = ent_fact.get("id")
            if not eid:
                continue
            try:
                rels = await self._kg.get_related_entities(eid, limit=5)
                for rel in rels:
                    facts.append(
                        {
                            "type": "relationship",
                            "source_name": ent_fact["name"],
                            "source_type": ent_fact["entity_type"],
                            "relation": rel.get("relationship_type", "RELATED_TO"),
                            "target_name": rel.get("name", ""),
                            "target_type": rel.get("entity_type", ""),
                            "document_id": rel.get("document_id", ""),
                        }
                    )
            except Exception as e:
                logger.debug("[HYBRID] get_related_entities failed for %s: %s", eid, e)

        return facts


def _fuse(
    query: str,
    intent: QueryIntent,
    text_chunks: list[SearchResult],
    kg_facts: list[dict],
) -> str:
    """Combine KG facts and text passages into a single LLM context block."""
    parts: list[str] = []

    if kg_facts:
        parts.append("## Knowledge Graph Facts")
        entities = [f for f in kg_facts if f["type"] == "entity"]
        rels = [f for f in kg_facts if f["type"] == "relationship"]

        if entities:
            parts.append(f"\nEntities ({len(entities)} matched):")
            for e in entities[:15]:
                title = f"  ({e['document_title']})" if e.get("document_title") else ""
                parts.append(f"  - [{e['entity_type']}] {e['name']}{title}")

        if rels:
            parts.append(f"\nRelationships ({len(rels)} found):")
            for r in rels[:20]:
                parts.append(
                    f"  - [{r['source_type']}] {r['source_name']}"
                    f" —[{r['relation']}]→ "
                    f"[{r['target_type']}] {r['target_name']}"
                )

    if text_chunks:
        parts.append("\n## Relevant Text Passages")
        for i, chunk in enumerate(text_chunks, 1):
            parts.append(
                f"\n[{i}] {chunk.document_title} (relevance: {chunk.similarity:.2f})\n"
                f"{chunk.content}"
            )

    if not parts:
        return "No relevant information found for this query."

    return "\n".join(parts)

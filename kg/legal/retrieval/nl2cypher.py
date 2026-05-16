"""
NL→Cypher converter — deterministic, no LLM.

Pipeline:
  1. IntentParser maps the query to (intent, params) using regex.
  2. QUERY_CAPABILITIES[intent] builds a safe Cypher string from params.

No model calls, no prompt injection surface, no schema drift.
"""
from __future__ import annotations

import logging

from kg.legal.intent_parser import IntentMatch, IntentParser
from kg.legal.query_builder import QUERY_CAPABILITIES

logger = logging.getLogger(__name__)


class NL2CypherConverter:
    """Convert a natural-language question to an openCypher MATCH query."""

    def __init__(self) -> None:
        self._parser = IntentParser()

    async def convert(self, question: str, schema: str = "") -> str:
        """
        Return a Cypher MATCH query for *question*.

        *schema* is accepted for API compatibility but unused — schema
        knowledge is encoded directly in the builder functions.
        """
        match: IntentMatch = self._parser.parse(question)
        builder = QUERY_CAPABILITIES[match.intent]
        cypher = builder(match.params)
        logger.debug(
            "[nl2cypher] intent=%r params=%r cypher=%r",
            match.intent, match.params, cypher,
        )
        return cypher

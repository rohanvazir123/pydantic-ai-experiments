# Copyright 2024 The Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NL→Cypher converter — deterministic, no LLM.

Pipeline:
  1. IntentParser maps the query to (intent, params) using regex.
  2. QUERY_CAPABILITIES[intent] builds a safe Cypher string from params.

No model calls, no prompt injection surface, no schema drift.
"""
from __future__ import annotations

import logging

from kg.legal.retrieval.intent_parser import IntentMatch, IntentParser
from kg.legal.retrieval.query_builder import QUERY_CAPABILITIES

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

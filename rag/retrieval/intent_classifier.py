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
Intent classifier for hybrid KG + text retrieval routing.

Three intents:
  STRUCTURED  — KG/SQL only (count, aggregate, distribution queries)
  SEMANTIC    — text chunks only (pure text retrieval)
  HYBRID      — both paths in parallel (default for most questions)
"""

import re
from enum import Enum


class QueryIntent(str, Enum):
    SEMANTIC = "semantic"
    STRUCTURED = "structured"
    HYBRID = "hybrid"


_ANALYTICAL = re.compile(
    r"\b(how many|count\b|total number|on average|average number|distribution of|"
    r"ratio of|frequency of|percentage of|per contract|most common|top[\s\-]?\d+|"
    r"highest|lowest|which year|aggregate|median|variance|"
    r"how often|what proportion|what percentage)\b",
    re.IGNORECASE,
)

_GRAPH_TRAVERSAL = re.compile(
    r"\b(starting from|traverse|two hops?|multi.?hop|subgraph|shortest path|"
    r"connected to|linked to|hops? away|path between|reachable from)\b",
    re.IGNORECASE,
)


class IntentClassifier:
    """Classify query intent to route to the appropriate retrieval path(s)."""

    def classify(self, query: str) -> QueryIntent:
        if _ANALYTICAL.search(query) and not _needs_text(query):
            return QueryIntent.STRUCTURED
        return QueryIntent.HYBRID

    def needs_semantic(self, intent: QueryIntent) -> bool:
        return intent in (QueryIntent.SEMANTIC, QueryIntent.HYBRID)

    def needs_structured(self, intent: QueryIntent) -> bool:
        return intent in (QueryIntent.STRUCTURED, QueryIntent.HYBRID)


def _needs_text(query: str) -> bool:
    """Return True if the analytical query also needs clause text for context."""
    text_signals = re.compile(
        r"\b(exact text|exact language|exact wording|"
        r"what does .{0,30} say|what does .{0,30} state|"
        r"how does .{0,30} read|explain the|describe the|"
        r"retrieve the text|show me the|full text|"
        r"say\b|says\b|states?\b|reads?\b|wording)\b",
        re.IGNORECASE,
    )
    return bool(text_signals.search(query))

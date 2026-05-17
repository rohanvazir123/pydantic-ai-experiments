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

"""Query processing and transformation for improved retrieval.

This module provides different query processing strategies:
- LLMQueryExpander: Generates alternative query phrasings
- HyDEProcessor: Hypothetical Document Embeddings
- MultiQueryProcessor: Combines multiple query variations
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class LLMQueryExpander(BaseQueryProcessor):
    """
    Expands queries using LLM to generate alternative phrasings.

    This helps capture different ways users might express the same intent,
    improving recall by matching documents that use different terminology.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        num_expansions: int = 3,
    ):
        """
        Initialize query expander.

        Args:
            model: LLM model name
            base_url: API base URL
            api_key: API key
            num_expansions: Number of alternative queries to generate
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.num_expansions = num_expansions
        self._client = None

    def _get_client(self):
        """Get or create the async OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._client

    async def process(self, query: str) -> dict[str, Any]:
        """
        Expand query into multiple variations.

        Args:
            query: Original search query

        Returns:
            Dictionary with original and expanded queries
        """
        expansions = await self.expand(query)
        return {
            "original": query,
            "expansions": expansions,
            "all_queries": [query] + expansions,
        }

    async def expand(self, query: str) -> list[str]:
        """
        Generate alternative phrasings of the query.

        Args:
            query: Original search query

        Returns:
            List of alternative query phrasings
        """
        client = self._get_client()

        prompt = f"""Generate {self.num_expansions} alternative phrasings of this search query.
Each alternative should capture the same intent but use different words or structure.

Original query: "{query}"

Return ONLY the alternative queries, one per line, without numbering or bullets."""

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
            )

            content = response.choices[0].message.content.strip()
            expansions = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and line.strip() != query
            ]

            logger.info(f"Generated {len(expansions)} query expansions")
            return expansions[: self.num_expansions]

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return []


class MultiQueryProcessor(BaseQueryProcessor):
    """
    Combines multiple query processing strategies.

    Generates multiple query variations and retrieves results for each,
    then merges them using reciprocal rank fusion or other strategies.
    """

    def __init__(
        self,
        expander: LLMQueryExpander | None = None,
        hyde: HyDEProcessor | None = None,
        use_expansion: bool = True,
        use_hyde: bool = True,
    ):
        """
        Initialize multi-query processor.

        Args:
            expander: Query expander instance
            hyde: HyDE processor instance
            use_expansion: Whether to use query expansion
            use_hyde: Whether to use HyDE
        """
        self.expander = expander or LLMQueryExpander()
        self.hyde = hyde or HyDEProcessor()
        self.use_expansion = use_expansion
        self.use_hyde = use_hyde

    async def process(self, query: str) -> dict[str, Any]:
        """
        Process query using multiple strategies.

        Args:
            query: Original search query

        Returns:
            Dictionary with all query variations and embeddings
        """
        result: dict[str, Any] = {
            "original": query,
            "queries": [query],
            "embeddings": {},
        }

        tasks = []

        if self.use_expansion:
            tasks.append(("expansion", self.expander.process(query)))

        if self.use_hyde:
            tasks.append(("hyde", self.hyde.process(query)))

        # Run processors concurrently
        if tasks:
            task_results = await asyncio.gather(
                *[t[1] for t in tasks],
                return_exceptions=True,
            )

            for (name, _), task_result in zip(tasks, task_results):
                if isinstance(task_result, Exception):
                    logger.warning(f"{name} processing failed: {task_result}")
                    continue

                if isinstance(task_result, dict):
                    if name == "expansion" and "all_queries" in task_result:
                        result["queries"].extend(task_result["expansions"])
                        result["expansion_result"] = task_result

                    elif name == "hyde":
                        result["hyde_result"] = task_result
                        result["embeddings"]["hyde"] = task_result.get("hyde_embedding")

        # Deduplicate queries
        result["queries"] = list(dict.fromkeys(result["queries"]))

        logger.info(
            f"Multi-query processing complete: {len(result['queries'])} queries"
        )
        return result


class QueryDecomposer(BaseQueryProcessor):
    """
    Decomposes complex queries into simpler sub-queries.

    Useful for multi-hop reasoning where a single query requires
    information from multiple sources.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ):
        """
        Initialize query decomposer.

        Args:
            model: LLM model name
            base_url: API base URL
            api_key: API key
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Get or create the async OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._client

    async def process(self, query: str) -> dict[str, Any]:
        """
        Decompose query into sub-queries.

        Args:
            query: Complex search query

        Returns:
            Dictionary with original and decomposed queries
        """
        sub_queries = await self.decompose(query)
        return {
            "original": query,
            "sub_queries": sub_queries,
            "is_complex": len(sub_queries) > 1,
        }

    async def decompose(self, query: str) -> list[str]:
        """
        Break down a complex query into simpler sub-queries.

        Args:
            query: Complex query

        Returns:
            List of simpler sub-queries
        """
        client = self._get_client()

        prompt = f"""Analyze this question and break it down into simpler sub-questions
that need to be answered to fully respond to the original question.

Original question: "{query}"

If the question is already simple and doesn't need decomposition, just return the original question.
Otherwise, return each sub-question on a new line.

Sub-questions:"""

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )

            content = response.choices[0].message.content.strip()
            sub_queries = [
                line.strip().lstrip("0123456789.-) ")
                for line in content.split("\n")
                if line.strip()
            ]

            # If only one sub-query that's the same as original, return original
            if len(sub_queries) == 1 and sub_queries[0].lower() == query.lower():
                return [query]

            logger.info(f"Decomposed query into {len(sub_queries)} sub-queries")
            return sub_queries if sub_queries else [query]

        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]


def create_query_processor(
    processor_type: str = "expander",
    **kwargs: Any,
) -> BaseQueryProcessor:
    """
    Factory function to create a query processor.

    Args:
        processor_type: Type of processor (expander, hyde, multi, decomposer)
        **kwargs: Additional arguments for the processor

    Returns:
        Query processor instance
    """
    processors = {
        "expander": LLMQueryExpander,
        "hyde": HyDEProcessor,
        "multi": MultiQueryProcessor,
        "decomposer": QueryDecomposer,
    }

    if processor_type not in processors:
        raise ValueError(
            f"Unknown processor type: {processor_type}. "
            f"Available: {list(processors.keys())}"
        )

    return processors[processor_type](**kwargs)

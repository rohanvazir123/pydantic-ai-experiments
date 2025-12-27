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


class BaseQueryProcessor(ABC):
    """Abstract base class for query processors."""

    @abstractmethod
    async def process(self, query: str) -> dict[str, Any]:
        """
        Process a query and return enhanced query information.

        Args:
            query: Original search query

        Returns:
            Dictionary containing processed query information
        """
        pass


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


class HyDEProcessor(BaseQueryProcessor):
    """
    Hypothetical Document Embeddings (HyDE) processor.

    Instead of embedding the query directly, HyDE generates a hypothetical
    document that would answer the query, then embeds that document.
    This often improves retrieval because the hypothetical document
    is more similar to actual documents than the query itself.

    Reference: https://arxiv.org/abs/2212.10496
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        embedding_model: str = "nomic-embed-text:latest",
        embedding_base_url: str = "http://localhost:11434/v1",
    ):
        """
        Initialize HyDE processor.

        Args:
            model: LLM model name for generation
            base_url: LLM API base URL
            api_key: LLM API key
            embedding_model: Embedding model name
            embedding_base_url: Embedding API base URL
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.embedding_base_url = embedding_base_url
        self._llm_client = None
        self._embed_client = None

    def _get_llm_client(self):
        """Get or create the LLM client."""
        if self._llm_client is None:
            from openai import AsyncOpenAI

            self._llm_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._llm_client

    def _get_embed_client(self):
        """Get or create the embedding client."""
        if self._embed_client is None:
            from openai import AsyncOpenAI

            self._embed_client = AsyncOpenAI(
                base_url=self.embedding_base_url,
                api_key=self.api_key,
            )
        return self._embed_client

    async def process(self, query: str) -> dict[str, Any]:
        """
        Generate hypothetical document and its embedding.

        Args:
            query: Original search query

        Returns:
            Dictionary with hypothetical document and its embedding
        """
        hypothetical = await self.generate_hypothetical(query)
        embedding = await self.embed(hypothetical)

        return {
            "original_query": query,
            "hypothetical_document": hypothetical,
            "hyde_embedding": embedding,
        }

    async def generate_hypothetical(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.

        Args:
            query: Search query

        Returns:
            Hypothetical document text
        """
        client = self._get_llm_client()

        prompt = f"""Write a short passage (2-3 paragraphs) that would be found in a document
that directly answers this question:

Question: {query}

Write as if you are quoting from an authoritative source. Be specific and factual.
Do not say "according to" or "the document says" - just write the content directly."""

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.5,
            )

            hypothetical = response.choices[0].message.content.strip()
            logger.info(f"Generated hypothetical document ({len(hypothetical)} chars)")
            return hypothetical

        except Exception as e:
            logger.warning(f"Hypothetical generation failed: {e}")
            return query  # Fallback to original query

    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        client = self._get_embed_client()

        try:
            response = await client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise


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

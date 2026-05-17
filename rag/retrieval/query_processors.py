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



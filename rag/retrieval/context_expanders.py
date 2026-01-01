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

"""Context expansion for enriching search results with surrounding content.

This module provides strategies to expand retrieved chunks with additional context:
- AdjacentChunkExpander: Fetches chunks before/after the matched chunk
- SectionExpander: Fetches the entire section containing the match
- DocumentSummaryExpander: Adds document-level summary to context
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from rag.ingestion.models import SearchResult

logger = logging.getLogger(__name__)


class BaseContextExpander(ABC):
    """Abstract base class for context expanders."""

    @abstractmethod
    async def expand(
        self,
        result: SearchResult,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Expand a search result with additional context.

        Args:
            result: Search result to expand
            **kwargs: Additional expansion parameters

        Returns:
            Dictionary with expanded context
        """
        pass


class AdjacentChunkExpander(BaseContextExpander):
    """
    Expands context by fetching adjacent chunks.

    When a chunk matches a query, the surrounding chunks often contain
    relevant context that helps the LLM understand and answer better.
    """

    def __init__(self, store: Any):
        """
        Initialize adjacent chunk expander.

        Args:
            store: MongoDB store instance with chunk retrieval methods
        """
        self.store = store

    async def expand(
        self,
        result: SearchResult,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Expand result with adjacent chunks.

        Args:
            result: Search result to expand
            **kwargs: Additional parameters (context_before, context_after)

        Returns:
            Dictionary with main result and surrounding context
        """
        context_before = kwargs.get("context_before", 1)
        context_after = kwargs.get("context_after", 1)

        # Get the matched chunk's details
        chunk = await self.store.get_chunk_by_id(result.chunk_id)
        if not chunk:
            return {
                "main": result,
                "context_before": [],
                "context_after": [],
                "combined_content": result.content,
            }

        chunk_index = chunk.get("chunk_index", 0)
        document_id = str(chunk.get("document_id", result.document_id))

        # Calculate index range
        start_index = max(0, chunk_index - context_before)
        end_index = chunk_index + context_after + 1

        # Fetch adjacent chunks
        adjacent_chunks = await self.store.get_chunks_by_document(
            document_id=document_id,
            start_index=start_index,
            end_index=end_index,
        )

        # Separate into before, current, and after
        before_chunks = []
        after_chunks = []

        for adj_chunk in adjacent_chunks:
            adj_index = adj_chunk.get("chunk_index", 0)
            if adj_index < chunk_index:
                before_chunks.append(adj_chunk)
            elif adj_index > chunk_index:
                after_chunks.append(adj_chunk)

        # Sort chunks by index
        before_chunks.sort(key=lambda x: x.get("chunk_index", 0))
        after_chunks.sort(key=lambda x: x.get("chunk_index", 0))

        # Combine content
        all_chunks = before_chunks + [chunk] + after_chunks
        combined_content = self._combine_chunks(all_chunks)

        logger.info(
            f"Expanded chunk {chunk_index} with {len(before_chunks)} before, "
            f"{len(after_chunks)} after"
        )

        return {
            "main": result,
            "context_before": before_chunks,
            "context_after": after_chunks,
            "combined_content": combined_content,
            "chunk_range": (start_index, end_index - 1),
        }

    def _combine_chunks(self, chunks: list[dict]) -> str:
        """Combine chunks into a single text block."""
        sorted_chunks = sorted(chunks, key=lambda x: x.get("chunk_index", 0))
        return "\n\n".join(c.get("content", "") for c in sorted_chunks)

    async def expand_multiple(
        self,
        results: list[SearchResult],
        context_before: int = 1,
        context_after: int = 1,
        deduplicate: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Expand multiple results with context.

        Args:
            results: List of search results
            context_before: Number of chunks before each result
            context_after: Number of chunks after each result
            deduplicate: Whether to remove overlapping context

        Returns:
            List of expanded results
        """
        expanded = []
        seen_chunks: set[str] = set()

        for result in results:
            expansion = await self.expand(
                result,
                context_before=context_before,
                context_after=context_after,
            )

            if deduplicate:
                # Filter out already-seen chunks from context
                expansion["context_before"] = [
                    c
                    for c in expansion["context_before"]
                    if str(c.get("_id", "")) not in seen_chunks
                ]
                expansion["context_after"] = [
                    c
                    for c in expansion["context_after"]
                    if str(c.get("_id", "")) not in seen_chunks
                ]

                # Track seen chunks
                seen_chunks.add(result.chunk_id)
                for c in expansion["context_before"] + expansion["context_after"]:
                    seen_chunks.add(str(c.get("_id", "")))

            expanded.append(expansion)

        return expanded


class SectionExpander(BaseContextExpander):
    """
    Expands context by fetching the entire section.

    Uses document structure (headings) to identify section boundaries
    and retrieves all chunks within the same section.
    """

    def __init__(self, store: Any):
        """
        Initialize section expander.

        Args:
            store: MongoDB store instance
        """
        self.store = store

    async def expand(
        self,
        result: SearchResult,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Expand result with entire section content.

        Args:
            result: Search result to expand
            **kwargs: Additional parameters (max_chunks)

        Returns:
            Dictionary with section context
        """
        max_chunks = kwargs.get("max_chunks", 10)

        chunk = await self.store.get_chunk_by_id(result.chunk_id)
        if not chunk:
            return {
                "main": result,
                "section_chunks": [],
                "section_content": result.content,
            }

        # Try to find section boundaries from metadata
        metadata = chunk.get("metadata", {})
        section_path = metadata.get("section_path", "")
        heading = metadata.get("heading", "")

        document_id = str(chunk.get("document_id", result.document_id))

        # Fetch chunks with same section path or heading
        section_chunks = await self._get_section_chunks(
            document_id=document_id,
            section_path=section_path,
            heading=heading,
            max_chunks=max_chunks,
        )

        section_content = self._combine_chunks(section_chunks)

        logger.info(
            f"Expanded to section '{heading or section_path}' "
            f"with {len(section_chunks)} chunks"
        )

        return {
            "main": result,
            "section_chunks": section_chunks,
            "section_content": section_content,
            "section_heading": heading,
            "section_path": section_path,
        }

    async def _get_section_chunks(
        self,
        document_id: str,
        section_path: str,
        heading: str,
        max_chunks: int,
    ) -> list[dict]:
        """Get all chunks in the same section."""
        # This requires a MongoDB query - implement based on your schema
        # For now, return empty list as this needs store method
        logger.warning("Section expansion requires get_section_chunks store method")
        return []

    def _combine_chunks(self, chunks: list[dict]) -> str:
        """Combine chunks into section content."""
        sorted_chunks = sorted(chunks, key=lambda x: x.get("chunk_index", 0))
        return "\n\n".join(c.get("content", "") for c in sorted_chunks)


class DocumentSummaryExpander(BaseContextExpander):
    """
    Adds document-level summary to provide broader context.

    Useful when the retrieved chunk is specific but understanding
    requires knowing the document's overall topic and purpose.
    """

    def __init__(
        self,
        store: Any,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ):
        """
        Initialize document summary expander.

        Args:
            store: MongoDB store instance
            model: LLM model for summarization
            base_url: LLM API base URL
            api_key: LLM API key
        """
        self.store = store
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._client = None
        self._summary_cache: dict[str, str] = {}

    def _get_client(self):
        """Get or create the LLM client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._client

    async def expand(
        self,
        result: SearchResult,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Expand result with document summary.

        Args:
            result: Search result to expand
            **kwargs: Additional parameters (include_summary)

        Returns:
            Dictionary with document context
        """
        document = await self.store.get_document_by_id(result.document_id)

        include_summary = kwargs.get("include_summary", True)
        document_summary = ""
        if include_summary and document:
            # Check cache first
            if result.document_id in self._summary_cache:
                document_summary = self._summary_cache[result.document_id]
            else:
                document_summary = await self._generate_summary(document)
                self._summary_cache[result.document_id] = document_summary

        return {
            "main": result,
            "document_title": result.document_title,
            "document_source": result.document_source,
            "document_summary": document_summary,
            "combined_context": self._format_context(result, document_summary),
        }

    async def _generate_summary(self, document: dict) -> str:
        """Generate a summary of the document."""
        content = document.get("content", "")
        if not content:
            return ""

        # Truncate for LLM context
        max_chars = 8000
        if len(content) > max_chars:
            content = content[:max_chars] + "..."

        client = self._get_client()

        prompt = f"""Summarize this document in 2-3 sentences, focusing on its main topic and purpose:

{content}

Summary:"""

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3,
            )

            summary = response.choices[0].message.content.strip()
            logger.info(f"Generated document summary ({len(summary)} chars)")
            return summary

        except Exception as e:
            logger.warning(f"Document summarization failed: {e}")
            return ""

    def _format_context(self, result: SearchResult, summary: str) -> str:
        """Format the expanded context for LLM consumption."""
        parts = []

        if summary:
            parts.append(f"## Document Overview\n{summary}\n")

        parts.append(f"## Relevant Section (from: {result.document_title})")
        parts.append(result.content)

        return "\n\n".join(parts)


class CompositeExpander(BaseContextExpander):
    """
    Combines multiple expansion strategies.

    Allows using adjacent chunks, section context, and document summary
    together for maximum context enrichment.
    """

    def __init__(
        self,
        store: Any,
        use_adjacent: bool = True,
        use_section: bool = False,
        use_summary: bool = True,
        adjacent_before: int = 1,
        adjacent_after: int = 1,
    ):
        """
        Initialize composite expander.

        Args:
            store: MongoDB store instance
            use_adjacent: Whether to include adjacent chunks
            use_section: Whether to include section context
            use_summary: Whether to include document summary
            adjacent_before: Chunks before for adjacent expansion
            adjacent_after: Chunks after for adjacent expansion
        """
        self.store = store
        self.use_adjacent = use_adjacent
        self.use_section = use_section
        self.use_summary = use_summary
        self.adjacent_before = adjacent_before
        self.adjacent_after = adjacent_after

        # Initialize sub-expanders
        if use_adjacent:
            self.adjacent_expander = AdjacentChunkExpander(store)
        if use_section:
            self.section_expander = SectionExpander(store)
        if use_summary:
            self.summary_expander = DocumentSummaryExpander(store)

    async def expand(
        self,
        result: SearchResult,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Expand result using multiple strategies.

        Args:
            result: Search result to expand
            **kwargs: Additional parameters

        Returns:
            Combined expansion from all strategies
        """
        expansion: dict[str, Any] = {
            "main": result,
        }

        # Adjacent chunk expansion
        if self.use_adjacent:
            adjacent = await self.adjacent_expander.expand(
                result,
                context_before=self.adjacent_before,
                context_after=self.adjacent_after,
            )
            expansion["adjacent"] = adjacent

        # Section expansion
        if self.use_section:
            section = await self.section_expander.expand(result)
            expansion["section"] = section

        # Summary expansion
        if self.use_summary:
            summary = await self.summary_expander.expand(result)
            expansion["summary"] = summary

        # Combine into final context
        expansion["combined_content"] = self._combine_expansions(expansion)

        return expansion

    def _combine_expansions(self, expansion: dict) -> str:
        """Combine all expansions into unified context."""
        parts = []

        # Document summary first (if available)
        if "summary" in expansion and expansion["summary"].get("document_summary"):
            parts.append(f"## Document: {expansion['main'].document_title}")
            parts.append(expansion["summary"]["document_summary"])
            parts.append("")

        # Main content with adjacent context
        if "adjacent" in expansion:
            parts.append("## Relevant Content")
            parts.append(expansion["adjacent"]["combined_content"])
        else:
            parts.append("## Relevant Content")
            parts.append(expansion["main"].content)

        return "\n\n".join(parts)


def create_context_expander(
    store: Any,
    expander_type: str = "adjacent",
    **kwargs: Any,
) -> BaseContextExpander:
    """
    Factory function to create a context expander.

    Args:
        store: MongoDB store instance
        expander_type: Type of expander (adjacent, section, summary, composite)
        **kwargs: Additional arguments

    Returns:
        Context expander instance
    """
    expanders = {
        "adjacent": AdjacentChunkExpander,
        "section": SectionExpander,
        "summary": DocumentSummaryExpander,
        "composite": CompositeExpander,
    }

    if expander_type not in expanders:
        raise ValueError(
            f"Unknown expander type: {expander_type}. "
            f"Available: {list(expanders.keys())}"
        )

    return expanders[expander_type](store, **kwargs)

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
Docling HybridChunker implementation for intelligent document splitting.

Module: rag.ingestion.chunkers.docling
======================================

This module provides intelligent document chunking using Docling's HybridChunker.
Combines token-aware splitting with document structure preservation.

Features
--------
- Token-aware chunking (uses HuggingFace tokenizer)
- Document structure preservation (headings, sections, tables)
- Semantic boundary respect (paragraphs, code blocks)
- Contextualized output (chunks include heading hierarchy)
- Fallback to simple chunking when DoclingDocument unavailable

Classes
-------
DoclingHybridChunker
    Wrapper around Docling's HybridChunker for document splitting.

    Methods:
        __init__(config: ChunkingConfig)
            Initialize with chunking configuration.

        async chunk_document(
            content: str,
            title: str,
            source: str,
            metadata: dict | None = None,
            docling_doc: DoclingDocument | None = None
        ) -> list[ChunkData]
            Chunk a document using HybridChunker or fallback.

        _simple_fallback_chunk(content: str, base_metadata: dict) -> list[ChunkData]
            Simple sliding window chunking (used when no DoclingDocument).

    Attributes:
        config: ChunkingConfig      - Chunking configuration
        chunker: HybridChunker      - Docling HybridChunker instance
        tokenizer: AutoTokenizer    - HuggingFace tokenizer for token counting

Functions
---------
create_chunker(config: ChunkingConfig) -> DoclingHybridChunker
    Factory function to create DoclingHybridChunker instance.

Constants
---------
TOKENIZER_MODEL: str
    HuggingFace tokenizer model (default: "sentence-transformers/all-MiniLM-L6-v2")

Usage
-----
    from rag.ingestion.chunkers.docling import create_chunker
    from rag.ingestion.models import ChunkingConfig

    # Create chunker with config
    config = ChunkingConfig(max_tokens=512)
    chunker = create_chunker(config)

    # Chunk document (with DoclingDocument for best results)
    chunks = await chunker.chunk_document(
        content="Document text...",
        title="My Document",
        source="doc.pdf",
        docling_doc=dl_doc  # From Docling converter
    )

    # Fallback chunking (without DoclingDocument)
    chunks = await chunker.chunk_document(
        content="Plain text...",
        title="Text File",
        source="file.txt"
    )
"""

import logging
from typing import Any

from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from transformers import AutoTokenizer

from rag.ingestion.models import ChunkData, ChunkingConfig

logger = logging.getLogger(__name__)

# Tokenizer model for chunking
TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DoclingHybridChunker:
    """Docling HybridChunker wrapper for intelligent document splitting.

    This chunker uses Docling's built-in HybridChunker which:
    - Respects document structure (sections, paragraphs, tables)
    - Is token-aware (fits embedding model limits)
    - Preserves semantic coherence
    - Includes heading context in chunks
    """

    def __init__(self, config: ChunkingConfig):
        """
        Initialize chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config

        # Create HybridChunker with tokenizer object
        # Note: max_tokens is a valid param per docs, but missing from type stubs
        logger.info(f"Initializing HybridChunker with tokenizer: {TOKENIZER_MODEL}")
        tokenizer_obj = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        self.chunker = HybridChunker(
            tokenizer=tokenizer_obj,
            max_tokens=config.max_tokens,  # type: ignore[call-arg]
            merge_peers=True,
        )

        # Store the tokenizer for token counting
        self.tokenizer = tokenizer_obj

        logger.info(f"HybridChunker initialized (max_tokens={config.max_tokens})")

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: dict[str, Any] | None = None,
        docling_doc: DoclingDocument | None = None,
    ) -> list[ChunkData]:
        """
        Chunk a document using Docling's HybridChunker.

        Args:
            content: Document content (markdown format)
            title: Document title
            source: Document source
            metadata: Additional metadata
            docling_doc: Optional pre-converted DoclingDocument (for efficiency)

        Returns:
            List of document chunks with contextualized content
        """
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "hybrid",
            **(metadata or {}),
        }

        # If we don't have a DoclingDocument, we need to create one from markdown
        if docling_doc is None:
            logger.warning(
                "No DoclingDocument provided, using simple chunking fallback"
            )
            return self._simple_fallback_chunk(content, base_metadata)

        try:
            # Use HybridChunker to chunk the DoclingDocument
            chunk_iter = self.chunker.chunk(dl_doc=docling_doc)
            chunks = list(chunk_iter)

            # Convert Docling chunks to ChunkData objects
            document_chunks = []
            current_pos = 0

            for i, chunk in enumerate(chunks):
                # Get contextualized text (includes heading hierarchy)
                contextualized_text = self.chunker.contextualize(chunk=chunk)

                # Count actual tokens
                token_count = len(self.tokenizer.encode(contextualized_text))

                # Create chunk metadata
                chunk_metadata = {
                    **base_metadata,
                    "total_chunks": len(chunks),
                    "token_count": token_count,
                    "has_context": True,  # Flag indicating contextualized chunk
                }

                # Estimate character positions
                start_char = current_pos
                end_char = start_char + len(contextualized_text)

                document_chunks.append(
                    ChunkData(
                        content=contextualized_text.strip(),
                        index=i,
                        start_char=start_char,
                        end_char=end_char,
                        metadata=chunk_metadata,
                        token_count=token_count,
                    )
                )

                current_pos = end_char

            logger.info(f"Created {len(document_chunks)} chunks using HybridChunker")
            return document_chunks

        except Exception as e:
            logger.error(f"HybridChunker failed: {e}, falling back to simple chunking")
            return self._simple_fallback_chunk(content, base_metadata)

    def _simple_fallback_chunk(
        self, content: str, base_metadata: dict[str, Any]
    ) -> list[ChunkData]:
        """
        Simple fallback chunking when HybridChunker can't be used.

        This is used when:
        - No DoclingDocument is provided
        - HybridChunker fails

        Args:
            content: Content to chunk
            base_metadata: Base metadata for chunks

        Returns:
            List of document chunks
        """
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        # Simple sliding window approach
        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + chunk_size

            if end >= len(content):
                # Last chunk
                chunk_text = content[start:]
            else:
                # Try to end at sentence boundary
                chunk_end = end
                for i in range(
                    end, max(start + self.config.min_chunk_size, end - 200), -1
                ):
                    if i < len(content) and content[i] in ".!?\n":
                        chunk_end = i + 1
                        break
                chunk_text = content[start:chunk_end]
                end = chunk_end

            if chunk_text.strip():
                token_count = len(self.tokenizer.encode(chunk_text))

                chunks.append(
                    ChunkData(
                        content=chunk_text.strip(),
                        index=chunk_index,
                        start_char=start,
                        end_char=end,
                        metadata={
                            **base_metadata,
                            "chunk_method": "simple_fallback",
                            "total_chunks": -1,  # Will update after
                        },
                        token_count=token_count,
                    )
                )

                chunk_index += 1

            # Move forward with overlap
            start = end - overlap

        # Update total chunks
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        logger.info(f"Created {len(chunks)} chunks using simple fallback")
        return chunks


def create_chunker(config: ChunkingConfig) -> DoclingHybridChunker:
    """
    Create DoclingHybridChunker for intelligent document splitting.

    Args:
        config: Chunking configuration

    Returns:
        DoclingHybridChunker instance
    """
    return DoclingHybridChunker(config)


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _logger = logging.getLogger(__name__)

    async def main():
        _logger.info("=" * 60)
        _logger.info("RAG Docling Chunker Module Test")
        _logger.info("=" * 60)

        # Create chunker with config
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=100,
            max_tokens=256,
        )
        chunker = create_chunker(config)

        _logger.info("[Chunker Created]")
        _logger.info(f"  Tokenizer: {TOKENIZER_MODEL}")
        _logger.info(f"  Max Tokens: {config.max_tokens}")
        _logger.info(f"  Chunk Size: {config.chunk_size}")
        _logger.info(f"  Chunk Overlap: {config.chunk_overlap}")

        # Test fallback chunking (no DoclingDocument)
        _logger.info("--- Fallback Chunking Test ---")
        sample_text = """
# Introduction to RAG

Retrieval-Augmented Generation (RAG) is a technique that combines retrieval
and generation for better AI responses.

## How RAG Works

RAG systems first retrieve relevant documents from a knowledge base,
then use those documents as context for generating responses.

### Key Components

1. **Document Store**: Where documents are indexed and stored
2. **Retriever**: Finds relevant documents for a query
3. **Generator**: Creates responses using retrieved context

## Benefits of RAG

- Reduces hallucinations
- Enables knowledge updates without retraining
- Provides source attribution
- Improves factual accuracy
"""

        chunks = await chunker.chunk_document(
            content=sample_text,
            title="RAG Introduction",
            source="test.md",
            metadata={"type": "markdown"},
        )

        _logger.info(f"  Input length: {len(sample_text)} chars")
        _logger.info(f"  Chunks created: {len(chunks)}")
        _logger.info(f"  Chunk method: {chunks[0].metadata.get('chunk_method', 'unknown')}")

        for i, chunk in enumerate(chunks):
            _logger.info(f"  Chunk {i}:")
            _logger.info(f"    Token count: {chunk.token_count}")
            _logger.info(f"    Char range: {chunk.start_char}-{chunk.end_char}")
            content_preview = chunk.content[:80].replace("\n", " ")
            _logger.info(f"    Content: {content_preview}...")

        # Test with very short content
        _logger.info("--- Short Content Test ---")
        short_chunks = await chunker.chunk_document(
            content="Just a short sentence.",
            title="Short Doc",
            source="short.txt",
        )
        _logger.info(f"  Short content chunks: {len(short_chunks)}")

        # Test with empty content
        _logger.info("--- Empty Content Test ---")
        empty_chunks = await chunker.chunk_document(
            content="   ",
            title="Empty Doc",
            source="empty.txt",
        )
        _logger.info(f"  Empty content chunks: {len(empty_chunks)}")

        _logger.info("=" * 60)
        _logger.info("Docling chunker test completed successfully!")
        _logger.info("=" * 60)

    asyncio.run(main())

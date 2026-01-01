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

"""Semantic chunking based on embedding similarity.

This chunker splits documents at semantic boundaries by detecting
low similarity between adjacent sentences. This creates more
coherent chunks that preserve meaning better than fixed-size chunking.
"""

import logging  # noqa: I001
from typing import Any

import numpy as np

from rag.ingestion.models import ChunkData, ChunkingConfig
from sentence_transformers import SentenceTransformer  # type: ignore

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Semantic chunker that splits at meaning boundaries.

    Uses sentence embeddings to detect semantic breaks between
    adjacent sentences. When similarity drops below a threshold,
    a new chunk is started.

    This approach creates chunks that are more semantically coherent
    than fixed-size chunking, improving retrieval quality.
    """

    def __init__(
        self,
        config: ChunkingConfig,
        similarity_threshold: float = 0.5,
        min_sentences: int = 2,
        max_sentences: int = 15,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize semantic chunker.

        Args:
            config: Chunking configuration
            similarity_threshold: Minimum similarity to stay in same chunk (0-1)
            min_sentences: Minimum sentences per chunk
            max_sentences: Maximum sentences per chunk
            embedding_model: Sentence transformer model for embeddings
        """
        self.config = config
        self.similarity_threshold = similarity_threshold
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.embedding_model = embedding_model
        self._model = None

    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                logger.info(f"Loading sentence transformer: {self.embedding_model}")
                self._model = SentenceTransformer(self.embedding_model)
            except ImportError as err:
                raise ImportError(
                    "sentence-transformers is required for SemanticChunker. "
                    "Install with: pip install sentence-transformers"
                ) from err
        return self._model

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkData]:
        """
        Chunk document using semantic boundaries.

        Args:
            content: Document content
            title: Document title
            source: Document source path
            metadata: Additional metadata

        Returns:
            List of semantically coherent chunks
        """
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "semantic",
            **(metadata or {}),
        }

        # Split into sentences
        sentences = self._split_sentences(content)

        if len(sentences) <= self.min_sentences:
            # Document too short for semantic chunking
            return [
                ChunkData(
                    content=content.strip(),
                    index=0,
                    start_char=0,
                    end_char=len(content),
                    metadata={**base_metadata, "total_chunks": 1},
                    token_count=self._count_tokens(content),
                )
            ]

        # Get embeddings for all sentences
        model = self._load_model()
        embeddings = model.encode(sentences, show_progress_bar=False)

        # Find semantic breaks
        chunks = self._find_semantic_chunks(sentences, embeddings)

        # Convert to ChunkData objects
        document_chunks = []
        current_pos = 0

        for i, chunk_sentences in enumerate(chunks):
            chunk_content = " ".join(chunk_sentences)
            token_count = self._count_tokens(chunk_content)

            # Find actual character positions
            start_char = content.find(chunk_sentences[0], current_pos)
            if start_char == -1:
                start_char = current_pos

            end_char = start_char + len(chunk_content)

            document_chunks.append(
                ChunkData(
                    content=chunk_content.strip(),
                    index=i,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        **base_metadata,
                        "total_chunks": len(chunks),
                        "sentence_count": len(chunk_sentences),
                    },
                    token_count=token_count,
                )
            )

            current_pos = end_char

        logger.info(
            f"Semantic chunking created {len(document_chunks)} chunks "
            f"from {len(sentences)} sentences"
        )
        return document_chunks

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Uses simple rules; could be enhanced with spaCy or nltk.
        """
        import re

        # Simple sentence splitting
        # Handle common abbreviations and edge cases
        text = re.sub(r"([.!?])\s+", r"\1\n", text)
        sentences = [s.strip() for s in text.split("\n") if s.strip()]

        # Filter out very short "sentences" (likely errors)
        sentences = [s for s in sentences if len(s) > 10]

        return sentences

    def _find_semantic_chunks(
        self,
        sentences: list[str],
        embeddings: np.ndarray,
    ) -> list[list[str]]:
        """
        Find semantic chunk boundaries based on embedding similarity.

        Args:
            sentences: List of sentences
            embeddings: Sentence embeddings

        Returns:
            List of chunks, each containing a list of sentences
        """
        chunks = []
        current_chunk: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = self._cosine_similarity(embeddings[i - 1], embeddings[i])

            # Decide whether to start new chunk
            should_split = (
                similarity < self.similarity_threshold
                and len(current_chunk) >= self.min_sentences
            )

            # Also split if chunk is too long
            if len(current_chunk) >= self.max_sentences:
                should_split = True

            if should_split:
                chunks.append(current_chunk)
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Simple approximation: ~4 characters per token
        return len(text) // 4


class GradientSemanticChunker:
    """
    Advanced semantic chunker using gradient of similarity.

    Instead of a fixed threshold, this chunker looks for significant
    drops in similarity (gradients) to find natural breakpoints.
    This is more robust to varying document styles.
    """

    def __init__(
        self,
        config: ChunkingConfig,
        percentile_threshold: int = 25,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize gradient semantic chunker.

        Args:
            config: Chunking configuration
            percentile_threshold: Percentile for similarity drop detection
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            embedding_model: Sentence transformer model
        """
        self.config = config
        self.percentile_threshold = percentile_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.embedding_model = embedding_model
        self._model = None

    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.embedding_model)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkData]:
        """
        Chunk document using gradient-based semantic boundaries.

        Args:
            content: Document content
            title: Document title
            source: Document source path
            metadata: Additional metadata

        Returns:
            List of semantically coherent chunks
        """
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "gradient_semantic",
            **(metadata or {}),
        }

        # Split into sentences
        sentences = self._split_sentences(content)

        if len(sentences) < 3:
            return [
                ChunkData(
                    content=content.strip(),
                    index=0,
                    start_char=0,
                    end_char=len(content),
                    metadata={**base_metadata, "total_chunks": 1},
                    token_count=len(content) // 4,
                )
            ]

        # Get embeddings
        model = self._load_model()
        embeddings = model.encode(sentences, show_progress_bar=False)

        # Calculate similarities between adjacent sentences
        similarities = []
        for i in range(len(sentences) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Find breakpoints using percentile threshold
        breakpoints = self._find_breakpoints(sentences, similarities)

        # Create chunks
        chunks = self._create_chunks_from_breakpoints(sentences, breakpoints)

        # Convert to ChunkData
        document_chunks = []
        current_pos = 0

        for i, chunk_text in enumerate(chunks):
            start_char = content.find(chunk_text[:50], current_pos)
            if start_char == -1:
                start_char = current_pos
            end_char = start_char + len(chunk_text)

            document_chunks.append(
                ChunkData(
                    content=chunk_text.strip(),
                    index=i,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={**base_metadata, "total_chunks": len(chunks)},
                    token_count=len(chunk_text) // 4,
                )
            )
            current_pos = end_char

        logger.info(f"Gradient semantic chunking created {len(document_chunks)} chunks")
        return document_chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re

        text = re.sub(r"([.!?])\s+", r"\1\n", text)
        sentences = [s.strip() for s in text.split("\n") if s.strip()]
        return [s for s in sentences if len(s) > 10]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _find_breakpoints(
        self,
        sentences: list[str],
        similarities: list[float],
    ) -> list[int]:
        """
        Find breakpoints using gradient analysis.

        Args:
            sentences: List of sentences
            similarities: Similarity scores between adjacent sentences

        Returns:
            List of sentence indices where chunks should break
        """
        if not similarities:
            return []

        # Calculate the threshold based on percentile
        threshold = np.percentile(similarities, self.percentile_threshold)

        breakpoints = []
        current_size = 0

        for i, sim in enumerate(similarities):
            sentence_len = len(sentences[i])
            current_size += sentence_len

            # Check if we should break here
            should_break = (
                sim < threshold and current_size >= self.min_chunk_size
            ) or current_size >= self.max_chunk_size

            if should_break:
                breakpoints.append(i + 1)  # Break after this sentence
                current_size = 0

        return breakpoints

    def _create_chunks_from_breakpoints(
        self,
        sentences: list[str],
        breakpoints: list[int],
    ) -> list[str]:
        """Create chunk texts from breakpoints."""
        chunks = []
        start = 0

        for bp in breakpoints:
            chunk_sentences = sentences[start:bp]
            if chunk_sentences:
                chunks.append(" ".join(chunk_sentences))
            start = bp

        # Add remaining sentences
        if start < len(sentences):
            chunks.append(" ".join(sentences[start:]))

        return chunks


def create_semantic_chunker(
    config: ChunkingConfig,
    chunker_type: str = "basic",
    **kwargs: Any,
) -> SemanticChunker | GradientSemanticChunker:
    """
    Factory function to create a semantic chunker.

    Args:
        config: Chunking configuration
        chunker_type: Type of chunker (basic, gradient)
        **kwargs: Additional arguments

    Returns:
        Semantic chunker instance
    """
    chunkers: dict[str, type[SemanticChunker | GradientSemanticChunker]] = {
        "basic": SemanticChunker,
        "gradient": GradientSemanticChunker,
    }

    if chunker_type not in chunkers:
        raise ValueError(
            f"Unknown chunker type: {chunker_type}. Available: {list(chunkers.keys())}"
        )

    return chunkers[chunker_type](config, **kwargs)

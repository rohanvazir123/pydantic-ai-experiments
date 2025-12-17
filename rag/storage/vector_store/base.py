"""Base protocols for vector store implementations."""

from typing import Protocol, List
from rag.ingestion.models import DocumentChunk, RetrievedChunk, SearchResult


class VectorStore(Protocol):
    """Protocol defining the vector store interface."""

    def add(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]]
    ) -> None:
        """
        Store document chunks with their embeddings.

        Args:
            chunks: List of document chunks
            embeddings: List of embedding vectors
        """
        ...

    def query(
        self,
        embedding: List[float],
        query_text: str,
        k: int,
    ) -> List[RetrievedChunk]:
        """
        Query the vector store for similar documents.

        Args:
            embedding: Query embedding vector
            query_text: Original query text for hybrid search
            k: Number of results to return

        Returns:
            List of retrieved chunks with scores
        """
        ...

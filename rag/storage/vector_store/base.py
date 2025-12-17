"""Base protocols for vector store implementations."""

from typing import Protocol

from rag.ingestion.models import DocumentChunk, RetrievedChunk


class VectorStore(Protocol):
    """Protocol defining the vector store interface."""

    def add(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """
        Store document chunks with their embeddings.

        Args:
            chunks: List of document chunks
            embeddings: List of embedding vectors
        """
        ...

    def query(
        self,
        embedding: list[float],
        query_text: str,
        k: int,
    ) -> list[RetrievedChunk]:
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

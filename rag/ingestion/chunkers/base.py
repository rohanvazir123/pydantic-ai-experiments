"""Base protocols for document chunkers."""

from typing import Any, Protocol

from rag.ingestion.models import ChunkData, DocumentChunk, IngestedDocument


class Chunker(Protocol):
    """Protocol defining the chunker interface."""

    def chunk(self, document: IngestedDocument) -> list[DocumentChunk]:
        """
        Chunk a document into smaller pieces.

        Args:
            document: Document to chunk

        Returns:
            List of document chunks
        """
        ...


class AsyncChunker(Protocol):
    """Protocol for async chunkers like DoclingHybridChunker."""

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: dict[str, Any] | None = None,
        docling_doc: Any | None = None,
    ) -> list[ChunkData]:
        """
        Chunk a document asynchronously.

        Args:
            content: Document content
            title: Document title
            source: Document source
            metadata: Additional metadata
            docling_doc: Optional Docling document for advanced chunking

        Returns:
            List of chunk data objects
        """
        ...

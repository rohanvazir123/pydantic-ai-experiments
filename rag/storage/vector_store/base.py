from typing import Protocol, List
from rag.ingestion.models import DocumentChunk


class RetrievedChunk(DocumentChunk):
    score: float


class VectorStore(Protocol):
    def add(
        self, chunks: List[DocumentChunk], embeddings: List[List[float]]
    ) -> None: ...

    def query(
        self,
        embedding: List[float],
        query_text: str,
        k: int,
    ) -> List[RetrievedChunk]: ...

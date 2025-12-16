from typing import Protocol, List
from rag.ingestion.models import IngestedDocument, DocumentChunk


class Chunker(Protocol):
    def chunk(self, document: IngestedDocument) -> List[DocumentChunk]: ...

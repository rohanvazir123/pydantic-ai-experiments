from rag.ingestion.chunkers.base import Chunker
from rag.ingestion.models import IngestedDocument, DocumentChunk


class DoclingChunker:
    def chunk(self, document: IngestedDocument):
        # Replace with real Docling logic
        return [
            DocumentChunk(
                id=f"{document.id}_0",
                text=document.text,
                metadata=document.metadata,
            )
        ]

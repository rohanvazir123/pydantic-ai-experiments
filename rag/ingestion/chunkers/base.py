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

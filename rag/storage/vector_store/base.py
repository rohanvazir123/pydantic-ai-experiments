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

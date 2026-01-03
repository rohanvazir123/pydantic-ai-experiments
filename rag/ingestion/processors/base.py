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
Base classes for multimodal content processors.

This module provides the foundation for processing different content types
(images, tables, equations) and extracting structured information for RAG.

Classes:
    ExtractedEntity: An entity extracted from content
    ExtractedRelationship: A relationship between entities
    ProcessedContent: Result of processing multimodal content
    BaseProcessor: Abstract base class for all processors
"""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Types of content that can be processed."""

    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    CHART = "chart"
    DIAGRAM = "diagram"
    GENERIC = "generic"


class ExtractedEntity(BaseModel):
    """An entity extracted from multimodal content."""

    name: str = Field(description="Name or identifier of the entity")
    entity_type: str = Field(
        description="Type of entity (e.g., Person, Organization, Concept, Object)"
    )
    description: str = Field(default="", description="Brief description of the entity")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Additional properties"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score"
    )
    source_content_type: ContentType = Field(
        default=ContentType.GENERIC,
        description="Type of content this was extracted from",
    )


class ExtractedRelationship(BaseModel):
    """A relationship between extracted entities."""

    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relationship_type: str = Field(
        description="Type of relationship in SCREAMING_SNAKE_CASE"
    )
    description: str = Field(default="", description="Description of the relationship")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Additional properties"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score"
    )


class ProcessedContent(BaseModel):
    """Result of processing multimodal content."""

    content_type: ContentType = Field(description="Type of content processed")
    source_path: str | None = Field(default=None, description="Path to the source file")
    description: str = Field(description="Natural language description of the content")
    summary: str = Field(default="", description="Brief summary for retrieval")
    entities: list[ExtractedEntity] = Field(
        default_factory=list, description="Entities extracted from content"
    )
    relationships: list[ExtractedRelationship] = Field(
        default_factory=list, description="Relationships between entities"
    )
    raw_text: str = Field(
        default="", description="Any text extracted directly from content"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    context_used: str = Field(
        default="", description="Surrounding context provided for processing"
    )

    def to_chunk_content(self) -> str:
        """Convert to text content suitable for chunking/embedding."""
        parts = []

        if self.description:
            parts.append(f"[{self.content_type.value.upper()}]\n{self.description}")

        if self.raw_text:
            parts.append(f"\nExtracted Text:\n{self.raw_text}")

        if self.entities:
            entity_strs = [
                f"- {e.name} ({e.entity_type}): {e.description}"
                for e in self.entities
                if e.description
            ]
            if entity_strs:
                parts.append("\nEntities:\n" + "\n".join(entity_strs))

        return "\n".join(parts)


class BaseProcessor(ABC):
    """
    Abstract base class for multimodal content processors.

    Processors analyze specific content types (images, tables, etc.)
    and extract structured information for the RAG pipeline.

    Subclasses must implement:
        - process(): Main processing method
        - content_type: Property returning the ContentType handled
    """

    @property
    @abstractmethod
    def content_type(self) -> ContentType:
        """Return the content type this processor handles."""
        ...

    @abstractmethod
    async def process(
        self,
        content: str | Path | bytes,
        context: str = "",
        **kwargs: Any,
    ) -> ProcessedContent:
        """
        Process content and extract structured information.

        Args:
            content: The content to process (path, URL, or raw bytes)
            context: Surrounding text context for better understanding
            **kwargs: Additional processor-specific arguments

        Returns:
            ProcessedContent with description, entities, and relationships
        """
        ...

    async def describe(
        self,
        content: str | Path | bytes,
        context: str = "",
        **kwargs: Any,
    ) -> str:
        """
        Generate a text description of the content.

        This is a lightweight alternative to full processing when only
        a description is needed (e.g., for batch processing stage 1).

        Args:
            content: The content to describe
            context: Surrounding text context
            **kwargs: Additional arguments

        Returns:
            Natural language description of the content
        """
        result = await self.process(content, context, **kwargs)
        return result.description

    def _validate_path(self, path: str | Path) -> Path:
        """Validate and return a Path object."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Content file not found: {path}")
        return path

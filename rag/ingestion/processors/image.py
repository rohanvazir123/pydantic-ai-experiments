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
Image processor for multimodal RAG.

This module provides vision model integration for processing images,
extracting descriptions, entities, and relationships for the knowledge graph.

Supports:
- OpenAI GPT-4V / GPT-4o
- Ollama with LLaVA or similar vision models
- Anthropic Claude with vision

Usage:
    from rag.ingestion.processors import ImageProcessor

    processor = ImageProcessor()
    result = await processor.process(
        "path/to/image.png",
        context="This image appears in a document about machine learning."
    )
    print(result.description)
    print(result.entities)
"""

import base64
import json
import logging
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from rag.config.settings import load_settings
from rag.ingestion.processors.base import (
    BaseProcessor,
    ContentType,
    ExtractedEntity,
    ExtractedRelationship,
    ProcessedContent,
)

logger = logging.getLogger(__name__)


# =============================================================================
# STRUCTURED OUTPUT MODELS
# =============================================================================


class ImageAnalysis(BaseModel):
    """Structured output from image analysis."""

    description: str = Field(description="Detailed description of what the image shows")
    summary: str = Field(
        description="One-sentence summary suitable for search/retrieval"
    )
    image_type: str = Field(
        description="Type of image (photo, diagram, chart, screenshot, illustration, etc.)"
    )
    entities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Entities found in the image with name, type, and description",
    )
    relationships: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Relationships between entities with source, target, and type",
    )
    text_content: str = Field(
        default="",
        description="Any text visible in the image (OCR)",
    )
    key_elements: list[str] = Field(
        default_factory=list,
        description="Key visual elements or objects in the image",
    )


# =============================================================================
# PROMPTS
# =============================================================================

IMAGE_ANALYSIS_SYSTEM_PROMPT = """You are an expert image analyst for a knowledge graph system.
Your task is to analyze images and extract structured information.

For each image, provide:
1. A detailed description of what the image shows
2. A brief one-sentence summary for search indexing
3. The type of image (photo, diagram, chart, screenshot, illustration, etc.)
4. Entities visible or represented (people, objects, concepts, organizations)
5. Relationships between entities
6. Any text visible in the image
7. Key visual elements

For entities, extract:
- name: The name or identifier
- type: Person, Object, Organization, Concept, Location, Product, etc.
- description: Brief description of the entity in this context

For relationships, extract:
- source: Source entity name
- target: Target entity name
- type: Relationship type in SCREAMING_SNAKE_CASE (e.g., CONTAINS, SHOWS, CONNECTED_TO)

Be thorough but factual. Only describe what you can actually see or reasonably infer."""

IMAGE_ANALYSIS_USER_PROMPT = """Analyze this image and extract structured information.

{context_section}

Provide your analysis as a JSON object with these fields:
- description: Detailed description
- summary: One-sentence summary
- image_type: Type of image
- entities: List of {{name, type, description}}
- relationships: List of {{source, target, type}}
- text_content: Any visible text
- key_elements: List of key visual elements"""


# =============================================================================
# IMAGE PROCESSOR
# =============================================================================


class ImageProcessor(BaseProcessor):
    """
    Process images using vision models for RAG.

    Extracts descriptions, entities, and relationships from images
    using GPT-4V, LLaVA, or other vision-capable models.

    Attributes:
        settings: Application settings
    """

    def __init__(self) -> None:
        """Initialize the image processor."""
        self.settings = load_settings()
        self._client: httpx.AsyncClient | None = None

    @property
    def content_type(self) -> ContentType:
        """Return the content type this processor handles."""
        return ContentType.IMAGE

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    def _encode_image(self, image_path: Path) -> tuple[str, str]:
        """
        Encode an image file to base64.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (base64_data, media_type)
        """
        suffix = image_path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        media_type = media_types.get(suffix, "image/png")

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        return image_data, media_type

    def _build_user_prompt(self, context: str) -> str:
        """Build the user prompt with optional context."""
        if context:
            context_section = f"Context from surrounding document:\n{context}\n"
        else:
            context_section = ""

        return IMAGE_ANALYSIS_USER_PROMPT.format(context_section=context_section)

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Try to extract JSON from markdown code block
        import re

        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            text = json_match.group(1)

        # Try to find JSON object
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Return empty dict if parsing fails
        return {}

    async def _call_vision_api(
        self,
        image_url: str,
        user_prompt: str,
        detail: str = "high",
    ) -> dict[str, Any]:
        """
        Call the vision API directly.

        Args:
            image_url: Base64 data URL for the image
            user_prompt: The user prompt
            detail: Detail level (low, high, auto)

        Returns:
            Parsed JSON response from the model
        """
        api_key = self.settings.vision_model_api_key or self.settings.llm_api_key
        base_url = (
            self.settings.vision_model_base_url or self.settings.llm_base_url or ""
        )

        # Ensure base_url ends properly
        if not base_url.endswith("/"):
            base_url += "/"

        url = f"{base_url}chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        payload = {
            "model": self.settings.vision_model,
            "messages": [
                {"role": "system", "content": IMAGE_ANALYSIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url, "detail": detail},
                        },
                    ],
                },
            ],
            "max_tokens": self.settings.vision_max_tokens,
        }

        client = await self._get_client()
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        return self._parse_json_response(content)

    async def process(
        self,
        content: str | Path | bytes,
        context: str = "",
        **kwargs: Any,
    ) -> ProcessedContent:
        """
        Process an image and extract structured information.

        Args:
            content: Path to image file, URL, or raw bytes
            context: Surrounding text context for better understanding
            **kwargs: Additional arguments (detail_level, etc.)

        Returns:
            ProcessedContent with description, entities, and relationships
        """
        # Handle different input types
        if isinstance(content, (str, Path)):
            image_path = self._validate_path(content)
            image_data, media_type = self._encode_image(image_path)
            source_path = str(image_path)
        elif isinstance(content, bytes):
            image_data = base64.b64encode(content).decode("utf-8")
            media_type = "image/png"  # Default for raw bytes
            source_path = None
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

        # Build the image URL for the API
        detail = kwargs.get("detail_level", self.settings.image_description_detail)
        image_url = f"data:{media_type};base64,{image_data}"

        # Create message with image
        user_prompt = self._build_user_prompt(context)

        try:
            # Call vision API directly
            analysis = await self._call_vision_api(image_url, user_prompt, detail)

            # Extract fields with defaults
            description = analysis.get("description", "No description available")
            summary = analysis.get("summary", "")
            image_type = analysis.get("image_type", "unknown")
            text_content = analysis.get("text_content", "")
            key_elements = analysis.get("key_elements", [])

            # Convert entities
            entities = [
                ExtractedEntity(
                    name=e.get("name", "Unknown"),
                    entity_type=e.get("type", "Object"),
                    description=e.get("description", ""),
                    source_content_type=ContentType.IMAGE,
                )
                for e in analysis.get("entities", [])
                if isinstance(e, dict)
            ]

            # Convert relationships
            relationships = [
                ExtractedRelationship(
                    source=r.get("source", ""),
                    target=r.get("target", ""),
                    relationship_type=r.get("type", "RELATED_TO"),
                )
                for r in analysis.get("relationships", [])
                if isinstance(r, dict) and r.get("source") and r.get("target")
            ]

            return ProcessedContent(
                content_type=ContentType.IMAGE,
                source_path=source_path,
                description=description,
                summary=summary,
                entities=entities,
                relationships=relationships,
                raw_text=text_content,
                metadata={
                    "image_type": image_type,
                    "key_elements": key_elements,
                    "detail_level": detail,
                },
                context_used=context,
            )

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            # Return a minimal result on error
            return ProcessedContent(
                content_type=ContentType.IMAGE,
                source_path=source_path,
                description=f"[Error processing image: {e}]",
                summary="Image processing failed",
                metadata={"error": str(e)},
                context_used=context,
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def describe(
        self,
        content: str | Path | bytes,
        context: str = "",
        **kwargs: Any,
    ) -> str:
        """
        Generate a text description of the image.

        This is a lightweight method that returns just the description
        without full entity/relationship extraction.

        Args:
            content: Path to image file or raw bytes
            context: Surrounding text context
            **kwargs: Additional arguments

        Returns:
            Natural language description of the image
        """
        # For now, use full processing and return description
        # Could be optimized with a simpler prompt if needed
        result = await self.process(content, context, **kwargs)
        return result.description

    async def batch_process(
        self,
        images: list[str | Path],
        contexts: list[str] | None = None,
        **kwargs: Any,
    ) -> list[ProcessedContent]:
        """
        Process multiple images.

        Args:
            images: List of image paths
            contexts: Optional list of contexts (one per image)
            **kwargs: Additional arguments

        Returns:
            List of ProcessedContent results
        """
        if contexts is None:
            contexts = [""] * len(images)
        elif len(contexts) != len(images):
            raise ValueError("contexts must have same length as images")

        results = []
        for image, context in zip(images, contexts):
            try:
                result = await self.process(image, context, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image}: {e}")
                results.append(
                    ProcessedContent(
                        content_type=ContentType.IMAGE,
                        source_path=str(image),
                        description=f"[Error: {e}]",
                        summary="Processing failed",
                        metadata={"error": str(e)},
                    )
                )

        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def process_image(
    image_path: str | Path,
    context: str = "",
    **kwargs: Any,
) -> ProcessedContent:
    """
    Convenience function to process a single image.

    Args:
        image_path: Path to the image file
        context: Optional surrounding context
        **kwargs: Additional arguments

    Returns:
        ProcessedContent with extracted information
    """
    processor = ImageProcessor()
    return await processor.process(image_path, context, **kwargs)


async def describe_image(
    image_path: str | Path,
    context: str = "",
    **kwargs: Any,
) -> str:
    """
    Convenience function to get an image description.

    Args:
        image_path: Path to the image file
        context: Optional surrounding context
        **kwargs: Additional arguments

    Returns:
        Text description of the image
    """
    processor = ImageProcessor()
    return await processor.describe(image_path, context, **kwargs)


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def main():
        """Test the image processor."""
        if len(sys.argv) < 2:
            print("Usage: python -m rag.ingestion.processors.image <image_path>")
            print("Example: python -m rag.ingestion.processors.image test.png")
            sys.exit(1)

        image_path = sys.argv[1]
        context = sys.argv[2] if len(sys.argv) > 2 else ""

        print(f"Processing image: {image_path}")
        if context:
            print(f"Context: {context}")

        processor = ImageProcessor()
        result = await processor.process(image_path, context)

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"\nDescription:\n{result.description}")
        print(f"\nSummary: {result.summary}")
        print(f"\nImage Type: {result.metadata.get('image_type', 'unknown')}")

        if result.entities:
            print(f"\nEntities ({len(result.entities)}):")
            for e in result.entities:
                print(f"  - {e.name} ({e.entity_type}): {e.description}")

        if result.relationships:
            print(f"\nRelationships ({len(result.relationships)}):")
            for r in result.relationships:
                print(f"  - {r.source} --[{r.relationship_type}]--> {r.target}")

        if result.raw_text:
            print(f"\nExtracted Text:\n{result.raw_text}")

        print(f"\nKey Elements: {result.metadata.get('key_elements', [])}")

        print("\n" + "=" * 60)
        print("Chunk Content (for embedding):")
        print("=" * 60)
        print(result.to_chunk_content())

    asyncio.run(main())

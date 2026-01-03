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
MinerU document parser for multimodal document extraction.

Module: rag.ingestion.chunkers.mineru
=====================================

This module provides document parsing using MinerU 2.5 vision-language model.
Extracts text, tables, figures, and layout information from PDFs and images.

Features:
- PDF and image processing
- Layout-aware extraction (headers, titles, text, tables, figures)
- Automatic figure/diagram description using VLM
- GPU-accelerated processing
- Bounding box extraction for visual elements

Requirements:
- CUDA-capable GPU
- MinerU model: opendatalab/MinerU2.5-2509-1.2B
- Dependencies: mineru_vl_utils, transformers, pypdfium2

Classes:
    ExtractedBlock: A single extracted content block
    MinerUParser: Main parser class for document extraction
    MinerUContext: Context holding model references

Usage:
    from rag.ingestion.chunkers.mineru import MinerUParser

    parser = MinerUParser()
    await parser.initialize()

    # Process a PDF
    blocks = await parser.parse_file("document.pdf")
    for block in blocks:
        print(f"{block.block_type}: {block.content[:100]}")

    await parser.close()
"""

import asyncio
import base64
import io
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BlockType(str, Enum):
    """Types of content blocks extracted by MinerU."""

    HEADER = "header"
    TITLE = "title"
    TEXT = "text"
    LIST = "list"
    TABLE = "table"
    TABLE_CAPTION = "table_caption"
    FIGURE = "figure"
    FIGURE_CAPTION = "figure_caption"
    IMAGE = "image"
    EQUATION = "equation"
    UNKNOWN = "unknown"


class ExtractedBlock(BaseModel):
    """A content block extracted from a document."""

    block_type: BlockType = Field(description="Type of content block")
    content: str = Field(description="Text content or HTML for tables")
    bbox: list[float] | None = Field(
        default=None,
        description="Normalized bounding box [x1, y1, x2, y2] in range 0-1",
    )
    page_number: int = Field(default=1, description="Page number (1-indexed)")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Extraction confidence"
    )
    figure_description: str | None = Field(
        default=None, description="VLM-generated description for figures"
    )
    figure_image_base64: str | None = Field(
        default=None, description="Base64-encoded cropped figure image"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def to_text(self) -> str:
        """Convert block to plain text for chunking."""
        if self.block_type == BlockType.FIGURE and self.figure_description:
            return f"[Figure: {self.figure_description}]"
        elif self.block_type == BlockType.TABLE:
            # Strip HTML tags for plain text
            import re

            return re.sub(r"<[^>]+>", " ", self.content).strip()
        return self.content


class ParsedDocument(BaseModel):
    """Result of parsing a document with MinerU."""

    source_path: str = Field(description="Path to source file")
    total_pages: int = Field(default=1, description="Total number of pages")
    blocks: list[ExtractedBlock] = Field(
        default_factory=list, description="Extracted content blocks"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )

    def to_text(self) -> str:
        """Convert entire document to plain text."""
        return "\n\n".join(block.to_text() for block in self.blocks if block.content)

    def get_figures(self) -> list[ExtractedBlock]:
        """Get all figure blocks."""
        return [
            b
            for b in self.blocks
            if b.block_type in (BlockType.FIGURE, BlockType.IMAGE)
        ]

    def get_tables(self) -> list[ExtractedBlock]:
        """Get all table blocks."""
        return [b for b in self.blocks if b.block_type == BlockType.TABLE]


@dataclass
class MinerUContext:
    """Context holding MinerU model references."""

    client: Any = None
    model: Any = None
    processor: Any = None
    describe_figures: bool = True
    initialized: bool = False


class MinerUParser:
    """
    Document parser using MinerU 2.5 vision-language model.

    Extracts text, tables, figures, and layout information from
    PDFs and images with GPU acceleration.

    Attributes:
        ctx: MinerUContext holding model references
        dpi: DPI for PDF rendering (default: 200)
    """

    def __init__(self, describe_figures: bool = True, dpi: int = 200):
        """
        Initialize the MinerU parser.

        Args:
            describe_figures: Whether to generate VLM descriptions for figures
            dpi: DPI for rendering PDF pages to images
        """
        self.ctx = MinerUContext(describe_figures=describe_figures)
        self.dpi = dpi
        self._lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """
        Initialize MinerU models (lazy loading).

        Returns:
            True if initialization successful, False otherwise
        """
        async with self._lock:
            if self.ctx.initialized:
                return True

            try:
                # Run initialization in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(None, self._initialize_sync)
                return success
            except Exception as e:
                logger.error(f"Failed to initialize MinerU: {e}")
                return False

    def _initialize_sync(self) -> bool:
        """Synchronous initialization of MinerU models."""
        import torch

        # Check GPU
        if not torch.cuda.is_available():
            logger.error("CUDA GPU required for MinerU")
            return False

        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

        try:
            from mineru_vl_utils import MinerUClient
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            logger.info("Loading MinerU model...")
            self.ctx.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "opendatalab/MinerU2.5-2509-1.2B",
                torch_dtype="auto",
                device_map="auto",
            )

            logger.info("Loading processor...")
            self.ctx.processor = AutoProcessor.from_pretrained(
                "opendatalab/MinerU2.5-2509-1.2B",
                use_fast=True,
            )

            logger.info("Initializing MinerU client...")
            self.ctx.client = MinerUClient(
                backend="transformers",
                model=self.ctx.model,
                processor=self.ctx.processor,
            )

            self.ctx.initialized = True
            logger.info("MinerU initialized successfully")
            return True

        except ImportError as e:
            logger.error(f"Missing MinerU dependencies: {e}")
            logger.error("Install with: pip install mineru-vl-utils transformers")
            return False
        except Exception as e:
            logger.error(f"Error initializing MinerU: {e}")
            return False

    async def close(self) -> None:
        """Release MinerU resources."""
        if self.ctx.model is not None:
            import torch

            del self.ctx.model
            del self.ctx.processor
            del self.ctx.client
            torch.cuda.empty_cache()
            self.ctx.initialized = False
            logger.info("MinerU resources released")

    def _pdf_to_images(self, pdf_path: str) -> list[Image.Image]:
        """Convert PDF pages to PIL Images."""
        import pypdfium2 as pdfium

        images = []
        try:
            pdf = pdfium.PdfDocument(pdf_path)
            n_pages = len(pdf)
            logger.info(f"PDF has {n_pages} pages")

            for i in range(n_pages):
                page = pdf[i]
                scale = self.dpi / 72
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil()
                images.append(pil_image)

            pdf.close()
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")

        return images

    def _crop_figure(
        self, image: Image.Image, bbox: list[float], padding: int = 10
    ) -> Image.Image:
        """Crop a figure from an image based on normalized bbox."""
        width, height = image.size
        x1, y1, x2, y2 = bbox

        left = max(0, int(x1 * width) - padding)
        top = max(0, int(y1 * height) - padding)
        right = min(width, int(x2 * width) + padding)
        bottom = min(height, int(y2 * height) + padding)

        return image.crop((left, top, right, bottom))

    def _describe_figure(self, image: Image.Image) -> str:
        """Generate VLM description for a figure."""
        if not self.ctx.describe_figures:
            return ""

        try:
            from qwen_vl_utils import process_vision_info

            prompt = (
                "Describe this diagram or figure in detail. "
                "Include any text, labels, arrows, and relationships between elements."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.ctx.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.ctx.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.ctx.model.device)

            generated_ids = self.ctx.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.ctx.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            return output_text[0] if output_text else ""
        except Exception as e:
            logger.error(f"Error describing figure: {e}")
            return ""

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _extract_page(
        self, image: Image.Image, page_number: int
    ) -> list[ExtractedBlock]:
        """Extract content blocks from a single page image."""
        blocks = []

        try:
            raw_blocks = self.ctx.client.two_step_extract(image)
            logger.info(f"Extracted {len(raw_blocks)} blocks from page {page_number}")

            for raw_block in raw_blocks:
                block_type_str = raw_block.get("type", "unknown")
                try:
                    block_type = BlockType(block_type_str)
                except ValueError:
                    block_type = BlockType.UNKNOWN

                content = raw_block.get("content", "")
                bbox = raw_block.get("bbox")

                # Handle figures - describe and crop
                figure_description = None
                figure_base64 = None

                if block_type in (BlockType.FIGURE, BlockType.IMAGE) and bbox:
                    cropped = self._crop_figure(image, bbox)
                    figure_base64 = self._image_to_base64(cropped)

                    if self.ctx.describe_figures:
                        figure_description = self._describe_figure(cropped)
                        logger.info(
                            f"Described figure on page {page_number}: "
                            f"{figure_description[:50]}..."
                        )

                block = ExtractedBlock(
                    block_type=block_type,
                    content=str(content) if content else "",
                    bbox=bbox,
                    page_number=page_number,
                    figure_description=figure_description,
                    figure_image_base64=figure_base64,
                )
                blocks.append(block)

        except Exception as e:
            logger.error(f"Error extracting page {page_number}: {e}")

        return blocks

    async def parse_file(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse a document file (PDF or image).

        Args:
            file_path: Path to the file

        Returns:
            ParsedDocument with extracted blocks
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Ensure initialized
        if not await self.initialize():
            raise RuntimeError("Failed to initialize MinerU")

        ext = file_path.suffix.lower()
        loop = asyncio.get_event_loop()

        if ext == ".pdf":
            return await loop.run_in_executor(
                None, self._parse_pdf_sync, str(file_path)
            )
        elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"):
            return await loop.run_in_executor(
                None, self._parse_image_sync, str(file_path)
            )
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _parse_pdf_sync(self, pdf_path: str) -> ParsedDocument:
        """Synchronously parse a PDF file."""
        images = self._pdf_to_images(pdf_path)
        if not images:
            return ParsedDocument(source_path=pdf_path, total_pages=0)

        all_blocks = []
        for i, img in enumerate(images):
            page_blocks = self._extract_page(img, page_number=i + 1)
            all_blocks.extend(page_blocks)

        return ParsedDocument(
            source_path=pdf_path,
            total_pages=len(images),
            blocks=all_blocks,
            metadata={"parser": "mineru", "dpi": self.dpi},
        )

    def _parse_image_sync(self, image_path: str) -> ParsedDocument:
        """Synchronously parse an image file."""
        image = Image.open(image_path)
        blocks = self._extract_page(image, page_number=1)

        return ParsedDocument(
            source_path=image_path,
            total_pages=1,
            blocks=blocks,
            metadata={"parser": "mineru"},
        )

    async def parse_image(self, image: Image.Image) -> list[ExtractedBlock]:
        """
        Parse a PIL Image directly.

        Args:
            image: PIL Image to parse

        Returns:
            List of extracted blocks
        """
        if not await self.initialize():
            raise RuntimeError("Failed to initialize MinerU")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._extract_page, image, 1)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def parse_document(file_path: str | Path) -> ParsedDocument:
    """
    Convenience function to parse a document with MinerU.

    Args:
        file_path: Path to PDF or image file

    Returns:
        ParsedDocument with extracted content
    """
    parser = MinerUParser()
    try:
        return await parser.parse_file(file_path)
    finally:
        await parser.close()


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def main():
        if len(sys.argv) < 2:
            print("Usage: python -m rag.ingestion.chunkers.mineru <file_path>")
            print("Example: python -m rag.ingestion.chunkers.mineru document.pdf")
            sys.exit(1)

        file_path = sys.argv[1]
        print(f"Parsing: {file_path}")

        result = await parse_document(file_path)

        print("\n" + "=" * 60)
        print(f"RESULTS: {result.total_pages} pages, {len(result.blocks)} blocks")
        print("=" * 60)

        # Summary by block type
        type_counts: dict[str, int] = {}
        for block in result.blocks:
            type_counts[block.block_type.value] = (
                type_counts.get(block.block_type.value, 0) + 1
            )
        print(f"\nBlock types: {type_counts}")

        # Show first few blocks
        print("\nFirst 5 blocks:")
        for block in result.blocks[:5]:
            content_preview = block.content[:100].replace("\n", " ")
            print(f"  [{block.block_type.value}] {content_preview}...")
            if block.figure_description:
                print(f"    -> Figure: {block.figure_description[:80]}...")

        # Show figures
        figures = result.get_figures()
        if figures:
            print(f"\nFigures ({len(figures)}):")
            for fig in figures:
                if fig.figure_description:
                    print(
                        f"  Page {fig.page_number}: {fig.figure_description[:100]}..."
                    )

        # Show full text preview
        print("\n" + "=" * 60)
        print("FULL TEXT (first 500 chars):")
        print("=" * 60)
        print(result.to_text()[:500])

    asyncio.run(main())

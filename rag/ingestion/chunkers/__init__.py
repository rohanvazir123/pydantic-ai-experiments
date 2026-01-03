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
Document chunking and parsing modules.

Available chunkers:
- DoclingHybridChunker: Intelligent document chunking with Docling
- MinerUParser: Multimodal document extraction with MinerU VLM
"""

# Docling chunker (always available)
from rag.ingestion.chunkers.docling import DoclingHybridChunker, create_chunker

# MinerU parser (requires GPU and additional dependencies)
try:
    from rag.ingestion.chunkers.mineru import (
        BlockType as BlockType,
    )
    from rag.ingestion.chunkers.mineru import (
        ExtractedBlock as ExtractedBlock,
    )
    from rag.ingestion.chunkers.mineru import (
        MinerUParser as MinerUParser,
    )
    from rag.ingestion.chunkers.mineru import (
        ParsedDocument as ParsedDocument,
    )
    from rag.ingestion.chunkers.mineru import (
        parse_document as parse_document,
    )

    MINERU_AVAILABLE = True
except ImportError:
    MINERU_AVAILABLE = False
    # Define placeholders for type hints
    BlockType = None  # type: ignore[misc,assignment]
    ExtractedBlock = None  # type: ignore[misc,assignment]
    MinerUParser = None  # type: ignore[misc,assignment]
    ParsedDocument = None  # type: ignore[misc,assignment]
    parse_document = None  # type: ignore[misc,assignment]

__all__ = [
    "DoclingHybridChunker",
    "create_chunker",
    "MINERU_AVAILABLE",
    "MinerUParser",
    "ParsedDocument",
    "ExtractedBlock",
    "BlockType",
    "parse_document",
]

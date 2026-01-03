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
Multimodal content processors for RAG ingestion.

This module provides processors for handling different content types:
- Images (using vision models)
- Tables (structured data extraction)
- Equations (mathematical content)
- Generic (fallback for other content)

Usage:
    from rag.ingestion.processors import (
        ImageProcessor,
        TableProcessor,
        EquationProcessor,
        ProcessedContent,
    )

    # Process an image
    image_processor = ImageProcessor()
    result = await image_processor.process(image_path, context="surrounding text")
    print(result.description)
    print(result.entities)

    # Process a table
    table_processor = TableProcessor()
    result = await table_processor.process(table_markdown, context="sales report")
    print(result.description)
    print(result.metadata["key_insights"])

    # Process an equation
    equation_processor = EquationProcessor()
    result = await equation_processor.process("E = mc^2", context="special relativity")
    print(result.description)
    print(result.metadata["variables"])
"""

from rag.ingestion.processors.base import BaseProcessor, ProcessedContent
from rag.ingestion.processors.equation import EquationProcessor
from rag.ingestion.processors.image import ImageProcessor
from rag.ingestion.processors.table import TableProcessor

__all__ = [
    "BaseProcessor",
    "ProcessedContent",
    "ImageProcessor",
    "TableProcessor",
    "EquationProcessor",
]

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
Table processor for multimodal RAG.

This module provides LLM integration for processing tables,
extracting descriptions, entities, relationships, and structured data
for the knowledge graph.

Supports:
- Markdown tables
- HTML tables
- CSV/TSV data
- Raw tabular text

Usage:
    from rag.ingestion.processors import TableProcessor

    processor = TableProcessor()
    result = await processor.process(
        table_content,
        context="This table shows quarterly sales data."
    )
    print(result.description)
    print(result.entities)
"""

import json
import logging
import re
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


class ColumnInfo(BaseModel):
    """Information about a table column."""

    name: str = Field(description="Column name/header")
    data_type: str = Field(
        description="Data type (numeric, text, date, percentage, currency, etc.)"
    )
    description: str = Field(default="", description="What this column represents")


class TableAnalysis(BaseModel):
    """Structured output from table analysis."""

    description: str = Field(
        description="Detailed description of what the table contains and shows"
    )
    summary: str = Field(
        description="One-sentence summary suitable for search/retrieval"
    )
    table_type: str = Field(
        description="Type of table (data, comparison, summary, schedule, etc.)"
    )
    columns: list[ColumnInfo] = Field(
        default_factory=list, description="Information about each column"
    )
    row_count: int = Field(default=0, description="Number of data rows")
    entities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Entities found in the table with name, type, and description",
    )
    relationships: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Relationships between entities with source, target, and type",
    )
    key_insights: list[str] = Field(
        default_factory=list,
        description="Key insights or patterns found in the data",
    )
    statistics: dict[str, Any] = Field(
        default_factory=dict,
        description="Any notable statistics (min, max, avg, trends, etc.)",
    )


# =============================================================================
# PROMPTS
# =============================================================================

TABLE_ANALYSIS_SYSTEM_PROMPT = """You are an expert data analyst for a knowledge graph system.
Your task is to analyze tables and extract structured information.

For each table, provide:
1. A detailed description of what the table contains and represents
2. A brief one-sentence summary for search indexing
3. The type of table (data, comparison, summary, schedule, lookup, etc.)
4. Information about each column (name, data type, meaning)
5. Entities mentioned or represented in the table
6. Relationships between entities
7. Key insights or patterns in the data
8. Notable statistics if applicable

For entities, extract:
- name: The name or identifier
- type: Person, Organization, Product, Metric, Category, Location, etc.
- description: Brief description of the entity in this context

For relationships, extract:
- source: Source entity name
- target: Target entity name
- type: Relationship type in SCREAMING_SNAKE_CASE (e.g., COMPARED_TO, MEASURED_BY, BELONGS_TO)

Be thorough but factual. Focus on extracting meaningful information for later retrieval."""

TABLE_ANALYSIS_USER_PROMPT = """Analyze this table and extract structured information.

{context_section}

Table Content:
{table_content}

Provide your analysis as a JSON object with these fields:
- description: Detailed description of the table
- summary: One-sentence summary
- table_type: Type of table
- columns: List of {{name, data_type, description}}
- row_count: Number of data rows
- entities: List of {{name, type, description}}
- relationships: List of {{source, target, type}}
- key_insights: List of key insights
- statistics: Object with notable statistics"""


# =============================================================================
# TABLE PROCESSOR
# =============================================================================


class TableProcessor(BaseProcessor):
    """
    Process tables using LLM for RAG.

    Extracts descriptions, entities, relationships, and structured data
    from tables using language models.

    Attributes:
        settings: Application settings
    """

    def __init__(self) -> None:
        """Initialize the table processor."""
        self.settings = load_settings()
        self._client: httpx.AsyncClient | None = None

    @property
    def content_type(self) -> ContentType:
        """Return the content type this processor handles."""
        return ContentType.TABLE

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    def _normalize_table(self, content: str) -> str:
        """
        Normalize table content to a consistent format.

        Handles markdown, HTML, CSV, and plain text tables.

        Args:
            content: Raw table content

        Returns:
            Normalized table string
        """
        content = content.strip()

        # Already markdown table
        if "|" in content and "-" in content:
            return content

        # HTML table - convert to markdown
        if "<table" in content.lower():
            return self._html_to_markdown(content)

        # CSV/TSV - convert to markdown
        if "," in content or "\t" in content:
            return self._csv_to_markdown(content)

        # Return as-is for other formats
        return content

    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML table to markdown format."""
        # Simple HTML table conversion
        # Remove tags and extract cell contents
        html = re.sub(r"</?table[^>]*>", "", html, flags=re.IGNORECASE)
        html = re.sub(r"</?thead[^>]*>", "", html, flags=re.IGNORECASE)
        html = re.sub(r"</?tbody[^>]*>", "", html, flags=re.IGNORECASE)

        rows = []
        for row_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", html, re.IGNORECASE | re.DOTALL):
            row_content = row_match.group(1)
            cells = []
            for cell_match in re.finditer(
                r"<t[hd][^>]*>(.*?)</t[hd]>", row_content, re.IGNORECASE | re.DOTALL
            ):
                cell_text = re.sub(r"<[^>]+>", "", cell_match.group(1)).strip()
                cells.append(cell_text)
            if cells:
                rows.append(cells)

        if not rows:
            return html  # Return original if parsing failed

        # Convert to markdown
        result = []
        for i, row in enumerate(rows):
            result.append("| " + " | ".join(row) + " |")
            if i == 0:
                result.append("|" + "|".join(["---"] * len(row)) + "|")

        return "\n".join(result)

    def _csv_to_markdown(self, csv_content: str) -> str:
        """Convert CSV/TSV content to markdown format."""
        import csv
        import io

        # Detect delimiter
        delimiter = "\t" if "\t" in csv_content else ","

        rows = []
        reader = csv.reader(io.StringIO(csv_content), delimiter=delimiter)
        for row in reader:
            if row:  # Skip empty rows
                rows.append(row)

        if not rows:
            return csv_content

        # Convert to markdown
        result = []
        max_cols = max(len(row) for row in rows)

        for i, row in enumerate(rows):
            # Pad row to max columns
            padded = row + [""] * (max_cols - len(row))
            result.append("| " + " | ".join(padded) + " |")
            if i == 0:
                result.append("|" + "|".join(["---"] * max_cols) + "|")

        return "\n".join(result)

    def _build_user_prompt(self, table_content: str, context: str) -> str:
        """Build the user prompt with table content and context."""
        if context:
            context_section = f"Context from surrounding document:\n{context}\n"
        else:
            context_section = ""

        return TABLE_ANALYSIS_USER_PROMPT.format(
            context_section=context_section, table_content=table_content
        )

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Try to extract JSON from markdown code block
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

    async def _call_llm_api(
        self,
        user_prompt: str,
    ) -> dict[str, Any]:
        """
        Call the LLM API for table analysis.

        Args:
            user_prompt: The user prompt with table content

        Returns:
            Parsed JSON response from the model
        """
        api_key = self.settings.llm_api_key
        base_url = self.settings.llm_base_url or ""

        # Ensure base_url ends properly
        if base_url and not base_url.endswith("/"):
            base_url += "/"

        url = f"{base_url}chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": TABLE_ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,  # Low temperature for consistent extraction
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
        Process a table and extract structured information.

        Args:
            content: Table content (markdown, HTML, CSV, or file path)
            context: Surrounding text context for better understanding
            **kwargs: Additional arguments (caption, footnote, etc.)

        Returns:
            ProcessedContent with description, entities, and relationships
        """
        # Handle different input types
        source_path = None
        if isinstance(content, Path):
            source_path = str(content)
            content = content.read_text(encoding="utf-8")
        elif isinstance(content, bytes):
            content = content.decode("utf-8")
        elif isinstance(content, str) and Path(content).exists():
            source_path = content
            content = Path(content).read_text(encoding="utf-8")

        # Add caption/footnote to context if provided
        caption = kwargs.get("caption", "")
        footnote = kwargs.get("footnote", "")
        if caption:
            context = f"Table Caption: {caption}\n{context}".strip()
        if footnote:
            context = f"{context}\nTable Footnote: {footnote}".strip()

        # Normalize table format
        normalized_table = self._normalize_table(content)

        # Build prompt and call API
        user_prompt = self._build_user_prompt(normalized_table, context)

        try:
            analysis = await self._call_llm_api(user_prompt)

            # Extract fields with defaults
            description = analysis.get("description", "Table content")
            summary = analysis.get("summary", "")
            table_type = analysis.get("table_type", "data")
            columns = analysis.get("columns", [])
            row_count = analysis.get("row_count", 0)
            key_insights = analysis.get("key_insights", [])
            statistics = analysis.get("statistics", {})

            # Convert entities
            entities = [
                ExtractedEntity(
                    name=e.get("name", "Unknown"),
                    entity_type=e.get("type", "Data"),
                    description=e.get("description", ""),
                    source_content_type=ContentType.TABLE,
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
                content_type=ContentType.TABLE,
                source_path=source_path,
                description=description,
                summary=summary,
                entities=entities,
                relationships=relationships,
                raw_text=normalized_table,
                metadata={
                    "table_type": table_type,
                    "columns": columns,
                    "row_count": row_count,
                    "key_insights": key_insights,
                    "statistics": statistics,
                },
                context_used=context,
            )

        except Exception as e:
            logger.error(f"Error processing table: {e}")
            # Return a minimal result on error
            return ProcessedContent(
                content_type=ContentType.TABLE,
                source_path=source_path,
                description=f"[Error processing table: {e}]",
                summary="Table processing failed",
                raw_text=normalized_table,
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
        Generate a text description of the table.

        Args:
            content: Table content
            context: Surrounding text context
            **kwargs: Additional arguments

        Returns:
            Natural language description of the table
        """
        result = await self.process(content, context, **kwargs)
        return result.description

    async def extract_as_dict(
        self,
        content: str | Path | bytes,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Extract table data as a list of dictionaries.

        Args:
            content: Table content
            **kwargs: Additional arguments

        Returns:
            List of row dictionaries with column names as keys
        """
        # Handle different input types
        if isinstance(content, Path):
            content = content.read_text(encoding="utf-8")
        elif isinstance(content, bytes):
            content = content.decode("utf-8")
        elif isinstance(content, str) and Path(content).exists():
            content = Path(content).read_text(encoding="utf-8")

        normalized = self._normalize_table(content)

        # Parse markdown table
        lines = [line.strip() for line in normalized.split("\n") if line.strip()]
        if len(lines) < 2:
            return []

        # Extract headers
        header_line = lines[0]
        headers = [h.strip() for h in header_line.split("|") if h.strip()]

        # Skip separator line and extract data rows
        data_rows = []
        for line in lines[2:]:  # Skip header and separator
            if "|" in line:
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if cells:
                    row_dict = {}
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            row_dict[headers[i]] = cell
                    data_rows.append(row_dict)

        return data_rows


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def process_table(
    table_content: str,
    context: str = "",
    **kwargs: Any,
) -> ProcessedContent:
    """
    Convenience function to process a table.

    Args:
        table_content: Table content (markdown, HTML, CSV)
        context: Optional surrounding context
        **kwargs: Additional arguments

    Returns:
        ProcessedContent with extracted information
    """
    processor = TableProcessor()
    try:
        return await processor.process(table_content, context, **kwargs)
    finally:
        await processor.close()


async def describe_table(
    table_content: str,
    context: str = "",
    **kwargs: Any,
) -> str:
    """
    Convenience function to get a table description.

    Args:
        table_content: Table content
        context: Optional surrounding context
        **kwargs: Additional arguments

    Returns:
        Text description of the table
    """
    processor = TableProcessor()
    try:
        return await processor.describe(table_content, context, **kwargs)
    finally:
        await processor.close()


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
        """Test the table processor."""
        # Sample table for testing
        sample_table = """
| Product | Q1 Sales | Q2 Sales | Q3 Sales | Q4 Sales |
|---------|----------|----------|----------|----------|
| Widget A | $10,000 | $12,500 | $15,000 | $18,000 |
| Widget B | $8,000 | $9,500 | $11,000 | $12,500 |
| Widget C | $5,500 | $6,000 | $7,500 | $9,000 |
| Total | $23,500 | $28,000 | $33,500 | $39,500 |
"""

        if len(sys.argv) > 1:
            # Read table from file
            table_path = sys.argv[1]
            context = sys.argv[2] if len(sys.argv) > 2 else ""
            print(f"Processing table from: {table_path}")
            processor = TableProcessor()
            result = await processor.process(table_path, context)
        else:
            # Use sample table
            print("Processing sample sales table...")
            processor = TableProcessor()
            result = await processor.process(
                sample_table, context="Quarterly sales report for 2024"
            )

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"\nDescription:\n{result.description}")
        print(f"\nSummary: {result.summary}")
        print(f"\nTable Type: {result.metadata.get('table_type', 'unknown')}")

        columns = result.metadata.get("columns", [])
        if columns:
            print(f"\nColumns ({len(columns)}):")
            for col in columns:
                print(f"  - {col.get('name')}: {col.get('data_type')} - {col.get('description', '')}")

        if result.entities:
            print(f"\nEntities ({len(result.entities)}):")
            for e in result.entities:
                print(f"  - {e.name} ({e.entity_type}): {e.description}")

        if result.relationships:
            print(f"\nRelationships ({len(result.relationships)}):")
            for r in result.relationships:
                print(f"  - {r.source} --[{r.relationship_type}]--> {r.target}")

        insights = result.metadata.get("key_insights", [])
        if insights:
            print(f"\nKey Insights:")
            for insight in insights:
                print(f"  - {insight}")

        stats = result.metadata.get("statistics", {})
        if stats:
            print(f"\nStatistics: {stats}")

        print("\n" + "=" * 60)
        print("Chunk Content (for embedding):")
        print("=" * 60)
        print(result.to_chunk_content())

        await processor.close()

    asyncio.run(main())

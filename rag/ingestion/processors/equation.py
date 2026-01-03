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
Equation processor for multimodal RAG.

This module provides LLM integration for processing mathematical equations,
extracting descriptions, variable definitions, and conceptual relationships
for the knowledge graph.

Supports:
- LaTeX equations
- MathML
- Plain text mathematical expressions
- Unicode math symbols

Usage:
    from rag.ingestion.processors import EquationProcessor

    processor = EquationProcessor()
    result = await processor.process(
        "E = mc^2",
        context="Einstein's famous equation from special relativity."
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


class VariableInfo(BaseModel):
    """Information about a variable in an equation."""

    symbol: str = Field(description="The variable symbol (e.g., 'x', 'E', 'λ')")
    name: str = Field(description="Full name of the variable")
    description: str = Field(default="", description="What this variable represents")
    unit: str = Field(default="", description="Unit of measurement if applicable")
    typical_values: str = Field(default="", description="Typical range or values")


class EquationAnalysis(BaseModel):
    """Structured output from equation analysis."""

    description: str = Field(
        description="Detailed explanation of what the equation represents and means"
    )
    summary: str = Field(
        description="One-sentence summary suitable for search/retrieval"
    )
    equation_type: str = Field(
        description="Type of equation (algebraic, differential, integral, statistical, etc.)"
    )
    field: str = Field(
        description="Scientific/mathematical field (physics, calculus, statistics, etc.)"
    )
    name: str = Field(
        default="", description="Common name if the equation is well-known"
    )
    variables: list[VariableInfo] = Field(
        default_factory=list, description="Variables used in the equation"
    )
    entities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Concepts and entities related to the equation",
    )
    relationships: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Relationships between concepts with source, target, and type",
    )
    applications: list[str] = Field(
        default_factory=list,
        description="Practical applications or use cases",
    )
    prerequisites: list[str] = Field(
        default_factory=list,
        description="Mathematical concepts needed to understand this equation",
    )


# =============================================================================
# PROMPTS
# =============================================================================

EQUATION_ANALYSIS_SYSTEM_PROMPT = """You are an expert mathematician and scientist for a knowledge graph system.
Your task is to analyze mathematical equations and extract structured information.

For each equation, provide:
1. A detailed explanation of what the equation represents and means
2. A brief one-sentence summary for search indexing
3. The type of equation (algebraic, differential, integral, statistical, trigonometric, etc.)
4. The scientific/mathematical field it belongs to
5. The common name if it's a well-known equation
6. Information about each variable (symbol, name, description, unit)
7. Related concepts and entities
8. Relationships between concepts
9. Practical applications
10. Mathematical prerequisites

For entities, extract:
- name: The concept or entity name
- type: Concept, Law, Theorem, Constant, Function, Operator, etc.
- description: Brief description in this context

For relationships, extract:
- source: Source entity name
- target: Target entity name
- type: Relationship type in SCREAMING_SNAKE_CASE (e.g., DERIVED_FROM, APPLIES_TO, USES_CONCEPT)

Be thorough and educational. Explain the equation in a way that would help someone understand and find it later."""

EQUATION_ANALYSIS_USER_PROMPT = """Analyze this mathematical equation and extract structured information.

{context_section}

Equation:
{equation_content}

Format: {equation_format}

Provide your analysis as a JSON object with these fields:
- description: Detailed explanation of the equation
- summary: One-sentence summary
- equation_type: Type of equation
- field: Scientific/mathematical field
- name: Common name (if well-known)
- variables: List of {{symbol, name, description, unit, typical_values}}
- entities: List of {{name, type, description}}
- relationships: List of {{source, target, type}}
- applications: List of practical applications
- prerequisites: List of prerequisite concepts"""


# =============================================================================
# EQUATION PROCESSOR
# =============================================================================


class EquationProcessor(BaseProcessor):
    """
    Process mathematical equations using LLM for RAG.

    Extracts descriptions, variable definitions, and conceptual relationships
    from equations using language models.

    Attributes:
        settings: Application settings
    """

    def __init__(self) -> None:
        """Initialize the equation processor."""
        self.settings = load_settings()
        self._client: httpx.AsyncClient | None = None

    @property
    def content_type(self) -> ContentType:
        """Return the content type this processor handles."""
        return ContentType.EQUATION

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    def _detect_format(self, content: str) -> str:
        """
        Detect the format of the equation.

        Args:
            content: Equation content

        Returns:
            Format string: "latex", "mathml", "unicode", or "plain"
        """
        content = content.strip()

        # LaTeX indicators
        latex_patterns = [
            r"\\frac\{",
            r"\\sqrt",
            r"\\sum",
            r"\\int",
            r"\\partial",
            r"\\alpha",
            r"\\beta",
            r"\\gamma",
            r"\\theta",
            r"\\lambda",
            r"\\pi",
            r"\\infty",
            r"\^{",
            r"_{",
            r"\\left",
            r"\\right",
            r"\\cdot",
            r"\\times",
            r"\\nabla",
            r"\\Delta",
        ]
        for pattern in latex_patterns:
            if re.search(pattern, content):
                return "latex"

        # MathML indicators
        if "<math" in content.lower() or "<mrow" in content.lower():
            return "mathml"

        # Unicode math symbols
        unicode_math = ["∑", "∫", "∂", "∞", "√", "≠", "≤", "≥", "±", "×", "÷", "∇", "Δ", "∈", "∉", "⊂", "⊃", "∪", "∩"]
        for symbol in unicode_math:
            if symbol in content:
                return "unicode"

        return "plain"

    def _normalize_equation(self, content: str, fmt: str) -> str:
        """
        Normalize equation content for display.

        Args:
            content: Raw equation content
            fmt: Detected format

        Returns:
            Normalized equation string
        """
        content = content.strip()

        # Remove surrounding $ or $$ for LaTeX
        if fmt == "latex":
            content = re.sub(r"^\$\$?|\$\$?$", "", content).strip()

        # MathML - extract the equation if wrapped in tags
        if fmt == "mathml":
            # Keep as-is for now, LLM can parse it
            pass

        return content

    def _build_user_prompt(self, equation: str, fmt: str, context: str) -> str:
        """Build the user prompt with equation and context."""
        if context:
            context_section = f"Context:\n{context}\n"
        else:
            context_section = ""

        return EQUATION_ANALYSIS_USER_PROMPT.format(
            context_section=context_section,
            equation_content=equation,
            equation_format=fmt,
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
        Call the LLM API for equation analysis.

        Args:
            user_prompt: The user prompt with equation content

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
                {"role": "system", "content": EQUATION_ANALYSIS_SYSTEM_PROMPT},
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
        Process an equation and extract structured information.

        Args:
            content: Equation content (LaTeX, MathML, Unicode, or plain text)
            context: Surrounding text context for better understanding
            **kwargs: Additional arguments (format, caption, etc.)

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
        elif isinstance(content, str) and len(content) < 500 and Path(content).exists():
            source_path = content
            content = Path(content).read_text(encoding="utf-8")

        # Detect format or use provided format
        fmt = kwargs.get("format") or self._detect_format(content)

        # Normalize equation
        normalized = self._normalize_equation(content, fmt)

        # Add caption to context if provided
        caption = kwargs.get("caption", "")
        if caption:
            context = f"Equation Caption: {caption}\n{context}".strip()

        # Build prompt and call API
        user_prompt = self._build_user_prompt(normalized, fmt, context)

        try:
            analysis = await self._call_llm_api(user_prompt)

            # Extract fields with defaults
            description = analysis.get("description", "Mathematical equation")
            summary = analysis.get("summary", "")
            equation_type = analysis.get("equation_type", "algebraic")
            field = analysis.get("field", "mathematics")
            equation_name = analysis.get("name", "")
            variables = analysis.get("variables", [])
            applications = analysis.get("applications", [])
            prerequisites = analysis.get("prerequisites", [])

            # Convert entities
            entities = [
                ExtractedEntity(
                    name=e.get("name", "Unknown"),
                    entity_type=e.get("type", "Concept"),
                    description=e.get("description", ""),
                    source_content_type=ContentType.EQUATION,
                )
                for e in analysis.get("entities", [])
                if isinstance(e, dict)
            ]

            # Add variables as entities
            for var in variables:
                if isinstance(var, dict) and var.get("symbol"):
                    var_desc = var.get("description", "")
                    if var.get("unit"):
                        var_desc += f" (unit: {var['unit']})"
                    entities.append(
                        ExtractedEntity(
                            name=var.get("name", var.get("symbol")),
                            entity_type="Variable",
                            description=var_desc,
                            properties={"symbol": var.get("symbol"), "unit": var.get("unit", "")},
                            source_content_type=ContentType.EQUATION,
                        )
                    )

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
                content_type=ContentType.EQUATION,
                source_path=source_path,
                description=description,
                summary=summary,
                entities=entities,
                relationships=relationships,
                raw_text=normalized,
                metadata={
                    "equation_type": equation_type,
                    "field": field,
                    "name": equation_name,
                    "format": fmt,
                    "variables": variables,
                    "applications": applications,
                    "prerequisites": prerequisites,
                },
                context_used=context,
            )

        except Exception as e:
            logger.error(f"Error processing equation: {e}")
            # Return a minimal result on error
            return ProcessedContent(
                content_type=ContentType.EQUATION,
                source_path=source_path,
                description=f"[Error processing equation: {e}]",
                summary="Equation processing failed",
                raw_text=normalized,
                metadata={"error": str(e), "format": fmt},
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
        Generate a text description of the equation.

        Args:
            content: Equation content
            context: Surrounding text context
            **kwargs: Additional arguments

        Returns:
            Natural language description of the equation
        """
        result = await self.process(content, context, **kwargs)
        return result.description

    async def get_variables(
        self,
        content: str | Path | bytes,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Extract variable information from an equation.

        Args:
            content: Equation content
            **kwargs: Additional arguments

        Returns:
            List of variable dictionaries with symbol, name, description, unit
        """
        result = await self.process(content, **kwargs)
        return result.metadata.get("variables", [])


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def process_equation(
    equation: str,
    context: str = "",
    **kwargs: Any,
) -> ProcessedContent:
    """
    Convenience function to process an equation.

    Args:
        equation: Equation content (LaTeX, MathML, or plain text)
        context: Optional surrounding context
        **kwargs: Additional arguments

    Returns:
        ProcessedContent with extracted information
    """
    processor = EquationProcessor()
    try:
        return await processor.process(equation, context, **kwargs)
    finally:
        await processor.close()


async def describe_equation(
    equation: str,
    context: str = "",
    **kwargs: Any,
) -> str:
    """
    Convenience function to get an equation description.

    Args:
        equation: Equation content
        context: Optional surrounding context
        **kwargs: Additional arguments

    Returns:
        Text description of the equation
    """
    processor = EquationProcessor()
    try:
        return await processor.describe(equation, context, **kwargs)
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

    # Sample equations for testing
    SAMPLE_EQUATIONS = {
        "einstein": ("E = mc^2", "Einstein's mass-energy equivalence from special relativity"),
        "quadratic": (
            "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}",
            "The quadratic formula for solving ax^2 + bx + c = 0",
        ),
        "schrodinger": (
            "i\\hbar\\frac{\\partial}{\\partial t}\\Psi = \\hat{H}\\Psi",
            "Time-dependent Schrödinger equation in quantum mechanics",
        ),
        "bayes": (
            "P(A|B) = \\frac{P(B|A) \\cdot P(A)}{P(B)}",
            "Bayes' theorem for conditional probability",
        ),
        "euler": (
            "e^{i\\pi} + 1 = 0",
            "Euler's identity connecting e, i, pi, 1, and 0",
        ),
    }

    async def main():
        """Test the equation processor."""
        processor = EquationProcessor()

        if len(sys.argv) > 1:
            # Use provided equation
            equation = sys.argv[1]
            context = sys.argv[2] if len(sys.argv) > 2 else ""
            print(f"Processing equation: {equation}")
        else:
            # Use sample equation
            name = "bayes"
            equation, context = SAMPLE_EQUATIONS[name]
            print(f"Processing sample equation ({name}): {equation}")

        result = await processor.process(equation, context)

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"\nEquation: {result.raw_text}")
        print(f"\nName: {result.metadata.get('name', 'N/A')}")
        print(f"Type: {result.metadata.get('equation_type', 'N/A')}")
        print(f"Field: {result.metadata.get('field', 'N/A')}")
        print(f"Format: {result.metadata.get('format', 'N/A')}")
        print(f"\nDescription:\n{result.description}")
        print(f"\nSummary: {result.summary}")

        variables = result.metadata.get("variables", [])
        if variables:
            print(f"\nVariables ({len(variables)}):")
            for var in variables:
                unit = f" [{var.get('unit')}]" if var.get("unit") else ""
                print(f"  - {var.get('symbol')} ({var.get('name')}){unit}: {var.get('description', '')}")

        if result.entities:
            print(f"\nEntities ({len(result.entities)}):")
            for e in result.entities:
                print(f"  - {e.name} ({e.entity_type}): {e.description}")

        if result.relationships:
            print(f"\nRelationships ({len(result.relationships)}):")
            for r in result.relationships:
                print(f"  - {r.source} --[{r.relationship_type}]--> {r.target}")

        applications = result.metadata.get("applications", [])
        if applications:
            print(f"\nApplications:")
            for app in applications:
                print(f"  - {app}")

        prerequisites = result.metadata.get("prerequisites", [])
        if prerequisites:
            print(f"\nPrerequisites:")
            for prereq in prerequisites:
                print(f"  - {prereq}")

        print("\n" + "=" * 60)
        print("Chunk Content (for embedding):")
        print("=" * 60)
        print(result.to_chunk_content())

        await processor.close()

    asyncio.run(main())

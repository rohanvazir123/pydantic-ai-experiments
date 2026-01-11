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
PDF Question Generator using RAG-Anything Multimodal Processors.

This script processes a single PDF using RAG-Anything's multimodal pipeline
and generates a list of questions based on the extracted content.

Usage:
    python -m rag.ingestion.processors.pdf_question_generator <pdf_path>
    python -m rag.ingestion.processors.pdf_question_generator C:/Users/rohan/Desktop/csd168/l1.pdf
    python -m rag.ingestion.processors.pdf_question_generator --use-ollama C:/path/to/file.pdf
    python -m rag.ingestion.processors.pdf_question_generator --list-dir C:/Users/rohan/Desktop/csd168

Requirements:
    - raganything installed
    - MinerU parser installed (for PDF parsing)
    - Ollama running locally OR OpenAI API key
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.ingestion.processors.lightrag_utils import (
    LightRAGConfig,
    get_ollama_llm_funcs,
    get_ollama_llm_funcs_async,
    get_ollama_embedding_func,
    get_openai_llm_funcs,
    get_openai_embedding_func,
    check_ollama_available,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result from processing a PDF."""

    pdf_path: str
    title: str = ""
    num_pages: int = 0
    num_text_chunks: int = 0
    num_images: int = 0
    num_tables: int = 0
    num_equations: int = 0
    content_summary: str = ""
    questions: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    context_chunks: str = ""  # Formatted context sent to LLM
    raw_llm_response: str = ""  # Raw LLM response
    error: str | None = None


QUESTION_GENERATION_SYSTEM_PROMPT = """You are an expert AI assistant that generates high-quality research questions.
Use ONLY the provided context chunks to generate questions.
If information is missing or unclear, acknowledge it.
Each question must be answerable using the provided context."""

QUESTION_GENERATION_PROMPT = """TASK:
Generate 5-8 high-quality research questions based on the provided context chunks.
Questions should:
1. Test understanding of key concepts
2. Explore relationships between topics
3. Encourage critical thinking
4. Be answerable using the provided chunks

CONTEXT:
{context_chunks}

OUTPUT FORMAT (valid JSON only):
{{
    "questions": [
        {{
            "question": "What is the relationship between X and Y?",
            "supported_by": ["c1", "c2"],
            "difficulty": "medium"
        }},
        {{
            "question": "How does concept Z apply to problem W?",
            "supported_by": ["c3"],
            "difficulty": "hard"
        }}
    ]
}}

Generate the JSON response:"""


@dataclass
class ChunkContext:
    """A chunk with context for LLM."""
    chunk_id: str
    content: str
    entity_name: str = ""
    entity_type: str = ""
    page_idx: int = 0
    content_type: str = "text"  # text, table, equation, image


def format_chunks_as_context(chunks: list[ChunkContext], max_chars: int = 12000) -> str:
    """Format chunks as structured context for LLM.

    Args:
        chunks: List of ChunkContext objects
        max_chars: Maximum total characters for context

    Returns:
        Formatted context string with chunk IDs and entity info
    """
    formatted_chunks = []
    total_chars = 0

    for chunk in chunks:
        # Build chunk header
        header_parts = [f"[chunk_id={chunk.chunk_id}]"]
        if chunk.entity_name:
            header_parts.append(f"[entity={chunk.entity_name}]")
        if chunk.entity_type:
            header_parts.append(f"[type={chunk.entity_type}]")
        if chunk.content_type != "text":
            header_parts.append(f"[content_type={chunk.content_type}]")
        header_parts.append(f"[page={chunk.page_idx}]")

        header = "".join(header_parts)

        # Truncate content if needed
        content = chunk.content
        remaining = max_chars - total_chars - len(header) - 10
        if remaining < 100:
            break
        if len(content) > remaining:
            content = content[:remaining] + "..."

        formatted = f"{header}\n{content}\n"
        formatted_chunks.append(formatted)
        total_chars += len(formatted)

    return "\n".join(formatted_chunks)


def extract_chunks_from_lightrag(rag, limit: int = 30) -> list[ChunkContext]:
    """Extract stored chunks from LightRAG instance.

    Args:
        rag: LightRAG instance
        limit: Maximum number of chunks to extract

    Returns:
        List of ChunkContext objects
    """
    chunks = []

    # Access the text_chunks KV store
    try:
        text_chunks_data = rag.text_chunks._data if hasattr(rag.text_chunks, '_data') else {}

        for i, (chunk_id, chunk_data) in enumerate(text_chunks_data.items()):
            if i >= limit:
                break

            content = chunk_data.get("content", "")
            if not content:
                continue

            chunks.append(ChunkContext(
                chunk_id=f"c{i+1}",
                content=content,
                entity_name=chunk_data.get("entity_name", ""),
                entity_type=chunk_data.get("entity_type", ""),
                page_idx=chunk_data.get("chunk_order_index", 0),
                content_type=chunk_data.get("content_type", "text"),
            ))
    except Exception as e:
        logger.warning(f"Could not extract chunks from LightRAG: {e}")

    return chunks


def extract_chunks_from_content_list(
    content_list: list[dict],
    entities: list[dict] | None = None,
    limit: int = 30,
) -> list[ChunkContext]:
    """Extract chunks from MinerU/PyPDF2 content list.

    Args:
        content_list: List of content items from parser
        entities: Optional list of extracted entities
        limit: Maximum number of chunks

    Returns:
        List of ChunkContext objects
    """
    chunks = []
    entity_map = {}

    # Build entity map if provided
    if entities:
        for ent in entities:
            if "source_chunk" in ent:
                entity_map[ent["source_chunk"]] = ent

    for i, item in enumerate(content_list[:limit]):
        item_type = item.get("type", "text")
        page_idx = item.get("page_idx", 0)

        # Extract content based on type
        if item_type == "text":
            content = item.get("text", "")
        elif item_type == "table":
            caption = item.get("table_caption", [""])[0] if item.get("table_caption") else ""
            body = item.get("table_body", "")
            content = f"[TABLE] {caption}\n{body}" if caption else f"[TABLE]\n{body}"
        elif item_type == "equation":
            eq_text = item.get("text", item.get("latex", ""))
            content = f"[EQUATION] {eq_text}"
        elif item_type == "image":
            caption = item.get("image_caption", [""])[0] if item.get("image_caption") else ""
            content = f"[IMAGE] {caption}" if caption else "[IMAGE]"
        else:
            content = str(item.get("content", item.get("text", "")))

        if not content or len(content.strip()) < 10:
            continue

        chunk_id = f"c{len(chunks)+1}"

        # Check for associated entity
        entity_info = entity_map.get(chunk_id, {})

        chunks.append(ChunkContext(
            chunk_id=chunk_id,
            content=content,
            entity_name=entity_info.get("entity_name", ""),
            entity_type=entity_info.get("entity_type", ""),
            page_idx=page_idx,
            content_type=item_type,
        ))

    return chunks


async def process_pdf_with_raganything(
    pdf_path: str,
    working_dir: str,
    use_ollama: bool = True,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ProcessingResult:
    """Process a PDF using RAG-Anything and extract content.

    Args:
        pdf_path: Path to the PDF file
        working_dir: Directory for RAG storage
        use_ollama: Use Ollama for LLM/embeddings
        api_key: OpenAI API key (if not using Ollama)
        base_url: Optional API base URL

    Returns:
        ProcessingResult with extracted content and generated questions
    """
    from raganything import RAGAnything, RAGAnythingConfig

    result = ProcessingResult(pdf_path=pdf_path)
    pdf_name = Path(pdf_path).stem
    result.title = pdf_name

    try:
        # Setup LLM and embedding functions
        if use_ollama:
            if not check_ollama_available():
                result.error = "Ollama not available. Start with 'ollama serve'"
                return result

            llm_func, vision_func = get_ollama_llm_funcs()
            embedding_func = get_ollama_embedding_func()
            logger.info("Using Ollama for LLM and embeddings")
        else:
            if not api_key:
                result.error = "API key required when not using Ollama"
                return result

            llm_func, vision_func = get_openai_llm_funcs(api_key, base_url)
            embedding_func = get_openai_embedding_func(api_key, base_url)
            logger.info("Using OpenAI for LLM and embeddings")

        # Configure RAG-Anything
        config = RAGAnythingConfig(
            working_dir=working_dir,
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # Initialize RAG-Anything
        logger.info(f"Initializing RAG-Anything for: {pdf_path}")
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_func,
            vision_model_func=vision_func,
            embedding_func=embedding_func,
        )

        # Process the document
        output_dir = os.path.join(working_dir, "parsed_output", pdf_name)
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Processing PDF: {pdf_path}")
        await rag.process_document_complete(
            file_path=pdf_path,
            output_dir=output_dir,
            parse_method="auto",
        )

        # Collect content statistics from parsed output
        content_list = []
        parsed_json = os.path.join(output_dir, f"{pdf_name}_content_list.json")
        if os.path.exists(parsed_json):
            with open(parsed_json, "r", encoding="utf-8") as f:
                content_list = json.load(f)

        # Count content types
        text_content = []
        for item in content_list:
            item_type = item.get("type", "")
            if item_type == "text":
                result.num_text_chunks += 1
                text_content.append(item.get("text", ""))
            elif item_type == "image":
                result.num_images += 1
            elif item_type == "table":
                result.num_tables += 1
            elif item_type == "equation":
                result.num_equations += 1

        # Get page count
        if content_list:
            result.num_pages = max(item.get("page_idx", 0) for item in content_list) + 1

        # Build content summary for question generation
        combined_text = "\n".join(text_content[:50])  # First 50 text chunks
        if len(combined_text) > 8000:
            combined_text = combined_text[:8000] + "..."

        multimodal_summary = []
        if result.num_images > 0:
            multimodal_summary.append(f"- Contains {result.num_images} images/figures")
        if result.num_tables > 0:
            multimodal_summary.append(f"- Contains {result.num_tables} tables")
        if result.num_equations > 0:
            multimodal_summary.append(f"- Contains {result.num_equations} equations")

        multimodal_text = "\n".join(multimodal_summary) if multimodal_summary else ""

        # Generate questions using LLM
        logger.info("Generating questions based on content...")
        question_prompt = QUESTION_GENERATION_PROMPT.format(
            content=combined_text,
            multimodal_summary=f"\nMultimodal Content:\n{multimodal_text}" if multimodal_text else "",
        )

        try:
            response = llm_func(question_prompt, system_prompt="You are a helpful assistant that generates educational questions.")

            # Parse JSON response
            # Handle potential markdown code blocks
            response_text = response.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            questions_data = json.loads(response_text)
            result.questions = questions_data.get("questions", [])
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse questions JSON: {e}")
            # Try to extract questions manually
            result.questions = [f"Error parsing questions: {response[:200]}..."]
        except Exception as e:
            logger.warning(f"Failed to generate questions: {e}")
            result.questions = [f"Question generation failed: {str(e)}"]

        # Query for entities (optional)
        try:
            entity_response = await rag.aquery(
                "List the main topics and entities discussed in this document.",
                mode="local",
            )
            result.entities = [entity_response[:500]] if entity_response else []
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")

        result.content_summary = f"Processed {result.num_pages} pages with {result.num_text_chunks} text blocks"

        return result

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        import traceback
        traceback.print_exc()
        result.error = str(e)
        return result


async def process_pdf_simple(
    pdf_path: str,
    working_dir: str,
    use_ollama: bool = True,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ProcessingResult:
    """Simpler PDF processing using MinerU directly + modal processors.

    This approach parses the PDF first, then processes multimodal content.

    Args:
        pdf_path: Path to the PDF file
        working_dir: Directory for RAG storage
        use_ollama: Use Ollama for LLM/embeddings
        api_key: OpenAI API key (if not using Ollama)
        base_url: Optional API base URL

    Returns:
        ProcessingResult with extracted content and generated questions
    """
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import initialize_pipeline_status

    result = ProcessingResult(pdf_path=pdf_path)
    pdf_name = Path(pdf_path).stem
    result.title = pdf_name

    try:
        # Setup LLM and embedding functions
        if use_ollama:
            if not check_ollama_available():
                result.error = "Ollama not available. Start with 'ollama serve'"
                return result

            llm_func, vision_func = get_ollama_llm_funcs()
            # Get async versions for modal processors (they require async functions)
            async_llm_func, async_vision_func = await get_ollama_llm_funcs_async()
            embedding_func = get_ollama_embedding_func()
            logger.info("Using Ollama for LLM and embeddings")
        else:
            if not api_key:
                result.error = "API key required when not using Ollama"
                return result

            llm_func, vision_func = get_openai_llm_funcs(api_key, base_url)
            # OpenAI functions are already async-compatible, use them directly
            async_llm_func, async_vision_func = llm_func, vision_func
            embedding_func = get_openai_embedding_func(api_key, base_url)
            logger.info("Using OpenAI for LLM and embeddings")

        # Initialize LightRAG with async LLM function
        os.makedirs(working_dir, exist_ok=True)
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=async_llm_func,  # Use async for LightRAG internals
            embedding_func=embedding_func,
        )
        await rag.initialize_storages()
        await initialize_pipeline_status()

        # Try to parse PDF with MinerU VLM (GPU required)
        logger.info(f"Parsing PDF with MinerU: {pdf_path}")
        content_list = []

        try:
            from rag.ingestion.chunkers.mineru import MinerUParser, ParsedDocument

            parser = MinerUParser(describe_figures=True)
            parsed_doc = await parser.parse_file(pdf_path)
            await parser.close()

            # Convert ParsedDocument blocks to content_list format
            for block in parsed_doc.blocks:
                item = {
                    "page_idx": block.page_number - 1,  # 0-indexed
                    "type": "text",  # default
                }

                if block.block_type.value in ("text", "header", "title", "list"):
                    item["type"] = "text"
                    item["text"] = block.content
                    item["text_level"] = 1 if block.block_type.value in ("header", "title") else 0
                elif block.block_type.value == "table":
                    item["type"] = "table"
                    item["table_body"] = block.content
                    item["table_caption"] = []
                elif block.block_type.value in ("figure", "image"):
                    item["type"] = "image"
                    item["img_path"] = ""
                    item["image_caption"] = [block.figure_description] if block.figure_description else []
                elif block.block_type.value == "equation":
                    item["type"] = "equation"
                    item["text"] = block.content
                    item["text_format"] = "LaTeX"
                else:
                    item["type"] = "text"
                    item["text"] = block.content

                content_list.append(item)

            logger.info(f"MinerU VLM extracted {len(content_list)} content items from {parsed_doc.total_pages} pages")

        except ImportError as e:
            logger.warning(f"MinerU VLM not available: {e}. Using basic text extraction.")
            # Fallback: try PyPDF2 or similar
            try:
                import PyPDF2
                with open(pdf_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for i, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text:
                            content_list.append({
                                "type": "text",
                                "text": text,
                                "page_idx": i,
                            })
                    result.num_pages = len(pdf_reader.pages)
            except ImportError:
                logger.error("Neither MinerU nor PyPDF2 available for PDF parsing")
                result.error = "No PDF parser available. Install magic-pdf or PyPDF2."
                return result

        except Exception as e:
            logger.error(f"MinerU parsing failed: {e}")
            result.error = f"PDF parsing failed: {str(e)}"
            return result

        # Process content list
        text_content = []
        table_content = []
        equation_content = []
        image_content = []

        for item in content_list:
            item_type = item.get("type", "")
            if item_type == "text":
                result.num_text_chunks += 1
                text_content.append(item.get("text", ""))
            elif item_type == "image":
                result.num_images += 1
                image_content.append(item)
            elif item_type == "table":
                result.num_tables += 1
                table_content.append(item)
            elif item_type == "equation":
                result.num_equations += 1
                equation_content.append(item)

        if content_list:
            result.num_pages = max(item.get("page_idx", 0) for item in content_list) + 1

        # Process multimodal content with modal processors
        from raganything.modalprocessors import (
            TableModalProcessor,
            EquationModalProcessor,
            ImageModalProcessor,
        )

        multimodal_descriptions = []

        # Process tables
        if table_content:
            logger.info(f"Processing {len(table_content)} tables...")
            table_processor = TableModalProcessor(lightrag=rag, modal_caption_func=async_llm_func)
            for i, table in enumerate(table_content[:3]):  # Limit to first 3
                try:
                    desc, entity_info, _ = await table_processor.process_multimodal_content(
                        modal_content=table,
                        content_type="table",
                        file_path=pdf_path,
                        entity_name=f"Table {i+1} from {pdf_name}",
                    )
                    if desc:
                        multimodal_descriptions.append(f"Table {i+1}: {desc[:200]}")
                except Exception as e:
                    logger.warning(f"Table processing failed: {e}")

        # Process equations
        if equation_content:
            logger.info(f"Processing {len(equation_content)} equations...")
            equation_processor = EquationModalProcessor(lightrag=rag, modal_caption_func=async_llm_func)
            for i, eq in enumerate(equation_content[:5]):  # Limit to first 5
                try:
                    desc, entity_info, _ = await equation_processor.process_multimodal_content(
                        modal_content=eq,
                        content_type="equation",
                        file_path=pdf_path,
                        entity_name=f"Equation {i+1} from {pdf_name}",
                    )
                    if desc:
                        multimodal_descriptions.append(f"Equation {i+1}: {desc[:200]}")
                except Exception as e:
                    logger.warning(f"Equation processing failed: {e}")

        # Build chunk-based context for question generation
        logger.info("Building chunk context for question generation...")
        chunks = extract_chunks_from_content_list(content_list, limit=30)

        # Add multimodal descriptions as additional chunks
        for i, desc in enumerate(multimodal_descriptions):
            chunks.append(ChunkContext(
                chunk_id=f"m{i+1}",
                content=desc,
                entity_name=f"Multimodal Content {i+1}",
                entity_type="multimodal",
                content_type="description",
            ))

        # Format chunks as structured context
        context_chunks = format_chunks_as_context(chunks, max_chars=10000)
        result.context_chunks = context_chunks  # Store for output
        logger.info(f"Prepared {len(chunks)} chunks for context")

        # Generate questions using chunk-based context
        logger.info("Generating questions based on chunk context...")
        question_prompt = QUESTION_GENERATION_PROMPT.format(context_chunks=context_chunks)

        try:
            response = await async_llm_func(
                question_prompt,
                system_prompt=QUESTION_GENERATION_SYSTEM_PROMPT,
            )
            result.raw_llm_response = response  # Store raw response

            # Parse JSON response
            response_text = response.strip()
            # Handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            questions_data = json.loads(response_text)
            result.questions = questions_data.get("questions", [])

            # Store chunk IDs for reference
            result.entities = [f"Chunks used: {', '.join(c.chunk_id for c in chunks[:10])}"]

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse questions JSON: {e}")
            logger.debug(f"Raw response: {response[:500]}")
            # Try to extract questions from raw response
            questions = []
            for line in response.split("\n"):
                line = line.strip()
                if "?" in line and len(line) > 10:
                    # Clean up JSON artifacts
                    q = line
                    q = q.replace('"question":', '').replace('"', '').strip()
                    q = q.rstrip(',').strip()
                    if q and not q.startswith('{') and not q.startswith('['):
                        questions.append({"question": q, "supported_by": [], "difficulty": "unknown"})
            result.questions = questions[:10] if questions else [{"question": "Could not parse LLM response", "supported_by": []}]
        except Exception as e:
            logger.warning(f"Failed to generate questions: {e}")
            result.questions = [{"question": f"Question generation failed: {str(e)}", "supported_by": []}]

        result.content_summary = f"Processed {result.num_pages} pages: {result.num_text_chunks} text, {result.num_tables} tables, {result.num_equations} equations, {result.num_images} images"

        # Cleanup
        await rag.finalize_storages()

        return result

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        import traceback
        traceback.print_exc()
        result.error = str(e)
        return result


def safe_print(text: str) -> None:
    """Print text safely, handling Unicode encoding issues."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace problematic characters with ASCII equivalents
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)


def print_result(result: ProcessingResult) -> None:
    """Print processing result in a formatted way."""
    safe_print("\n" + "=" * 70)
    safe_print(f"PDF: {result.title}")
    safe_print(f"Path: {result.pdf_path}")
    safe_print("=" * 70)

    if result.error:
        safe_print(f"\nERROR: {result.error}")
        return

    safe_print(f"\nContent Statistics:")
    safe_print(f"  - Pages: {result.num_pages}")
    safe_print(f"  - Text chunks: {result.num_text_chunks}")
    safe_print(f"  - Tables: {result.num_tables}")
    safe_print(f"  - Equations: {result.num_equations}")
    safe_print(f"  - Images: {result.num_images}")

    safe_print(f"\n{result.content_summary}")

    if result.questions:
        safe_print(f"\nGenerated Questions ({len(result.questions)}):")
        safe_print("-" * 40)
        for i, q in enumerate(result.questions, 1):
            if isinstance(q, dict):
                # New format with supported_by
                question_text = q.get("question", str(q))
                supported_by = q.get("supported_by", [])
                difficulty = q.get("difficulty", "")

                safe_print(f"  {i}. {question_text}")
                if supported_by:
                    safe_print(f"     [Supported by: {', '.join(supported_by)}]")
                if difficulty:
                    safe_print(f"     [Difficulty: {difficulty}]")
            else:
                # Simple string format
                safe_print(f"  {i}. {q}")

    if result.entities:
        safe_print(f"\nChunk References:")
        safe_print("-" * 40)
        for entity in result.entities:
            safe_print(f"  {entity}")

    safe_print("\n" + "=" * 70)


def list_pdfs(directory: str) -> list[str]:
    """List all PDF files in a directory."""
    pdf_dir = Path(directory)
    if not pdf_dir.exists():
        print(f"Directory not found: {directory}")
        return []

    pdfs = sorted([str(f) for f in pdf_dir.glob("*.pdf")])
    print(f"\nFound {len(pdfs)} PDF files in {directory}:")
    for i, pdf in enumerate(pdfs, 1):
        print(f"  {i}. {Path(pdf).name}")
    return pdfs


async def main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    if args.list_dir:
        list_pdfs(args.list_dir)
        return 0

    if not args.pdf_path:
        print("Error: PDF path required. Use --help for usage.")
        return 1

    pdf_path = args.pdf_path
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found: {pdf_path}")
        return 1

    # Setup working directory
    pdf_name = Path(pdf_path).stem
    working_dir = args.working_dir or f"./pdf_processing_{pdf_name}"

    # Clean previous run if requested
    if args.clean and os.path.exists(working_dir):
        logger.info(f"Cleaning previous run: {working_dir}")
        shutil.rmtree(working_dir)

    os.makedirs(working_dir, exist_ok=True)

    # Process PDF
    if args.simple:
        result = await process_pdf_simple(
            pdf_path=pdf_path,
            working_dir=working_dir,
            use_ollama=args.use_ollama,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    else:
        result = await process_pdf_with_raganything(
            pdf_path=pdf_path,
            working_dir=working_dir,
            use_ollama=args.use_ollama,
            api_key=args.api_key,
            base_url=args.base_url,
        )

    # Print result
    print_result(result)

    # Save result to JSON
    output_file = os.path.join(working_dir, f"{pdf_name}_questions.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "pdf_path": result.pdf_path,
            "title": result.title,
            "statistics": {
                "pages": result.num_pages,
                "text_chunks": result.num_text_chunks,
                "tables": result.num_tables,
                "equations": result.num_equations,
                "images": result.num_images,
            },
            "questions": result.questions,
            "entities": result.entities,
            "error": result.error,
        }, f, indent=2, ensure_ascii=False)

    # Save complete output with context and questions to readable file
    complete_output_file = os.path.join(working_dir, f"{pdf_name}_complete_output.txt")
    with open(complete_output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"PDF QUESTION GENERATOR OUTPUT\n")
        f.write(f"PDF: {result.title}\n")
        f.write(f"Path: {result.pdf_path}\n")
        f.write("=" * 80 + "\n\n")

        f.write("STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Pages: {result.num_pages}\n")
        f.write(f"Text chunks: {result.num_text_chunks}\n")
        f.write(f"Tables: {result.num_tables}\n")
        f.write(f"Equations: {result.num_equations}\n")
        f.write(f"Images: {result.num_images}\n\n")

        f.write("=" * 80 + "\n")
        f.write("CONTEXT CHUNKS (sent to LLM)\n")
        f.write("=" * 80 + "\n\n")
        f.write(result.context_chunks if result.context_chunks else "(no context)\n")
        f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("RAW LLM RESPONSE\n")
        f.write("=" * 80 + "\n\n")
        f.write(result.raw_llm_response if result.raw_llm_response else "(no response)\n")
        f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("GENERATED QUESTIONS\n")
        f.write("=" * 80 + "\n\n")
        for i, q in enumerate(result.questions, 1):
            if isinstance(q, dict):
                f.write(f"{i}. {q.get('question', str(q))}\n")
                if q.get('supported_by'):
                    f.write(f"   Supported by: {', '.join(q['supported_by'])}\n")
                if q.get('difficulty'):
                    f.write(f"   Difficulty: {q['difficulty']}\n")
            else:
                f.write(f"{i}. {q}\n")
            f.write("\n")

        if result.error:
            f.write("=" * 80 + "\n")
            f.write("ERROR\n")
            f.write("=" * 80 + "\n")
            f.write(f"{result.error}\n")

    safe_print(f"\nResults saved to:")
    safe_print(f"  - JSON: {output_file}")
    safe_print(f"  - Complete output: {complete_output_file}")

    return 0 if not result.error else 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process PDF with RAG-Anything and generate questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a single PDF with Ollama (default)
    python -m rag.ingestion.processors.pdf_question_generator C:/path/to/file.pdf

    # Process with OpenAI
    python -m rag.ingestion.processors.pdf_question_generator --api-key YOUR_KEY C:/path/to/file.pdf

    # List PDFs in a directory
    python -m rag.ingestion.processors.pdf_question_generator --list-dir C:/Users/rohan/Desktop/csd168

    # Use simple mode (MinerU + modal processors directly)
    python -m rag.ingestion.processors.pdf_question_generator --simple C:/path/to/file.pdf
        """,
    )

    parser.add_argument(
        "pdf_path",
        nargs="?",
        help="Path to the PDF file to process",
    )
    parser.add_argument(
        "--list-dir",
        help="List PDF files in directory (doesn't process)",
    )
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        default=True,
        help="Use Ollama for LLM/embeddings (default: True)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (disables Ollama)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL"),
        help="Optional API base URL",
    )
    parser.add_argument(
        "--working-dir", "-w",
        help="Working directory for RAG storage (default: ./pdf_processing_<name>)",
    )
    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        help="Clean previous processing results before running",
    )
    parser.add_argument(
        "--simple", "-s",
        action="store_true",
        help="Use simple mode (MinerU + modal processors) instead of full RAGAnything",
    )

    args = parser.parse_args()

    # If API key provided, disable Ollama
    if args.api_key and args.api_key != os.getenv("OPENAI_API_KEY"):
        args.use_ollama = False

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

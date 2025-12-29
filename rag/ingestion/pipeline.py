"""
Document ingestion pipeline for processing documents into MongoDB vector database.

Module: rag.ingestion.pipeline
==============================

This module provides the main document ingestion pipeline that:
- Discovers documents in a folder (PDF, DOCX, MD, audio, etc.)
- Converts and extracts text using Docling
- Transcribes audio files using Whisper ASR
- Chunks documents using HybridChunker
- Generates embeddings using OpenAI-compatible API
- Stores documents and chunks in MongoDB

Classes
-------
DocumentIngestionPipeline
    Main ingestion pipeline orchestrating all components.

    Methods:
        __init__(config, documents_folder, clean_before_ingest)
            Initialize pipeline with configuration.

        async initialize() -> None
            Initialize all components (store, chunker, embedder).

        async close() -> None
            Close all connections.

        async ingest_documents(verbose: bool = False) -> list[IngestionResult]
            Ingest all documents from the configured folder.

        async _ingest_single_document(file_path: str) -> IngestionResult
            Ingest a single document (internal).

    Internal Methods:
        _read_document(file_path) -> tuple[str, DoclingDocument | None]
            Read document content using Docling.

        _transcribe_audio(file_path) -> tuple[str, DoclingDocument | None]
            Transcribe audio using Whisper ASR.

        _find_document_files(folder) -> list[str]
            Find all supported documents in folder.

        _compute_content_hash(content) -> str
            Compute hash for incremental updates.

Supported File Formats
----------------------
- Text: .md, .markdown, .txt
- Documents: .pdf, .docx, .doc, .pptx, .ppt, .xlsx, .xls, .html
- Audio: .mp3, .wav, .m4a, .flac (requires FFmpeg + openai-whisper)

Functions
---------
create_pipeline(
    documents_folder: str = "documents",
    clean: bool = True,
    **config_kwargs
) -> DocumentIngestionPipeline
    Factory function to create configured pipeline.

Usage
-----
    from rag.ingestion.pipeline import create_pipeline

    # Create pipeline
    pipeline = create_pipeline(
        documents_folder="rag/documents",
        clean=True,
        chunk_size=1000,
        max_tokens=512
    )

    # Initialize and run
    await pipeline.initialize()
    results = await pipeline.ingest_documents(verbose=True)

    # Check results
    for r in results:
        print(f"{r.title}: {r.chunks_created} chunks")

    # Cleanup
    await pipeline.close()

CLI Usage
---------
    python -m rag.ingestion.pipeline --folder rag/documents --verbose
"""

import argparse  # noqa: I001
import asyncio
import glob
import hashlib
import logging
import os
import yaml


from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any



from rag.config.settings import load_settings
from rag.ingestion.chunkers.docling import create_chunker
from rag.ingestion.embedder import create_embedder
from rag.ingestion.models import (
    ChunkingConfig,
    IngestionConfig,
    IngestionResult,
)
from rag.storage.vector_store.mongo import MongoHybridStore

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents into MongoDB vector database."""

    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "documents",
        clean_before_ingest: bool = True,
    ):
        """
        Initialize ingestion pipeline.

        Args:
            config: Ingestion configuration
            documents_folder: Folder containing documents
            clean_before_ingest: Whether to clean existing data before ingestion
        """
        self.config = config
        self.documents_folder = documents_folder
        self.clean_before_ingest = clean_before_ingest

        # Load settings
        self.settings = load_settings()

        # Initialize components
        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
            max_tokens=config.max_tokens,
        )

        self.chunker = create_chunker(self.chunker_config)
        self.embedder = create_embedder()
        self.store = MongoHybridStore()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize MongoDB connection."""
        if self._initialized:
            return

        await self.store.initialize()
        self._initialized = True
        logger.info("Ingestion pipeline initialized")

    async def close(self) -> None:
        """Close MongoDB connection."""
        if self._initialized:
            await self.store.close()
            self._initialized = False
            logger.info("Ingestion pipeline closed")

    def _find_document_files(self) -> list[str]:
        """
        Find all supported document files in the documents folder.

        Returns:
            List of file paths
        """
        if not os.path.exists(self.documents_folder):
            logger.error(f"Documents folder not found: {self.documents_folder}")
            return []

        # Supported file patterns
        patterns = [
            "*.md",
            "*.markdown",
            "*.txt",  # Text formats
            "*.pdf",  # PDF
            "*.docx",
            "*.doc",  # Word
            "*.pptx",
            "*.ppt",  # PowerPoint
            "*.xlsx",
            "*.xls",  # Excel
            "*.html",
            "*.htm",  # HTML
            "*.mp3",
            "*.wav",
            "*.m4a",
            "*.flac",  # Audio formats
        ]
        files = []

        for pattern in patterns:
            files.extend(
                glob.glob(
                    os.path.join(self.documents_folder, "**", pattern), recursive=True
                )
            )

        return sorted(files)

    def _read_document(self, file_path: str) -> tuple[str, Any | None]:
        """
        Read document content from file.

        Args:
            file_path: Path to the document file

        Returns:
            Tuple of (content, docling_document)
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        # Audio formats - transcribe with Whisper ASR
        audio_formats = [".mp3", ".wav", ".m4a", ".flac"]
        if file_ext in audio_formats:
            return self._transcribe_audio(file_path)

        # Docling-supported formats
        docling_formats = [
            ".pdf",
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".xlsx",
            ".xls",
            ".html",
            ".htm",
            ".md",
            ".markdown",
        ]

        if file_ext in docling_formats:
            try:
                from docling.document_converter import DocumentConverter

                logger.info(
                    f"Converting {file_ext} file using Docling: {os.path.basename(file_path)}"
                )

                converter = DocumentConverter()
                result = converter.convert(file_path)

                markdown_content = result.document.export_to_markdown()
                logger.info(f"Successfully converted {os.path.basename(file_path)}")

                return (markdown_content, result.document)

            except Exception as e:
                logger.error(f"Failed to convert {file_path} with Docling: {e}")
                # Fall back to raw text
                try:
                    with open(file_path, encoding="utf-8") as f:
                        return (f.read(), None)
                except Exception:
                    return (
                        f"[Error: Could not read file {os.path.basename(file_path)}]",
                        None,
                    )

        # Text-based formats
        else:
            try:
                with open(file_path, encoding="utf-8") as f:
                    return (f.read(), None)
            except UnicodeDecodeError:
                with open(file_path, encoding="latin-1") as f:
                    return (f.read(), None)

    def _transcribe_audio(self, file_path: str) -> tuple[str, Any | None]:
        """
        Transcribe audio file using Whisper ASR via Docling.

        Args:
            file_path: Path to the audio file

        Returns:
            Tuple of (content, docling_document)
        """
        try:
            from docling.datamodel import asr_model_specs
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import AsrPipelineOptions
            from docling.document_converter import AudioFormatOption, DocumentConverter
            from docling.pipeline.asr_pipeline import AsrPipeline

            audio_path = Path(file_path).resolve()
            logger.info(f"Transcribing audio file: {audio_path.name}")

            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            pipeline_options = AsrPipelineOptions()
            pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

            converter = DocumentConverter(
                format_options={
                    InputFormat.AUDIO: AudioFormatOption(
                        pipeline_cls=AsrPipeline,
                        pipeline_options=pipeline_options,
                    )
                }
            )

            result = converter.convert(audio_path)
            markdown_content = result.document.export_to_markdown()
            logger.info(f"Successfully transcribed {os.path.basename(file_path)}")

            return (markdown_content, result.document)

        except Exception as e:
            logger.error(f"Failed to transcribe {file_path}: {e}")
            return (
                f"[Error: Could not transcribe audio file {os.path.basename(file_path)}]",
                None,
            )

    def _extract_title(self, content: str, file_path: str) -> str:
        """Extract title from document content or filename."""
        lines = content.split("\n")
        for line in lines[:10]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()

        return os.path.splitext(os.path.basename(file_path))[0]

    def _compute_file_hash(self, file_path: str) -> str:
        """
        Compute MD5 hash of a file for change detection.

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash string
        """
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _extract_document_metadata(
        self, content: str, file_path: str, content_hash: str | None = None
    ) -> dict[str, Any]:
        """Extract metadata from document content."""
        metadata = {
            "file_path": file_path,
            "file_size": len(content),
            "ingestion_date": datetime.now().isoformat(),
            "content_hash": content_hash or self._compute_file_hash(file_path),
        }

        # Try to extract YAML frontmatter
        if content.startswith("---"):
            try:
                end_marker = content.find("\n---\n", 4)
                if end_marker != -1:
                    frontmatter = content[4:end_marker]
                    yaml_metadata = yaml.safe_load(frontmatter)
                    if isinstance(yaml_metadata, dict):
                        metadata.update(yaml_metadata)
            except ImportError:
                logger.warning("PyYAML not installed, skipping frontmatter extraction")
            except Exception as e:
                logger.warning(f"Failed to parse frontmatter: {e}")

        lines = content.split("\n")
        metadata["line_count"] = len(lines)
        metadata["word_count"] = len(content.split())

        return metadata

    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        """
        Ingest a single document.

        Args:
            file_path: Path to the document file

        Returns:
            Ingestion result
        """
        start_time = datetime.now()

        # Read document
        document_content, docling_doc = self._read_document(file_path)
        document_title = self._extract_title(document_content, file_path)
        document_source = os.path.relpath(file_path, self.documents_folder)
        document_metadata = self._extract_document_metadata(document_content, file_path)

        logger.info(f"Processing document: {document_title}")

        # Chunk the document
        chunks = await self.chunker.chunk_document(
            content=document_content,
            title=document_title,
            source=document_source,
            metadata=document_metadata,
            docling_doc=docling_doc,
        )

        if not chunks:
            logger.warning(f"No chunks created for {document_title}")
            return IngestionResult(
                document_id="",
                title=document_title,
                chunks_created=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                errors=["No chunks created"],
            )

        logger.info(f"Created {len(chunks)} chunks")

        # Generate embeddings
        embedded_chunks = await self.embedder.embed_chunks(chunks)
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")

        # Save document to MongoDB
        document_id = await self.store.save_document(
            title=document_title,
            source=document_source,
            content=document_content,
            metadata=document_metadata,
        )

        # Save chunks to MongoDB
        await self.store.add(embedded_chunks, document_id)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return IngestionResult(
            document_id=document_id,
            title=document_title,
            chunks_created=len(chunks),
            processing_time_ms=processing_time,
            errors=[],
        )

    async def ingest_documents(
        self, progress_callback: Callable | None = None
    ) -> list[IngestionResult]:
        """
        Ingest all documents from the documents folder.

        Supports incremental indexing when clean_before_ingest is False:
        - New files are added
        - Modified files are re-indexed (old version deleted first)
        - Deleted files are removed from the database

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            List of ingestion results
        """
        await self.initialize()

        # Clean existing data if requested
        if self.clean_before_ingest:
            await self.store.clean_collections()

        # Find all supported document files
        document_files = self._find_document_files()

        if not document_files:
            logger.warning(
                f"No supported document files found in {self.documents_folder}"
            )
            return []

        logger.info(f"Found {len(document_files)} document files to process")

        results = []
        skipped_count = 0
        updated_count = 0
        new_count = 0
        deleted_count = 0

        # Build set of current file sources for tracking deletions
        current_sources = set()

        for i, file_path in enumerate(document_files):
            document_source = os.path.relpath(file_path, self.documents_folder)
            current_sources.add(document_source)

            try:
                # Compute hash for current file
                current_hash = self._compute_file_hash(file_path)

                # Check if document already exists (for incremental indexing)
                if not self.clean_before_ingest:
                    existing_hash = await self.store.get_document_hash(document_source)

                    if existing_hash == current_hash:
                        # File unchanged, skip
                        logger.info(
                            f"[SKIP] {i + 1}/{len(document_files)}: "
                            f"{document_source} (unchanged)"
                        )
                        skipped_count += 1
                        if progress_callback:
                            progress_callback(i + 1, len(document_files))
                        continue

                    if existing_hash is not None:
                        # File changed, delete old version first
                        logger.info(
                            f"[UPDATE] {i + 1}/{len(document_files)}: "
                            f"{document_source} (content changed)"
                        )
                        await self.store.delete_document_and_chunks(document_source)
                        updated_count += 1
                    else:
                        logger.info(
                            f"[NEW] {i + 1}/{len(document_files)}: {document_source}"
                        )
                        new_count += 1
                else:
                    logger.info(
                        f"Processing file {i + 1}/{len(document_files)}: {file_path}"
                    )

                result = await self._ingest_single_document(file_path)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(document_files))

            except Exception as e:
                logger.exception(f"Failed to process {file_path}: {e}")
                results.append(
                    IngestionResult(
                        document_id="",
                        title=os.path.basename(file_path),
                        chunks_created=0,
                        processing_time_ms=0,
                        errors=[str(e)],
                    )
                )

        # Handle deleted files (only in incremental mode)
        if not self.clean_before_ingest:
            existing_sources = await self.store.get_all_document_sources()
            for source in existing_sources:
                if source not in current_sources:
                    logger.info(f"[DELETE] Removing deleted document: {source}")
                    await self.store.delete_document_and_chunks(source)
                    deleted_count += 1

        # Log summary
        total_chunks = sum(r.chunks_created for r in results)
        total_errors = sum(len(r.errors) for r in results)

        if not self.clean_before_ingest:
            logger.info(
                f"Incremental ingestion complete: "
                f"{new_count} new, {updated_count} updated, "
                f"{skipped_count} unchanged, {deleted_count} deleted"
            )

        logger.info(
            f"Ingestion complete: {len(results)} documents processed, "
            f"{total_chunks} chunks created, {total_errors} errors"
        )

        return results


def create_pipeline(
    documents_folder: str = "documents",
    clean: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    max_tokens: int = 512,
) -> DocumentIngestionPipeline:
    """
    Factory function to create a configured ingestion pipeline.

    Args:
        documents_folder: Folder containing documents to ingest
        clean: Whether to clean existing data before ingestion
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        max_tokens: Maximum tokens per chunk for embeddings

    Returns:
        Configured DocumentIngestionPipeline instance
    """
    config = IngestionConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunk_size=chunk_size * 2,
        max_tokens=max_tokens,
    )
    return DocumentIngestionPipeline(
        config=config,
        documents_folder=documents_folder,
        clean_before_ingest=clean,
    )


async def run_ingestion_pipeline() -> None:
    """Main function for running ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into MongoDB vector database"
    )
    parser.add_argument(
        "--documents", "-d", default="documents", help="Documents folder path"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip cleaning existing data before ingestion",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for splitting documents",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=200, help="Chunk overlap size"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per chunk for embeddings",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create ingestion configuration
    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_chunk_size=args.chunk_size * 2,
        max_tokens=args.max_tokens,
    )

    # Create and run pipeline
    pipeline = DocumentIngestionPipeline(
        config=config,
        documents_folder=args.documents,
        clean_before_ingest=not args.no_clean,
    )

    def progress_callback(current: int, total: int) -> None:
        logger.info(f"Progress: {current}/{total} documents processed")

    try:
        start_time = datetime.now()

        results = await pipeline.ingest_documents(progress_callback)

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Log summary
        logger.info("=" * 50)
        logger.info("INGESTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Documents processed: {len(results)}")
        logger.info(f"Total chunks created: {sum(r.chunks_created for r in results)}")
        logger.info(f"Total errors: {sum(len(r.errors) for r in results)}")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info("")

        # Print individual results
        for result in results:
            status = "[OK]" if not result.errors else "[FAILED]"
            logger.info(f"{status} {result.title}: {result.chunks_created} chunks")

            if result.errors:
                for error in result.errors:
                    logger.error(f"  Error: {error}")

        # Log next steps
        logger.info("=" * 50)
        logger.info("NEXT STEPS")
        logger.info("=" * 50)
        logger.info("1. Create vector search index in Atlas UI:")
        logger.info("   - Index name: vector_index")
        logger.info("   - Collection: chunks")
        logger.info("   - Field: embedding")
        logger.info("   - Dimensions: 768 (for nomic-embed-text)")
        logger.info("2. Create text search index in Atlas UI:")
        logger.info("   - Index name: text_index")
        logger.info("   - Collection: chunks")
        logger.info("   - Field: content")

    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user")
    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        raise
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(run_ingestion_pipeline())

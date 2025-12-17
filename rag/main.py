#!/usr/bin/env python
"""
Main entry point for RAG system.

Usage:
    python -m rag.main              # Validate config and ingest documents
    python -m rag.main --validate   # Only validate config
    python -m rag.main --ingest     # Only ingest documents
    python -m rag.main --no-clean   # Ingest without cleaning existing data
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime

from rag.config.settings import load_settings, mask_credential
from rag.ingestion.models import IngestionConfig
from rag.ingestion.pipeline import DocumentIngestionPipeline


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger(__name__)


def validate_config() -> bool:
    """
    Validate configuration and display settings.

    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        print("=" * 60)
        print("RAG System - Configuration Validation")
        print("=" * 60)

        # Load settings
        print("\n[1/4] Loading settings...")
        settings = load_settings()
        print("  [OK] Settings loaded successfully")

        # Validate MongoDB configuration
        print("\n[2/4] Validating MongoDB configuration...")
        print(f"  MongoDB URI: {mask_credential(settings.mongodb_uri)}")
        print(f"  Database: {settings.mongodb_database}")
        print(f"  Documents Collection: {settings.mongodb_collection_documents}")
        print(f"  Chunks Collection: {settings.mongodb_collection_chunks}")
        print(f"  Vector Index: {settings.mongodb_vector_index}")
        print(f"  Text Index: {settings.mongodb_text_index}")
        print("  [OK] MongoDB configuration present")

        # Validate LLM configuration
        print("\n[3/4] Validating LLM configuration...")
        print(f"  Provider: {settings.llm_provider}")
        print(f"  Model: {settings.llm_model}")
        print(f"  Base URL: {settings.llm_base_url}")
        print(f"  API Key: {mask_credential(settings.llm_api_key)}")
        print("  [OK] LLM configuration present")

        # Validate Embedding configuration
        print("\n[4/4] Validating Embedding configuration...")
        print(f"  Provider: {settings.embedding_provider}")
        print(f"  Model: {settings.embedding_model}")
        print(f"  Base URL: {settings.embedding_base_url}")
        print(f"  Dimension: {settings.embedding_dimension}")
        print(f"  API Key: {mask_credential(settings.embedding_api_key)}")
        print("  [OK] Embedding configuration present")

        # Success summary
        print("\n" + "=" * 60)
        print("[OK] ALL CONFIGURATION CHECKS PASSED")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n[ERROR] Configuration validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_ingestion(
    documents_folder: str = "documents",
    clean_before_ingest: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    max_tokens: int = 512,
) -> bool:
    """
    Run document ingestion pipeline.

    Args:
        documents_folder: Path to documents folder
        clean_before_ingest: Whether to clean existing data
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        max_tokens: Maximum tokens per chunk

    Returns:
        True if ingestion succeeded, False otherwise
    """
    print("\n" + "=" * 60)
    print("RAG System - Document Ingestion")
    print("=" * 60)

    # Create ingestion configuration
    config = IngestionConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunk_size=chunk_size * 2,
        max_tokens=max_tokens,
    )

    # Create pipeline
    pipeline = DocumentIngestionPipeline(
        config=config,
        documents_folder=documents_folder,
        clean_before_ingest=clean_before_ingest,
    )

    def progress_callback(current: int, total: int) -> None:
        print(f"  Progress: {current}/{total} documents processed")

    try:
        start_time = datetime.now()

        print(f"\nIngesting documents from: {documents_folder}")
        print(f"Clean before ingest: {clean_before_ingest}")
        print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        print(f"Max tokens: {max_tokens}")
        print()

        results = await pipeline.ingest_documents(progress_callback)

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Print summary
        print("\n" + "-" * 40)
        print("INGESTION SUMMARY")
        print("-" * 40)
        print(f"Documents processed: {len(results)}")
        print(f"Total chunks created: {sum(r.chunks_created for r in results)}")
        print(f"Total errors: {sum(len(r.errors) for r in results)}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print()

        # Print individual results
        for result in results:
            status = "[OK]" if not result.errors else "[FAILED]"
            print(f"  {status} {result.title}: {result.chunks_created} chunks")
            if result.errors:
                for error in result.errors:
                    print(f"       Error: {error}")

        # Print next steps
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("1. Create vector search index in MongoDB Atlas UI:")
        print("   - Index name: vector_index")
        print("   - Collection: chunks")
        print("   - Field: embedding")
        print("   - Dimensions: 768 (for nomic-embed-text)")
        print()
        print("2. Create text search index in MongoDB Atlas UI:")
        print("   - Index name: text_index")
        print("   - Collection: chunks")
        print("   - Field: content")

        return sum(len(r.errors) for r in results) == 0

    except Exception as e:
        print(f"\n[ERROR] Ingestion failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        await pipeline.close()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG System - Validate config and ingest documents"
    )
    parser.add_argument(
        "--validate",
        "-v",
        action="store_true",
        help="Only validate configuration (skip ingestion)",
    )
    parser.add_argument(
        "--ingest",
        "-i",
        action="store_true",
        help="Only run ingestion (skip validation)",
    )
    parser.add_argument(
        "--documents",
        "-d",
        default="documents",
        help="Documents folder path (default: documents)",
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
        help="Chunk size in characters (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters (default: 200)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per chunk (default: 512)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Determine what to run
    run_validation = not args.ingest or args.validate
    run_ingest = not args.validate or args.ingest

    # If neither flag specified, run both
    if not args.validate and not args.ingest:
        run_validation = True
        run_ingest = True

    success = True

    # Validate configuration
    if run_validation:
        if not validate_config():
            print(
                "\n[ERROR] Configuration validation failed. Fix errors before proceeding."
            )
            return 1

    # Run ingestion
    if run_ingest:
        ingest_success = await run_ingestion(
            documents_folder=args.documents,
            clean_before_ingest=not args.no_clean,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_tokens=args.max_tokens,
        )
        success = success and ingest_success

    print("\n" + "=" * 60)
    if success:
        print("[OK] RAG System setup complete!")
    else:
        print("[WARNING] RAG System setup completed with errors")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

#!/usr/bin/env python
"""
Main entry point for RAG system.

Usage:
    python -m rag.main              # Validate config and ingest documents
    python -m rag.main --validate   # Only validate config
    python -m rag.main --ingest     # Only ingest documents
    python -m rag.main --no-clean   # Ingest without cleaning existing data
"""

import asyncio
import logging
import sys

from rag.config.settings import load_settings, mask_credential
from rag.ingestion.pipeline import run_ingestion_pipeline

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def validate_config() -> bool:
    """
    Validate configuration and display settings.

    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        logger.info("=" * 60)
        logger.info("RAG System - Configuration Validation")
        logger.info("=" * 60)

        # Load settings
        logger.info("[1/4] Loading settings...")
        settings = load_settings()
        logger.info("  [OK] Settings loaded successfully")

        # Validate MongoDB configuration
        logger.info("[2/4] Validating MongoDB configuration...")
        logger.info(f"  MongoDB URI: {mask_credential(settings.mongodb_uri)}")
        logger.info(f"  Database: {settings.mongodb_database}")
        logger.info(f"  Documents Collection: {settings.mongodb_collection_documents}")
        logger.info(f"  Chunks Collection: {settings.mongodb_collection_chunks}")
        logger.info(f"  Vector Index: {settings.mongodb_vector_index}")
        logger.info(f"  Text Index: {settings.mongodb_text_index}")
        logger.info("  [OK] MongoDB configuration present")

        # Validate LLM configuration
        logger.info("[3/4] Validating LLM configuration...")
        logger.info(f"  Provider: {settings.llm_provider}")
        logger.info(f"  Model: {settings.llm_model}")
        logger.info(f"  Base URL: {settings.llm_base_url}")
        logger.info(f"  API Key: {mask_credential(settings.llm_api_key)}")
        logger.info("  [OK] LLM configuration present")

        # Validate Embedding configuration
        logger.info("[4/4] Validating Embedding configuration...")
        logger.info(f"  Provider: {settings.embedding_provider}")
        logger.info(f"  Model: {settings.embedding_model}")
        logger.info(f"  Base URL: {settings.embedding_base_url}")
        logger.info(f"  Dimension: {settings.embedding_dimension}")
        logger.info(f"  API Key: {mask_credential(settings.embedding_api_key)}")
        logger.info("  [OK] Embedding configuration present")

        # Success summary
        logger.info("=" * 60)
        logger.info("[OK] ALL CONFIGURATION CHECKS PASSED")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        logger.exception("Full traceback:")
        return False


async def main() -> int:
    """Main entry point."""
    import argparse

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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--run-agent",
        "-r",
        action="store_true",
        help="Run the RAG agent conversation loop after ingestion/validation",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Determine what to run
    run_validation: bool = True if args.validate else False
    run_ingestion: bool = True if args.ingest else False
    run_agent: bool = True if args.run_agent else False

    # Validate configuration
    if run_validation:
        if not validate_config():
            logger.error(
                "Configuration validation failed. Fix errors before proceeding."
            )
            return 1
        logger.info("=" * 60)
        logger.info("[OK] RAG System setup complete!")
        logger.info("=" * 60)
        if not run_ingestion and not run_agent:
            return 0

    # Ingest documents
    if run_ingestion:
        # Modify sys.argv for run_ingestion_pipeline's argparse
        ingestion_args = [
            "rag.main",
            "--documents",
            args.documents,
            "--chunk-size",
            str(args.chunk_size),
            "--chunk-overlap",
            str(args.chunk_overlap),
            "--max-tokens",
            str(args.max_tokens),
        ]
        if args.no_clean:
            ingestion_args.append("--no-clean")
        if args.verbose:
            ingestion_args.append("--verbose")
        sys.argv = ingestion_args
        await run_ingestion_pipeline()

    # Run the RAG agent conversation loop
    if run_agent:
        from rag.agent.agent_main import agent_main
        await agent_main()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

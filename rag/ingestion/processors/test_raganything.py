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
Test script for RAG-Anything modal processors.

This script tests the RAG-Anything library's modal processors
(ImageModalProcessor, TableModalProcessor, EquationModalProcessor)
to verify they work correctly with our setup.

Usage:
    python -m rag.ingestion.processors.test_raganything
    python -m rag.ingestion.processors.test_raganything --api-key YOUR_KEY
    python -m rag.ingestion.processors.test_raganything --use-ollama
    python -m rag.ingestion.processors.test_raganything --use-ollama --use-mongodb
"""

import argparse
import asyncio
import logging
import os
import sys

from rag.ingestion.processors.lightrag_utils import (
    LightRAGConfig,
    initialize_lightrag,
    test_modal_processor,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)


# =============================================================================
# Test Data Constants
# =============================================================================

TABLE_TEST_CONTENT = {
    "table_body": """
| Model | Accuracy | F1-Score | Latency |
|-------|----------|----------|---------|
| GPT-4 | 95.2% | 0.94 | 120ms |
| Claude | 94.8% | 0.93 | 100ms |
| Llama | 89.5% | 0.88 | 50ms |
""",
    "table_caption": ["Performance Comparison of LLM Models"],
    "table_footnote": ["Benchmarked on MMLU dataset, 2024"],
}

EQUATION_TEST_CONTENT = {
    "text": r"L(\theta) = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(p_i) + (1-y_i)\log(1-p_i)]",
    "text_format": "LaTeX",
}

IMAGE_TEST_CONTENT = {
    "img_path": "",  # Empty for caption-only test
    "image_caption": ["Figure 1: Architecture diagram showing the RAG pipeline"],
    "image_footnote": ["Components include embedder, vector store, and LLM"],
}


async def run_all_tests(
    api_key: str | None,
    base_url: str | None,
    use_ollama: bool,
    use_mongodb: bool = False,
    mongo_uri: str | None = None,
    mongo_database: str = "test_raganything",
) -> bool:
    """Run all modal processor tests.

    Args:
        api_key: OpenAI API key
        base_url: OpenAI base URL
        use_ollama: Use Ollama for LLM/embeddings
        use_mongodb: Use MongoDB for storage
        mongo_uri: MongoDB connection URI
        mongo_database: MongoDB database name

    Returns:
        True if all tests passed
    """
    from raganything.modalprocessors import (
        TableModalProcessor,
        EquationModalProcessor,
        ImageModalProcessor,
    )

    # Initialize LightRAG
    logger.info("Initializing LightRAG...")
    config = LightRAGConfig(
        working_dir="./test_raganything_storage",
        use_ollama=use_ollama,
        api_key=api_key,
        base_url=base_url,
        use_mongodb=use_mongodb,
        mongo_uri=mongo_uri,
        mongo_database=mongo_database,
    )
    lightrag, llm_func, vision_func = await initialize_lightrag(config)

    # Define test cases
    test_cases = [
        {
            "processor_class": TableModalProcessor,
            "caption_func": llm_func,
            "test_content": TABLE_TEST_CONTENT,
            "content_type": "table",
            "file_path": "benchmark_report.pdf",
            "entity_name": "LLM Performance Table",
            "processor_name": "TableModalProcessor",
        },
        {
            "processor_class": EquationModalProcessor,
            "caption_func": llm_func,
            "test_content": EQUATION_TEST_CONTENT,
            "content_type": "equation",
            "file_path": "ml_paper.pdf",
            "entity_name": "Binary Cross-Entropy Loss",
            "processor_name": "EquationModalProcessor",
        },
        {
            "processor_class": ImageModalProcessor,
            "caption_func": vision_func,
            "test_content": IMAGE_TEST_CONTENT,
            "content_type": "image",
            "file_path": "architecture.pdf",
            "entity_name": "RAG Architecture Diagram",
            "processor_name": "ImageModalProcessor",
        },
    ]

    # Run tests
    results = {}
    for test_case in test_cases:
        name = test_case["processor_name"]
        results[name] = await test_modal_processor(
            processor_class=test_case["processor_class"],
            lightrag=lightrag,
            caption_func=test_case["caption_func"],
            test_content=test_case["test_content"],
            content_type=test_case["content_type"],
            file_path=test_case["file_path"],
            entity_name=test_case["entity_name"],
            processor_name=name,
        )

    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {name}: {status}")

    total = len(results)
    passed_count = sum(results.values())
    logger.info(f"\nTotal: {passed_count}/{total} passed")

    return all(results.values())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test RAG-Anything modal processors"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (default: OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL"),
        help="OpenAI base URL (optional)",
    )
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Use Ollama instead of OpenAI",
    )
    parser.add_argument(
        "--use-mongodb",
        action="store_true",
        help="Use MongoDB for vector/KV storage instead of file-based",
    )
    parser.add_argument(
        "--mongo-uri",
        default=os.getenv("MONGODB_URI"),
        help="MongoDB connection URI (default: MONGODB_URI env var)",
    )
    parser.add_argument(
        "--mongo-database",
        default="test_raganything",
        help="MongoDB database name (default: test_raganything)",
    )

    args = parser.parse_args()

    if not args.use_ollama and not args.api_key:
        logger.warning("No API key provided. Using --use-ollama for local testing.")
        args.use_ollama = True

    success = asyncio.run(
        run_all_tests(
            args.api_key,
            args.base_url,
            args.use_ollama,
            args.use_mongodb,
            args.mongo_uri,
            args.mongo_database,
        )
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

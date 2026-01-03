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
"""

import argparse
import asyncio
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_ollama_funcs():
    """Get LLM and vision functions using Ollama."""
    import httpx

    def ollama_complete_sync(
        prompt: str,
        system_prompt: str | None = None,
        model: str = "llama3.1:8b",
        **kwargs,
    ) -> str:
        """Call Ollama for text completion (sync)."""
        with httpx.Client(timeout=120.0) as client:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.post(
                "http://localhost:11434/v1/chat/completions",
                json={"model": model, "messages": messages},
                headers={"Authorization": "Bearer ollama"},
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    def llm_model_func(
        prompt, system_prompt=None, history_messages=None, **kwargs
    ) -> str:
        """Synchronous Ollama LLM function."""
        return ollama_complete_sync(prompt, system_prompt, "llama3.1:8b")

    def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        **kwargs,
    ) -> str:
        """Vision model function (falls back to text for Ollama)."""
        # Ollama's llava can handle images but for simplicity we use text
        if image_data:
            prompt = f"[Image provided - base64 data length: {len(image_data)}]\n{prompt}"
        return llm_model_func(prompt, system_prompt)

    return llm_model_func, vision_model_func


def get_openai_funcs(api_key: str, base_url: str | None = None):
    """Get LLM and vision functions using OpenAI API."""
    from lightrag.llm.openai import openai_complete_if_cache

    def llm_model_func(
        prompt, system_prompt=None, history_messages=None, **kwargs
    ) -> str:
        """OpenAI LLM function."""
        if history_messages is None:
            history_messages = []
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ) -> str:
        """OpenAI Vision function."""
        if history_messages is None:
            history_messages = []

        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                messages=[
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
                            },
                        ],
                    },
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    return llm_model_func, vision_model_func


async def test_table_processor(lightrag, llm_func):
    """Test the TableModalProcessor."""
    from raganything.modalprocessors import TableModalProcessor

    logger.info("=" * 60)
    logger.info("Testing TableModalProcessor")
    logger.info("=" * 60)

    processor = TableModalProcessor(lightrag=lightrag, modal_caption_func=llm_func)

    table_content = {
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

    try:
        result = await processor.process_multimodal_content(
            modal_content=table_content,
            content_type="table",
            file_path="benchmark_report.pdf",
            entity_name="LLM Performance Table",
        )

        # Handle variable return values (2 or 3)
        if isinstance(result, tuple):
            description = result[0] if len(result) > 0 else ""
            entity_info = result[1] if len(result) > 1 else {}
        else:
            description = str(result)
            entity_info = {}

        desc_preview = description[:200] if description else "(empty)"
        logger.info(f"Description: {desc_preview}...")
        logger.info(f"Entity Info: {entity_info}")
        logger.info("TableModalProcessor: SUCCESS")
        return True
    except Exception as e:
        logger.error(f"TableModalProcessor FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_equation_processor(lightrag, llm_func):
    """Test the EquationModalProcessor."""
    from raganything.modalprocessors import EquationModalProcessor

    logger.info("=" * 60)
    logger.info("Testing EquationModalProcessor")
    logger.info("=" * 60)

    processor = EquationModalProcessor(lightrag=lightrag, modal_caption_func=llm_func)

    equation_content = {
        "text": r"L(\theta) = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(p_i) + (1-y_i)\log(1-p_i)]",
        "text_format": "LaTeX",
    }

    try:
        result = await processor.process_multimodal_content(
            modal_content=equation_content,
            content_type="equation",
            file_path="ml_paper.pdf",
            entity_name="Binary Cross-Entropy Loss",
        )

        # Handle variable return values (2 or 3)
        if isinstance(result, tuple):
            description = result[0] if len(result) > 0 else ""
            entity_info = result[1] if len(result) > 1 else {}
        else:
            description = str(result)
            entity_info = {}

        desc_preview = description[:200] if description else "(empty)"
        logger.info(f"Description: {desc_preview}...")
        logger.info(f"Entity Info: {entity_info}")
        logger.info("EquationModalProcessor: SUCCESS")
        return True
    except Exception as e:
        logger.error(f"EquationModalProcessor FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_image_processor(lightrag, vision_func):
    """Test the ImageModalProcessor (without actual image)."""
    from raganything.modalprocessors import ImageModalProcessor

    logger.info("=" * 60)
    logger.info("Testing ImageModalProcessor")
    logger.info("=" * 60)

    processor = ImageModalProcessor(lightrag=lightrag, modal_caption_func=vision_func)

    # Test with caption only (no actual image file)
    image_content = {
        "img_path": "",  # No actual image
        "image_caption": ["Figure 1: Architecture diagram showing the RAG pipeline"],
        "image_footnote": ["Components include embedder, vector store, and LLM"],
    }

    try:
        result = await processor.process_multimodal_content(
            modal_content=image_content,
            content_type="image",
            file_path="architecture.pdf",
            entity_name="RAG Architecture Diagram",
        )

        # Handle variable return values (2 or 3)
        if isinstance(result, tuple):
            description = result[0] if len(result) > 0 else ""
            entity_info = result[1] if len(result) > 1 else {}
        else:
            description = str(result)
            entity_info = {}

        desc_preview = description[:200] if description else "(empty)"
        logger.info(f"Description: {desc_preview}...")
        logger.info(f"Entity Info: {entity_info}")
        logger.info("ImageModalProcessor: SUCCESS")
        return True
    except Exception as e:
        logger.error(f"ImageModalProcessor FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def initialize_lightrag(api_key: str | None, base_url: str | None, use_ollama: bool):
    """Initialize LightRAG instance."""
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.utils import EmbeddingFunc

    working_dir = "./test_raganything_storage"
    os.makedirs(working_dir, exist_ok=True)

    if use_ollama:
        # Use Ollama for embeddings
        import httpx
        import numpy as np

        async def ollama_embed(texts: list[str]) -> np.ndarray:
            async with httpx.AsyncClient(timeout=60.0) as client:
                embeddings = []
                for text in texts:
                    response = await client.post(
                        "http://localhost:11434/api/embeddings",
                        json={"model": "nomic-embed-text:latest", "prompt": text},
                    )
                    response.raise_for_status()
                    embeddings.append(response.json()["embedding"])
                return np.array(embeddings)

        embedding_func = EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=ollama_embed,
        )
        llm_func, vision_func = get_ollama_funcs()
    else:
        # Use OpenAI
        from lightrag.llm.openai import openai_embed

        embedding_func = EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model="text-embedding-3-large",
                api_key=api_key,
                base_url=base_url,
            ),
        )
        llm_func, vision_func = get_openai_funcs(api_key, base_url)

    # Create LightRAG instance
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        embedding_func=embedding_func,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag, llm_func, vision_func


async def main_async(api_key: str | None, base_url: str | None, use_ollama: bool):
    """Main async test function."""
    logger.info("Initializing LightRAG...")
    lightrag, llm_func, vision_func = await initialize_lightrag(
        api_key, base_url, use_ollama
    )

    results = {}

    # Test each processor
    results["table"] = await test_table_processor(lightrag, llm_func)
    results["equation"] = await test_equation_processor(lightrag, llm_func)
    results["image"] = await test_image_processor(lightrag, vision_func)

    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {name.capitalize()}Processor: {status}")

    total = len(results)
    passed = sum(results.values())
    logger.info(f"\nTotal: {passed}/{total} passed")

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

    args = parser.parse_args()

    if not args.use_ollama and not args.api_key:
        logger.warning("No API key provided. Using --use-ollama for local testing.")
        args.use_ollama = True

    success = asyncio.run(main_async(args.api_key, args.base_url, args.use_ollama))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

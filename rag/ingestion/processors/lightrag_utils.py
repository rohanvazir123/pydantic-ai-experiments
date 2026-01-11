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
Common utilities for LightRAG and RAG-Anything integration.

This module provides reusable functions for:
- LLM function factories (Ollama, OpenAI)
- Embedding function factories
- LightRAG initialization
- Modal processor testing utilities
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# LLM Function Factories
# =============================================================================


def get_ollama_llm_funcs(
    llm_model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    timeout: float = 120.0,
) -> tuple[Callable, Callable]:
    """Get LLM and vision functions using Ollama.

    Args:
        llm_model: Ollama model name for text completion
        base_url: Ollama server URL
        timeout: Request timeout in seconds

    Returns:
        Tuple of (llm_model_func, vision_model_func)
    """
    import httpx

    def ollama_complete_sync(
        prompt: str,
        system_prompt: str | None = None,
        model: str = llm_model,
        **kwargs,
    ) -> str:
        """Call Ollama for text completion (sync)."""
        with httpx.Client(timeout=timeout) as client:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.post(
                f"{base_url}/v1/chat/completions",
                json={"model": model, "messages": messages},
                headers={"Authorization": "Bearer ollama"},
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    def llm_model_func(
        prompt, system_prompt=None, history_messages=None, **kwargs
    ) -> str:
        """Synchronous Ollama LLM function."""
        return ollama_complete_sync(prompt, system_prompt, llm_model)

    def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        **kwargs,
    ) -> str:
        """Vision model function (falls back to text for Ollama)."""
        if image_data:
            prompt = f"[Image provided - base64 data length: {len(image_data)}]\n{prompt}"
        return llm_model_func(prompt, system_prompt)

    return llm_model_func, vision_model_func


async def get_ollama_llm_funcs_async(
    llm_model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    timeout: float = 120.0,
) -> tuple[Callable, Callable]:
    """Get async LLM and vision functions using Ollama.

    Use these for raganything modal processors that expect async functions.

    Args:
        llm_model: Ollama model name for text completion
        base_url: Ollama server URL
        timeout: Request timeout in seconds

    Returns:
        Tuple of (async_llm_model_func, async_vision_model_func)
    """
    import httpx

    async def ollama_complete_async(
        prompt: str,
        system_prompt: str | None = None,
        model: str = llm_model,
        **kwargs,
    ) -> str:
        """Call Ollama for text completion (async)."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json={"model": model, "messages": messages},
                headers={"Authorization": "Bearer ollama"},
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def async_llm_model_func(
        prompt, system_prompt=None, history_messages=None, **kwargs
    ) -> str:
        """Async Ollama LLM function for modal processors."""
        return await ollama_complete_async(prompt, system_prompt, llm_model)

    async def async_vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        **kwargs,
    ) -> str:
        """Async vision model function (falls back to text for Ollama)."""
        if image_data:
            prompt = f"[Image provided - base64 data length: {len(image_data)}]\n{prompt}"
        return await async_llm_model_func(prompt, system_prompt)

    return async_llm_model_func, async_vision_model_func


def get_openai_llm_funcs(
    api_key: str,
    base_url: str | None = None,
    llm_model: str = "gpt-4o-mini",
    vision_model: str = "gpt-4o",
) -> tuple[Callable, Callable]:
    """Get LLM and vision functions using OpenAI API.

    Args:
        api_key: OpenAI API key
        base_url: Optional base URL for API
        llm_model: Model for text completion
        vision_model: Model for vision tasks

    Returns:
        Tuple of (llm_model_func, vision_model_func)
    """
    from lightrag.llm.openai import openai_complete_if_cache

    def llm_model_func(
        prompt, system_prompt=None, history_messages=None, **kwargs
    ) -> str:
        """OpenAI LLM function."""
        if history_messages is None:
            history_messages = []
        return openai_complete_if_cache(
            llm_model,
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
                vision_model,
                "",
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        elif image_data:
            return openai_complete_if_cache(
                vision_model,
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


# =============================================================================
# Embedding Function Factories
# =============================================================================


def get_ollama_embedding_func(
    model: str = "nomic-embed-text:latest",
    base_url: str = "http://localhost:11434",
    embedding_dim: int = 768,
    max_token_size: int = 8192,
    timeout: float = 60.0,
):
    """Get embedding function using Ollama.

    Args:
        model: Ollama embedding model name
        base_url: Ollama server URL
        embedding_dim: Embedding dimension
        max_token_size: Maximum token size
        timeout: Request timeout in seconds

    Returns:
        EmbeddingFunc instance
    """
    import httpx
    from lightrag.utils import EmbeddingFunc

    async def ollama_embed(texts: list[str]) -> np.ndarray:
        async with httpx.AsyncClient(timeout=timeout) as client:
            embeddings = []
            for text in texts:
                response = await client.post(
                    f"{base_url}/api/embeddings",
                    json={"model": model, "prompt": text},
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
            return np.array(embeddings)

    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        func=ollama_embed,
    )


def get_openai_embedding_func(
    api_key: str,
    base_url: str | None = None,
    model: str = "text-embedding-3-large",
    embedding_dim: int = 3072,
    max_token_size: int = 8192,
):
    """Get embedding function using OpenAI.

    Args:
        api_key: OpenAI API key
        base_url: Optional base URL
        model: OpenAI embedding model name
        embedding_dim: Embedding dimension
        max_token_size: Maximum token size

    Returns:
        EmbeddingFunc instance
    """
    from lightrag.llm.openai import openai_embed
    from lightrag.utils import EmbeddingFunc

    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        func=lambda texts: openai_embed(
            texts,
            model=model,
            api_key=api_key,
            base_url=base_url,
        ),
    )


# =============================================================================
# LightRAG Initialization
# =============================================================================


@dataclass
class LightRAGConfig:
    """Configuration for LightRAG initialization."""

    working_dir: str = "./lightrag_storage"
    use_ollama: bool = True
    api_key: str | None = None
    base_url: str | None = None
    use_mongodb: bool = False
    mongo_uri: str | None = None
    mongo_database: str = "lightrag"
    ollama_llm_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text:latest"
    ollama_base_url: str = "http://localhost:11434"


async def initialize_lightrag(
    config: LightRAGConfig | None = None,
    **kwargs,
) -> tuple[Any, Callable, Callable]:
    """Initialize LightRAG instance with LLM and embedding functions.

    Args:
        config: LightRAGConfig instance (or pass kwargs directly)
        **kwargs: Override config values

    Returns:
        Tuple of (lightrag_instance, llm_func, vision_func)
    """
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import initialize_pipeline_status

    # Build config
    if config is None:
        config = LightRAGConfig(**kwargs)
    else:
        # Override with any kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    os.makedirs(config.working_dir, exist_ok=True)

    # Get LLM and embedding functions
    if config.use_ollama:
        embedding_func = get_ollama_embedding_func(
            model=config.ollama_embed_model,
            base_url=config.ollama_base_url,
        )
        llm_func, vision_func = get_ollama_llm_funcs(
            llm_model=config.ollama_llm_model,
            base_url=config.ollama_base_url,
        )
    else:
        if not config.api_key:
            raise ValueError("api_key required when not using Ollama")
        embedding_func = get_openai_embedding_func(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        llm_func, vision_func = get_openai_llm_funcs(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    # Configure LightRAG
    lightrag_kwargs = {
        "working_dir": config.working_dir,
        "llm_model_func": llm_func,
        "embedding_func": embedding_func,
    }

    # Configure MongoDB storage if requested
    if config.use_mongodb:
        mongo_uri = config.mongo_uri or os.getenv(
            "MONGODB_URI", "mongodb://localhost:27017/"
        )
        os.environ["MONGO_URI"] = mongo_uri
        os.environ["MONGO_DATABASE"] = config.mongo_database

        lightrag_kwargs["kv_storage"] = "MongoKVStorage"
        lightrag_kwargs["vector_storage"] = "MongoVectorDBStorage"
        logger.info(f"Using MongoDB storage: {mongo_uri} / {config.mongo_database}")
    else:
        logger.info(f"Using file-based storage: {config.working_dir}")

    # Create and initialize LightRAG
    rag = LightRAG(**lightrag_kwargs)
    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag, llm_func, vision_func


# =============================================================================
# Modal Processor Testing Utilities
# =============================================================================


def parse_processor_result(result: Any) -> tuple[str, dict]:
    """Parse modal processor result into description and entity_info.

    Args:
        result: Result from process_multimodal_content

    Returns:
        Tuple of (description, entity_info)
    """
    if isinstance(result, tuple):
        description = result[0] if len(result) > 0 else ""
        entity_info = result[1] if len(result) > 1 else {}
    else:
        description = str(result)
        entity_info = {}
    return description, entity_info


async def test_modal_processor(
    processor_class,
    lightrag,
    caption_func: Callable,
    test_content: dict,
    content_type: str,
    file_path: str,
    entity_name: str,
    processor_name: str | None = None,
) -> bool:
    """Generic test function for modal processors.

    Args:
        processor_class: The processor class to test
        lightrag: LightRAG instance
        caption_func: LLM or vision function for captions
        test_content: Content dict to process
        content_type: Type of content ("table", "equation", "image")
        file_path: Simulated file path
        entity_name: Name for the entity
        processor_name: Display name (default: class name)

    Returns:
        True if test passed, False otherwise
    """
    processor_name = processor_name or processor_class.__name__

    logger.info("=" * 60)
    logger.info(f"Testing {processor_name}")
    logger.info("=" * 60)

    try:
        processor = processor_class(lightrag=lightrag, modal_caption_func=caption_func)

        result = await processor.process_multimodal_content(
            modal_content=test_content,
            content_type=content_type,
            file_path=file_path,
            entity_name=entity_name,
        )

        description, entity_info = parse_processor_result(result)

        desc_preview = str(description)[:200] if description else "(empty)"
        logger.info(f"Description: {desc_preview}...")
        logger.info(f"Entity Info: {entity_info}")
        logger.info(f"{processor_name}: SUCCESS")
        return True

    except Exception as e:
        logger.error(f"{processor_name} FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Utility Functions
# =============================================================================


def check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is available.

    Args:
        base_url: Ollama server URL

    Returns:
        True if Ollama is available
    """
    import httpx

    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration.

    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

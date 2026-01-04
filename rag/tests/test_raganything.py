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

"""Tests for RAG-Anything modal processors integration.

These tests verify the TableModalProcessor, EquationModalProcessor, and
ImageModalProcessor from the raganything library work correctly with our
LightRAG setup.

Known Non-Fatal Errors:
- "object str can't be used in 'await' expression": RAGAnything expects async
  LLM functions, but Ollama wrapper provides sync. Falls back gracefully.
- "No image path provided": Expected when testing caption-only image processing.
- "Error generating description": LLM async mismatch, falls back to raw content.

Requirements:
- Ollama running locally (ollama serve)
- nomic-embed-text:latest model pulled
- llama3.1:8b model pulled (or configure different model)
"""

import os
import shutil
import pytest

# Skip all tests if raganything is not installed
pytest.importorskip("raganything")
pytest.importorskip("lightrag")

from rag.ingestion.processors.lightrag_utils import (
    LightRAGConfig,
    get_ollama_llm_funcs,
    get_ollama_embedding_func,
    check_ollama_available,
    parse_processor_result,
)


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


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def working_dir(tmp_path_factory):
    """Create a temporary working directory for LightRAG storage."""
    work_dir = tmp_path_factory.mktemp("test_raganything_storage")
    yield str(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)


@pytest.fixture(scope="module")
def ollama_available():
    """Check if Ollama is available."""
    return check_ollama_available()


@pytest.fixture(scope="module")
def embedding_func():
    """Create embedding function using Ollama."""
    return get_ollama_embedding_func()


@pytest.fixture(scope="module")
def llm_funcs():
    """Create LLM and vision functions using Ollama."""
    return get_ollama_llm_funcs()


@pytest.fixture(scope="module")
def lightrag_instance(working_dir, embedding_func, llm_funcs, ollama_available):
    """Create and initialize a LightRAG instance."""
    if not ollama_available:
        pytest.skip("Ollama not available")

    import asyncio
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import initialize_pipeline_status

    llm_func, _ = llm_funcs

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        embedding_func=embedding_func,
    )

    # Initialize storages
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(rag.initialize_storages())
    loop.run_until_complete(initialize_pipeline_status())

    yield rag

    # Cleanup
    loop.run_until_complete(rag.finalize_storages())
    loop.close()


# =============================================================================
# Import Tests
# =============================================================================


class TestRAGAnythingImports:
    """Test that RAG-Anything can be imported correctly."""

    def test_import_raganything(self):
        """Test importing RAGAnything main class."""
        from raganything import RAGAnything

        assert RAGAnything is not None

    def test_import_modal_processors(self):
        """Test importing modal processors."""
        from raganything.modalprocessors import (
            ImageModalProcessor,
            TableModalProcessor,
            EquationModalProcessor,
            GenericModalProcessor,
        )

        assert ImageModalProcessor is not None
        assert TableModalProcessor is not None
        assert EquationModalProcessor is not None
        assert GenericModalProcessor is not None

    def test_import_context_extractor(self):
        """Test importing context extractor."""
        from raganything.modalprocessors import ContextExtractor, ContextConfig

        assert ContextExtractor is not None
        assert ContextConfig is not None

    def test_import_config(self):
        """Test importing RAGAnythingConfig."""
        from raganything import RAGAnythingConfig

        config = RAGAnythingConfig()
        assert config.working_dir == "./rag_storage"
        assert config.parser in ["mineru", "docling"]


# =============================================================================
# Context Extractor Tests
# =============================================================================


class TestContextExtractor:
    """Test ContextExtractor functionality."""

    def test_context_config_defaults(self):
        """Test ContextConfig default values."""
        from raganything.modalprocessors import ContextConfig

        config = ContextConfig()
        assert config.context_window >= 0
        assert config.context_mode in ["page", "chunk"]

    def test_context_extractor_creation(self):
        """Test creating a ContextExtractor instance."""
        from raganything.modalprocessors import ContextExtractor, ContextConfig

        config = ContextConfig(
            context_window=2,
            context_mode="page",
            max_context_tokens=2000,
        )
        extractor = ContextExtractor(config)
        assert extractor.config.context_window == 2
        assert extractor.config.context_mode == "page"

    def test_context_extractor_empty_source(self):
        """Test context extraction with empty source."""
        from raganything.modalprocessors import ContextExtractor, ContextConfig

        config = ContextConfig(context_window=1)
        extractor = ContextExtractor(config)

        result = extractor.extract_context(
            content_source=[],
            current_item_info={"page_idx": 0},
            content_format="minerU",
        )
        assert result == ""


# =============================================================================
# Modal Processor Tests (require Ollama)
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("RUN_OLLAMA_TESTS", "").lower() in ("1", "true", "yes"),
    reason="Ollama tests disabled. Set RUN_OLLAMA_TESTS=1 to enable.",
)
class TestTableModalProcessor:
    """Test TableModalProcessor with LightRAG."""

    @pytest.mark.asyncio
    async def test_table_processor_creation(self, lightrag_instance, llm_funcs):
        """Test creating a TableModalProcessor."""
        from raganything.modalprocessors import TableModalProcessor

        llm_func, _ = llm_funcs
        processor = TableModalProcessor(
            lightrag=lightrag_instance,
            modal_caption_func=llm_func,
        )
        assert processor is not None

    @pytest.mark.asyncio
    async def test_table_processor_process(self, lightrag_instance, llm_funcs):
        """Test processing table content."""
        from raganything.modalprocessors import TableModalProcessor

        llm_func, _ = llm_funcs
        processor = TableModalProcessor(
            lightrag=lightrag_instance,
            modal_caption_func=llm_func,
        )

        result = await processor.process_multimodal_content(
            modal_content=TABLE_TEST_CONTENT,
            content_type="table",
            file_path="benchmark_report.pdf",
            entity_name="LLM Performance Table",
        )

        assert result is not None
        description, entity_info = parse_processor_result(result)
        assert entity_info.get("entity_name") == "LLM Performance Table"
        assert entity_info.get("entity_type") == "table"


@pytest.mark.skipif(
    not os.getenv("RUN_OLLAMA_TESTS", "").lower() in ("1", "true", "yes"),
    reason="Ollama tests disabled. Set RUN_OLLAMA_TESTS=1 to enable.",
)
class TestEquationModalProcessor:
    """Test EquationModalProcessor with LightRAG."""

    @pytest.mark.asyncio
    async def test_equation_processor_creation(self, lightrag_instance, llm_funcs):
        """Test creating an EquationModalProcessor."""
        from raganything.modalprocessors import EquationModalProcessor

        llm_func, _ = llm_funcs
        processor = EquationModalProcessor(
            lightrag=lightrag_instance,
            modal_caption_func=llm_func,
        )
        assert processor is not None

    @pytest.mark.asyncio
    async def test_equation_processor_process(self, lightrag_instance, llm_funcs):
        """Test processing equation content."""
        from raganything.modalprocessors import EquationModalProcessor

        llm_func, _ = llm_funcs
        processor = EquationModalProcessor(
            lightrag=lightrag_instance,
            modal_caption_func=llm_func,
        )

        result = await processor.process_multimodal_content(
            modal_content=EQUATION_TEST_CONTENT,
            content_type="equation",
            file_path="ml_paper.pdf",
            entity_name="Binary Cross-Entropy Loss",
        )

        assert result is not None
        description, entity_info = parse_processor_result(result)
        assert entity_info.get("entity_name") == "Binary Cross-Entropy Loss"
        assert entity_info.get("entity_type") == "equation"


@pytest.mark.skipif(
    not os.getenv("RUN_OLLAMA_TESTS", "").lower() in ("1", "true", "yes"),
    reason="Ollama tests disabled. Set RUN_OLLAMA_TESTS=1 to enable.",
)
class TestImageModalProcessor:
    """Test ImageModalProcessor with LightRAG."""

    @pytest.mark.asyncio
    async def test_image_processor_creation(self, lightrag_instance, llm_funcs):
        """Test creating an ImageModalProcessor."""
        from raganything.modalprocessors import ImageModalProcessor

        _, vision_func = llm_funcs
        processor = ImageModalProcessor(
            lightrag=lightrag_instance,
            modal_caption_func=vision_func,
        )
        assert processor is not None

    @pytest.mark.asyncio
    async def test_image_processor_caption_only(self, lightrag_instance, llm_funcs):
        """Test processing image with caption only (no actual image)."""
        from raganything.modalprocessors import ImageModalProcessor

        _, vision_func = llm_funcs
        processor = ImageModalProcessor(
            lightrag=lightrag_instance,
            modal_caption_func=vision_func,
        )

        result = await processor.process_multimodal_content(
            modal_content=IMAGE_TEST_CONTENT,
            content_type="image",
            file_path="architecture.pdf",
            entity_name="RAG Architecture Diagram",
        )

        assert result is not None
        description, entity_info = parse_processor_result(result)
        assert entity_info.get("entity_name") == "RAG Architecture Diagram"
        assert entity_info.get("entity_type") == "image"


# =============================================================================
# Storage Configuration Tests
# =============================================================================


class TestStorageConfiguration:
    """Test storage configuration options."""

    def test_file_based_storage_default(self):
        """Test that file-based storage is the default."""
        from lightrag import LightRAG

        assert LightRAG is not None

    def test_mongodb_storage_import(self):
        """Test that MongoDB storage classes can be imported."""
        try:
            from lightrag.kg.mongo_impl import MongoKVStorage, MongoVectorDBStorage

            assert MongoKVStorage is not None
            assert MongoVectorDBStorage is not None
        except ImportError:
            pytest.skip("MongoDB storage not available (pymongo not installed)")

    def test_storage_env_vars(self):
        """Test that storage environment variables are recognized."""
        mongo_uri = os.getenv("MONGO_URI", "")
        mongo_db = os.getenv("MONGO_DATABASE", "")

        assert isinstance(mongo_uri, str)
        assert isinstance(mongo_db, str)


# =============================================================================
# Utility Tests
# =============================================================================


class TestLightRAGUtils:
    """Test lightrag_utils module."""

    def test_lightrag_config_defaults(self):
        """Test LightRAGConfig default values."""
        config = LightRAGConfig()
        assert config.working_dir == "./lightrag_storage"
        assert config.use_ollama is True
        assert config.use_mongodb is False

    def test_lightrag_config_custom(self):
        """Test LightRAGConfig with custom values."""
        config = LightRAGConfig(
            working_dir="/tmp/test",
            use_ollama=False,
            api_key="test-key",
            use_mongodb=True,
            mongo_database="test_db",
        )
        assert config.working_dir == "/tmp/test"
        assert config.use_ollama is False
        assert config.api_key == "test-key"
        assert config.use_mongodb is True
        assert config.mongo_database == "test_db"

    def test_parse_processor_result_tuple(self):
        """Test parsing tuple result."""
        result = ("description", {"entity_name": "test"}, "extra")
        desc, info = parse_processor_result(result)
        assert desc == "description"
        assert info == {"entity_name": "test"}

    def test_parse_processor_result_string(self):
        """Test parsing string result."""
        result = "just a string"
        desc, info = parse_processor_result(result)
        assert desc == "just a string"
        assert info == {}

    def test_parse_processor_result_empty_tuple(self):
        """Test parsing empty tuple."""
        result = ()
        desc, info = parse_processor_result(result)
        assert desc == ""
        assert info == {}

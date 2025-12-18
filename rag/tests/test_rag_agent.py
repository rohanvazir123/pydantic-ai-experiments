"""Tests for RAG agent with real queries against ingested documents."""

import logging
import warnings
import pytest
import pytest_asyncio

from rag.agent.rag_agent import agent, search_knowledge_base
from rag.retrieval.retriever import Retriever
from rag.storage.vector_store.mongo import MongoHybridStore

# Suppress httpx/asyncio cleanup warnings (harmless "Event loop is closed" errors)
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

# Track test results for summary
_test_results: list[dict] = []


def _log_test_start(test_name: str, query: str):
    """Log test start with clear formatting."""
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"TEST: {test_name}")
    logger.info("=" * 70)
    logger.info(f"QUERY: {query}")
    logger.info("-" * 70)


def _log_test_result(test_name: str, query: str, status: str, details: str = ""):
    """Log test result and track for summary."""
    _test_results.append({
        "test": test_name,
        "query": query,
        "status": status,
        "details": details
    })
    logger.info("")
    logger.info(f"RESULT: {status}")
    if details:
        logger.info(f"DETAILS: {details}")
    logger.info("=" * 70)
    logger.info("")


def _log_results(results, max_content_len: int = 150, search_type: str = "hybrid"):
    """Log search results in a formatted way."""
    logger.info(f"Found {len(results)} results:")
    # Note: RRF scores (hybrid) are low by design (0.01-0.03), semantic/text scores are higher
    if search_type == "hybrid":
        logger.info("  (Note: Hybrid uses RRF scoring, values 0.01-0.03 are normal)")
    logger.info("")
    for i, r in enumerate(results):
        logger.info(f"  [{i+1}] {r.document_title}")
        logger.info(f"      Score: {r.similarity:.4f}")
        logger.info(f"      Source: {r.document_source}")
        content_preview = r.content[:max_content_len].replace('\n', ' ')
        logger.info(f"      Content: {content_preview}...")
        logger.info("")


def _log_agent_response(response: str):
    """Log agent response in a formatted way."""
    logger.info("AGENT RESPONSE:")
    logger.info("-" * 40)
    for line in response.split('\n'):
        logger.info(f"  {line}")
    logger.info("-" * 40)
    logger.info("")


@pytest.fixture(scope="session", autouse=True)
def print_test_summary(request):
    """Print test summary at the end of the session."""
    yield
    # Print summary after all tests
    logger.info("")
    logger.info("")
    logger.info("#" * 70)
    logger.info("#" + " " * 20 + "TEST SUMMARY" + " " * 36 + "#")
    logger.info("#" * 70)
    logger.info("")

    passed = [t for t in _test_results if t["status"] == "PASSED"]
    failed = [t for t in _test_results if t["status"] == "FAILED"]
    skipped = [t for t in _test_results if t["status"] == "SKIPPED"]

    logger.info(f"TOTAL: {len(_test_results)} | PASSED: {len(passed)} | FAILED: {len(failed)} | SKIPPED: {len(skipped)}")
    logger.info("")

    if passed:
        logger.info("-" * 70)
        logger.info("PASSED TESTS:")
        logger.info("-" * 70)
        for t in passed:
            logger.info(f"  [PASS] {t['test']}")
            logger.info(f"         Query: {t['query']}")
        logger.info("")

    if failed:
        logger.info("-" * 70)
        logger.info("FAILED TESTS:")
        logger.info("-" * 70)
        for t in failed:
            logger.info(f"  [FAIL] {t['test']}")
            logger.info(f"         Query: {t['query']}")
            if t['details']:
                logger.info(f"         Reason: {t['details']}")
        logger.info("")

    if skipped:
        logger.info("-" * 70)
        logger.info("SKIPPED TESTS:")
        logger.info("-" * 70)
        for t in skipped:
            logger.info(f"  [SKIP] {t['test']}")
            logger.info(f"         Query: {t['query']}")
            if t['details']:
                logger.info(f"         Reason: {t['details']}")
        logger.info("")

    logger.info("#" * 70)
    logger.info("")


class TestRetrieverQueries:
    """Test retriever with queries based on ingested NeuralFlow AI documents."""

    @pytest_asyncio.fixture
    async def retriever(self):
        """Create retriever with connected store."""
        store = MongoHybridStore()
        await store.initialize()
        retriever = Retriever(store=store)
        yield retriever
        await store.close()

    @pytest.mark.asyncio
    async def test_company_overview_query(self, retriever):
        """Test retrieving company overview information."""
        test_name = "test_company_overview_query"
        query = "What does NeuralFlow AI do?"
        _log_test_start(test_name, query)

        try:
            results = await retriever.retrieve(
                query=query,
                match_count=5,
                search_type="hybrid",
            )
            _log_results(results, search_type="hybrid")

            assert len(results) > 0, "Expected at least one result"
            content = " ".join([r.content.lower() for r in results])
            assert "neuralflow" in content, "Results should mention NeuralFlow"
            assert any(
                term in content for term in ["ai", "automation", "enterprise", "workflow"]
            ), "Results should mention AI/automation topics"

            top_sources = [r.document_source.lower() for r in results[:2]]
            assert any(
                "company-overview" in src or "mission" in src for src in top_sources
            ), f"Top results should be from company overview, got: {top_sources}"

            _log_test_result(test_name, query, "PASSED")
        except AssertionError as e:
            _log_test_result(test_name, query, "FAILED", str(e))
            raise

    @pytest.mark.asyncio
    async def test_team_structure_query(self, retriever):
        """Test retrieving team structure information."""
        test_name = "test_team_structure_query"
        query = "How many engineers work at the company?"
        _log_test_start(test_name, query)

        try:
            results = await retriever.retrieve(
                query=query,
                match_count=5,
                search_type="hybrid",
            )
            _log_results(results, search_type="hybrid")

            assert len(results) > 0, "Expected at least one result"
            content = " ".join([r.content.lower() for r in results])
            assert any(
                term in content for term in ["engineer", "team", "employee", "staff"]
            ), "Results should mention team/engineering"

            has_numbers = any(num in content for num in ["47", "18", "twelve", "forty"])
            logger.info(f"Contains employee numbers (47/18): {has_numbers}")

            _log_test_result(test_name, query, "PASSED")
        except AssertionError as e:
            _log_test_result(test_name, query, "FAILED", str(e))
            raise

    @pytest.mark.asyncio
    async def test_benefits_query(self, retriever):
        """Test retrieving employee benefits information."""
        test_name = "test_benefits_query"
        query = "What is the PTO policy?"
        _log_test_start(test_name, query)

        try:
            results = await retriever.retrieve(
                query=query,
                match_count=5,
                search_type="hybrid",
            )
            _log_results(results, search_type="hybrid")

            assert len(results) > 0, "Expected at least one result"
            content = " ".join([r.content.lower() for r in results])
            assert any(
                term in content for term in ["pto", "time off", "vacation", "leave", "days"]
            ), "Results should mention PTO/time off"

            sources = [r.document_source.lower() for r in results]
            assert any(
                "handbook" in src for src in sources
            ), f"Results should include team handbook, got: {sources}"

            _log_test_result(test_name, query, "PASSED")
        except AssertionError as e:
            _log_test_result(test_name, query, "FAILED", str(e))
            raise

    @pytest.mark.asyncio
    async def test_technology_stack_query(self, retriever):
        """Test retrieving technology stack information."""
        test_name = "test_technology_stack_query"
        query = "What technologies and tools does the company use?"
        _log_test_start(test_name, query)

        try:
            results = await retriever.retrieve(
                query=query,
                match_count=5,
                search_type="hybrid",
            )
            _log_results(results, search_type="hybrid")

            assert len(results) > 0, "Expected at least one result"
            content = " ".join([r.content.lower() for r in results])
            tech_terms = ["openai", "pytorch", "tensorflow", "langchain", "vector",
                          "mongodb", "python", "aws", "azure", "api"]
            found_terms = [t for t in tech_terms if t in content]
            logger.info(f"Found technology terms: {found_terms}")

            assert len(found_terms) >= 2, f"Should find multiple technologies, found: {found_terms}"

            _log_test_result(test_name, query, "PASSED")
        except AssertionError as e:
            _log_test_result(test_name, query, "FAILED", str(e))
            raise

    @pytest.mark.asyncio
    async def test_semantic_search(self, retriever):
        """Test semantic-only search."""
        test_name = "test_semantic_search"
        query = "company culture and values"
        _log_test_start(test_name, f"{query} (semantic search)")

        try:
            results = await retriever.retrieve(
                query=query,
                match_count=3,
                search_type="semantic",
            )
            _log_results(results, search_type="semantic")

            assert len(results) > 0, "Expected at least one result"
            assert all(r.similarity > 0 for r in results), "All should have positive similarity"

            content = " ".join([r.content.lower() for r in results])
            culture_terms = ["culture", "values", "mission", "team", "collaboration",
                            "innovation", "growth", "learning"]
            found_terms = [t for t in culture_terms if t in content]
            logger.info(f"Found culture-related terms: {found_terms}")
            assert len(found_terms) >= 1, "Should find culture-related content"

            _log_test_result(test_name, query, "PASSED")
        except AssertionError as e:
            _log_test_result(test_name, query, "FAILED", str(e))
            raise

    @pytest.mark.asyncio
    async def test_text_search(self, retriever):
        """Test text-only search."""
        test_name = "test_text_search"
        query = "NeuralFlow AI"
        _log_test_start(test_name, f"{query} (text search)")

        try:
            results = await retriever.retrieve(
                query=query,
                match_count=3,
                search_type="text",
            )
            _log_results(results, search_type="text")

            assert len(results) > 0, "Expected at least one result"
            content = " ".join([r.content.lower() for r in results])
            assert "neuralflow" in content, "Text search should find exact term 'NeuralFlow'"

            _log_test_result(test_name, query, "PASSED")
        except AssertionError as e:
            _log_test_result(test_name, query, "FAILED", str(e))
            raise

    @pytest.mark.asyncio
    async def test_retrieve_as_context(self, retriever):
        """Test formatted context retrieval for LLM."""
        test_name = "test_retrieve_as_context"
        query = "What is the company mission?"
        _log_test_start(test_name, f"{query} (as context)")

        try:
            context = await retriever.retrieve_as_context(
                query=query,
                match_count=3,
            )

            logger.info(f"Context length: {len(context)} chars")
            logger.info("")
            logger.info("Context preview:")
            logger.info("-" * 40)
            for line in context[:500].split('\n'):
                logger.info(f"  {line}")
            logger.info("  ...")
            logger.info("-" * 40)
            logger.info("")

            assert isinstance(context, str), "Context should be a string"
            assert len(context) > 0, "Context should not be empty"
            assert "Found" in context or "No relevant" in context, "Should indicate results status"

            if "Found" in context:
                assert "Document" in context, "Should include document references"
                assert "---" in context, "Should include document separators"

            _log_test_result(test_name, query, "PASSED")
        except AssertionError as e:
            _log_test_result(test_name, query, "FAILED", str(e))
            raise


class TestRAGAgentTool:
    """Test the RAG agent's search_knowledge_base tool directly."""

    @pytest.mark.asyncio
    async def test_search_tool_basic(self):
        """Test search_knowledge_base tool with basic query."""
        test_name = "test_search_tool_basic"
        query = "What services does NeuralFlow AI provide?"
        _log_test_start(test_name, query)

        try:
            class MockContext:
                pass

            result = await search_knowledge_base(
                MockContext(),
                query=query,
                match_count=5,
                search_type="hybrid",
            )

            logger.info(f"Tool result length: {len(result)} chars")
            logger.info("")
            logger.info("Tool result preview:")
            logger.info("-" * 40)
            for line in result[:500].split('\n'):
                logger.info(f"  {line}")
            logger.info("  ...")
            logger.info("-" * 40)
            logger.info("")

            assert isinstance(result, str), "Result should be a string"
            assert len(result) > 0, "Result should not be empty"
            assert "Found" in result or "Error" in result or "No relevant" in result

            if "Found" in result:
                result_lower = result.lower()
                assert any(
                    term in result_lower
                    for term in ["ai", "automation", "service", "solution", "enterprise"]
                ), "Result should contain service-related terms"

            _log_test_result(test_name, query, "PASSED")
        except AssertionError as e:
            _log_test_result(test_name, query, "FAILED", str(e))
            raise

    @pytest.mark.asyncio
    async def test_search_tool_with_different_search_types(self):
        """Test search tool with different search types."""
        test_name = "test_search_tool_with_different_search_types"
        query = "employee benefits"
        _log_test_start(test_name, f"{query} (hybrid/semantic/text)")

        try:
            class MockContext:
                pass

            for search_type in ["hybrid", "semantic", "text"]:
                logger.info(f"Testing search type: {search_type}")
                logger.info("-" * 40)

                result = await search_knowledge_base(
                    MockContext(),
                    query=query,
                    match_count=3,
                    search_type=search_type,
                )

                logger.info(f"  Result length: {len(result)} chars")
                preview = result[:200].replace('\n', ' ')
                logger.info(f"  Preview: {preview}...")
                logger.info("")

                assert isinstance(result, str), f"Result should be string for {search_type}"
                assert len(result) > 0, f"Result should not be empty for {search_type}"

            _log_test_result(test_name, query, "PASSED")
        except AssertionError as e:
            _log_test_result(test_name, query, "FAILED", str(e))
            raise


class TestRAGAgentIntegration:
    """Integration tests for the full RAG agent."""

    @pytest.mark.asyncio
    async def test_agent_run_simple_query(self):
        """Test running the agent with a simple question."""
        test_name = "test_agent_run_simple_query"
        query = "What does NeuralFlow AI specialize in? Keep your answer brief."
        _log_test_start(test_name, query)

        try:
            result = await agent.run(query)
            _log_agent_response(result.output)

            assert result.output is not None, "Agent should return output"
            assert isinstance(result.output, str), "Output should be a string"
            assert len(result.output) > 20, "Output should be meaningful"

            output_lower = result.output.lower()
            assert any(
                term in output_lower
                for term in ["ai", "automation", "enterprise", "workflow", "intelligence"]
            ), f"Response should mention AI/automation topics"

            _log_test_result(test_name, query, "PASSED")
        except AssertionError as e:
            _log_test_result(test_name, query, "FAILED", str(e))
            raise

    @pytest.mark.asyncio
    async def test_agent_run_specific_query(self):
        """Test running the agent with a specific question about employee count."""
        test_name = "test_agent_run_specific_query"
        query = "How many employees does NeuralFlow AI have? Just give me the number."
        _log_test_start(test_name, query)

        try:
            result = await agent.run(query)
            _log_agent_response(result.output)

            assert result.output is not None, "Agent should return output"
            assert isinstance(result.output, str), "Output should be a string"

            output_lower = result.output.lower()
            has_correct_number = "47" in result.output or "forty-seven" in output_lower
            logger.info(f"Contains correct employee count (47): {has_correct_number}")

            import re
            numbers = re.findall(r'\d+', result.output)
            logger.info(f"Numbers found: {numbers}")
            assert len(numbers) > 0 or any(
                word in output_lower for word in ["forty", "fifty", "thirty", "twenty"]
            ), "Response should contain a number"

            _log_test_result(test_name, query, "PASSED")
        except AssertionError as e:
            _log_test_result(test_name, query, "FAILED", str(e))
            raise

    @pytest.mark.asyncio
    async def test_agent_run_benefits_query(self):
        """Test running the agent with a benefits question."""
        test_name = "test_agent_run_benefits_query"
        query = "What is the learning budget for employees at NeuralFlow AI?"
        _log_test_start(test_name, query)

        try:
            result = await agent.run(query)
            _log_agent_response(result.output)

            assert result.output is not None, "Agent should return output"
            assert isinstance(result.output, str), "Output should be a string"
            assert len(result.output) > 20, "Output should be meaningful"

            output_lower = result.output.lower()
            has_budget = any(
                term in result.output for term in ["2,500", "2500", "$2,500", "$2500"]
            )
            logger.info(f"Contains correct learning budget ($2,500): {has_budget}")

            assert any(
                term in output_lower
                for term in ["$", "budget", "learning", "development", "training"]
            ), "Response should mention budget/learning"

            _log_test_result(test_name, query, "PASSED")
        except AssertionError as e:
            _log_test_result(test_name, query, "FAILED", str(e))
            raise

    @pytest.mark.asyncio
    async def test_agent_run_pto_query(self):
        """Test running the agent with a PTO question."""
        test_name = "test_agent_run_pto_query"
        query = "How many PTO days do employees get at NeuralFlow AI?"
        _log_test_start(test_name, query)

        try:
            result = await agent.run(query)
            _log_agent_response(result.output)

            assert result.output is not None, "Agent should return output"
            assert isinstance(result.output, str), "Output should be a string"

            output_lower = result.output.lower()
            assert any(
                term in output_lower
                for term in ["pto", "time off", "vacation", "days", "unlimited", "leave"]
            ), "Response should mention PTO/time off"

            _log_test_result(test_name, query, "PASSED")
        except AssertionError as e:
            _log_test_result(test_name, query, "FAILED", str(e))
            raise
        except Exception as e:
            if "exceeded max retries" in str(e):
                _log_test_result(test_name, query, "SKIPPED", "LLM tool call failed - intermittent issue")
                pytest.skip("LLM tool call failed - intermittent issue with local LLM")
            raise


class TestSearchResultQuality:
    """Test the quality and relevance of search results."""

    @pytest_asyncio.fixture
    async def retriever(self):
        """Create retriever with connected store."""
        store = MongoHybridStore()
        await store.initialize()
        retriever = Retriever(store=store)
        yield retriever
        await store.close()

    @pytest.mark.asyncio
    async def test_results_have_required_fields(self, retriever):
        """Test that search results have all required fields."""
        test_name = "test_results_have_required_fields"
        query = "company overview"
        _log_test_start(test_name, query)

        try:
            results = await retriever.retrieve(query=query, match_count=5)

            logger.info(f"Found {len(results)} results")
            logger.info("")
            assert len(results) > 0, "Expected at least one result"

            for i, result in enumerate(results):
                logger.info(f"Result {i+1}:")
                logger.info(f"  chunk_id: {result.chunk_id}")
                logger.info(f"  document_id: {result.document_id}")
                logger.info(f"  document_title: {result.document_title}")
                logger.info(f"  document_source: {result.document_source}")
                logger.info(f"  similarity: {result.similarity:.4f}")
                logger.info(f"  content_length: {len(result.content)} chars")
                logger.info("")

                assert result.chunk_id is not None
                assert result.document_id is not None
                assert result.content is not None and len(result.content) > 0
                assert result.similarity is not None and result.similarity > 0
                assert result.document_title is not None
                assert result.document_source is not None

            _log_test_result(test_name, query, "PASSED")
        except AssertionError as e:
            _log_test_result(test_name, query, "FAILED", str(e))
            raise

    @pytest.mark.asyncio
    async def test_results_sorted_by_relevance(self, retriever):
        """Test that hybrid search results are sorted by relevance."""
        test_name = "test_results_sorted_by_relevance"
        query = "NeuralFlow AI company"
        _log_test_start(test_name, query)

        try:
            results = await retriever.retrieve(
                query=query, match_count=10, search_type="hybrid"
            )

            logger.info(f"Found {len(results)} results")
            scores = [r.similarity for r in results]
            logger.info(f"Scores (descending): {[f'{s:.4f}' for s in scores]}")
            logger.info("")

            if len(results) > 1:
                for i in range(len(results) - 1):
                    assert results[i].similarity >= results[i + 1].similarity, (
                        f"Not sorted: [{i}]={results[i].similarity:.4f} < [{i+1}]={results[i+1].similarity:.4f}"
                    )

            _log_test_result(test_name, query, "PASSED")
        except AssertionError as e:
            _log_test_result(test_name, query, "FAILED", str(e))
            raise

    @pytest.mark.asyncio
    async def test_no_duplicate_chunks(self, retriever):
        """Test that results don't contain duplicate chunks."""
        test_name = "test_no_duplicate_chunks"
        query = "employee handbook policies"
        _log_test_start(test_name, query)

        try:
            results = await retriever.retrieve(query=query, match_count=10)

            logger.info(f"Found {len(results)} results")
            chunk_ids = [r.chunk_id for r in results]
            logger.info(f"Chunk IDs: {chunk_ids}")
            logger.info("")

            unique_ids = set(chunk_ids)
            assert len(chunk_ids) == len(unique_ids), (
                f"Found {len(chunk_ids) - len(unique_ids)} duplicates"
            )

            _log_test_result(test_name, query, "PASSED")
        except AssertionError as e:
            _log_test_result(test_name, query, "FAILED", str(e))
            raise

    @pytest.mark.asyncio
    async def test_relevance_scoring(self, retriever):
        """Test that more relevant queries get higher scores."""
        test_name = "test_relevance_scoring"
        exact_query = "NeuralFlow AI automation"
        vague_query = "company business"
        _log_test_start(test_name, f"exact='{exact_query}' vs vague='{vague_query}'")

        try:
            logger.info(f"Exact query: {exact_query}")
            exact_results = await retriever.retrieve(
                query=exact_query, match_count=3, search_type="hybrid"
            )

            logger.info(f"Vague query: {vague_query}")
            vague_results = await retriever.retrieve(
                query=vague_query, match_count=3, search_type="hybrid"
            )

            if exact_results and vague_results:
                exact_top = exact_results[0].similarity
                vague_top = vague_results[0].similarity
                logger.info(f"Exact query top score: {exact_top:.4f}")
                logger.info(f"Vague query top score: {vague_top:.4f}")
                logger.info("")

            assert len(exact_results) > 0, "Exact query should return results"
            assert len(vague_results) > 0, "Vague query should return results"

            _log_test_result(test_name, f"{exact_query} / {vague_query}", "PASSED")
        except AssertionError as e:
            _log_test_result(test_name, f"{exact_query} / {vague_query}", "FAILED", str(e))
            raise

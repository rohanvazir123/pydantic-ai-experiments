"""Tests for rag.ingestion.cuad_ingestion.

Unit and integration tests covering parsing, file extraction, eval pair
saving, and pipeline wiring. All filesystem I/O uses pytest's tmp_path
fixture. The pipeline is mocked so no live DB or Ollama is required.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.ingestion.cuad_ingestion import (
    CuadContract,
    CuadQA,
    _extract_contract_type,
    _extract_question_type,
    load_cuad,
    run_ingestion,
    save_eval_pairs,
    write_contract_files,
)

# ---------------------------------------------------------------------------
# Minimal CUAD-format JSON fixture
# ---------------------------------------------------------------------------

SAMPLE_CUAD_JSON = {
    "data": [
        {
            "title": "ACMECORP_20200101_10-K_EX-10_1234_EX-10_Distributor Agreement",
            "paragraphs": [
                {
                    "context": "This Distributor Agreement is entered into by Acme Corp and Beta Ltd. "
                    "The parties agree to distribute products in the Territory. "
                    "This agreement is governed by English law.",
                    "qas": [
                        {
                            "question": 'Highlight the parts related to "Parties" that should be reviewed.',
                            "is_impossible": False,
                            "answers": [
                                {"text": "Acme Corp", "answer_start": 51},
                                {"text": "Beta Ltd", "answer_start": 65},
                            ],
                        },
                        {
                            "question": 'Highlight the parts related to "Governing Law" that should be reviewed.',
                            "is_impossible": False,
                            "answers": [{"text": "English law", "answer_start": 130}],
                        },
                        {
                            "question": 'Highlight the parts related to "Expiration Date" that should be reviewed.',
                            "is_impossible": True,
                            "answers": [],
                        },
                    ],
                }
            ],
        },
        {
            "title": "BETACORP_20210601_S-1_EX-10_5678_EX-10_License Agreement",
            "paragraphs": [
                {
                    "context": "License Agreement between Beta Corp and Gamma Inc. "
                    "Licensor grants non-exclusive rights to use the Software.",
                    "qas": [
                        {
                            "question": 'Highlight the parts related to "Parties" that should be reviewed.',
                            "is_impossible": False,
                            "answers": [
                                {"text": "Beta Corp", "answer_start": 20},
                                {"text": "Gamma Inc", "answer_start": 34},
                            ],
                        },
                        {
                            "question": 'Highlight the parts related to "Expiration Date" that should be reviewed.',
                            "is_impossible": True,
                            "answers": [],
                        },
                    ],
                }
            ],
        },
    ]
}


@pytest.fixture
def cuad_json_file(tmp_path: Path) -> Path:
    """Write a minimal CUAD JSON file and return its path."""
    p = tmp_path / "CUAD_v1.json"
    p.write_text(json.dumps(SAMPLE_CUAD_JSON), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _extract_question_type
# ---------------------------------------------------------------------------


class TestExtractQuestionType:
    def test_extracts_type_from_standard_question(self):
        q = 'Highlight the parts related to "Parties" that should be reviewed.'
        assert _extract_question_type(q) == "Parties"

    def test_extracts_multi_word_type(self):
        q = 'Highlight the parts related to "Governing Law" that should be reviewed.'
        assert _extract_question_type(q) == "Governing Law"

    def test_falls_back_to_unknown_when_no_quotes(self):
        assert _extract_question_type("What is the governing law?") == "Unknown"

    def test_extracts_first_quoted_group(self):
        q = 'Related to "Termination" and "Notice Period"'
        assert _extract_question_type(q) == "Termination"


# ---------------------------------------------------------------------------
# _extract_contract_type
# ---------------------------------------------------------------------------


class TestExtractContractType:
    def test_extracts_last_segment(self):
        title = "ACMECORP_20200101_10-K_EX-10_1234_EX-10_Distributor Agreement"
        result = _extract_contract_type(title)
        assert "Distributor" in result
        assert "Agreement" in result

    def test_handles_title_with_no_underscore(self):
        # Should not raise
        result = _extract_contract_type("SIMPLE TITLE")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_titlecase(self):
        title = "ACMECORP_20200101_LICENSE AGREEMENT"
        result = _extract_contract_type(title)
        # Should be title-cased, not all-caps
        assert result == result.title() or result[0].isupper()


# ---------------------------------------------------------------------------
# CuadContract.safe_filename
# ---------------------------------------------------------------------------


class TestSafeFilename:
    def test_replaces_spaces_with_underscore(self):
        c = CuadContract(
            title="My Contract",
            contract_type="NDA",
            source_id="My Contract Title Here",
            text="content",
        )
        assert " " not in c.safe_filename

    def test_ends_with_md(self):
        c = CuadContract(
            title="Test",
            contract_type="NDA",
            source_id="ACME_20200101_AGREEMENT",
            text="content",
        )
        assert c.safe_filename.endswith(".md")

    def test_capped_at_120_chars_plus_extension(self):
        long_id = "A" * 200
        c = CuadContract(
            title="T", contract_type="NDA", source_id=long_id, text="x"
        )
        assert len(c.safe_filename) <= 124  # 120 + len(".md")

    def test_special_chars_removed(self):
        c = CuadContract(
            title="T",
            contract_type="NDA",
            source_id="File: Name/With\\Slashes?",
            text="x",
        )
        assert "/" not in c.safe_filename
        assert "\\" not in c.safe_filename


# ---------------------------------------------------------------------------
# load_cuad
# ---------------------------------------------------------------------------


class TestLoadCuad:
    def test_returns_correct_count(self, cuad_json_file):
        contracts = load_cuad(cuad_json_file)
        assert len(contracts) == 2

    def test_contract_fields_populated(self, cuad_json_file):
        contracts = load_cuad(cuad_json_file)
        c = contracts[0]
        assert "ACMECORP" in c.source_id
        assert "Acme Corp" in c.text
        assert c.contract_type  # non-empty

    def test_qas_parsed(self, cuad_json_file):
        contracts = load_cuad(cuad_json_file)
        c = contracts[0]
        assert len(c.qas) == 3

    def test_impossible_qas_flagged(self, cuad_json_file):
        contracts = load_cuad(cuad_json_file)
        impossibles = [qa for qa in contracts[0].qas if qa.is_impossible]
        answered = [qa for qa in contracts[0].qas if not qa.is_impossible]
        assert len(impossibles) == 1
        assert len(answered) == 2

    def test_answers_extracted(self, cuad_json_file):
        contracts = load_cuad(cuad_json_file)
        parties_qa = contracts[0].qas[0]
        assert "Acme Corp" in parties_qa.answers
        assert "Beta Ltd" in parties_qa.answers

    def test_question_types_extracted(self, cuad_json_file):
        contracts = load_cuad(cuad_json_file)
        types = [qa.question_type for qa in contracts[0].qas]
        assert "Parties" in types
        assert "Governing Law" in types


# ---------------------------------------------------------------------------
# write_contract_files
# ---------------------------------------------------------------------------


class TestWriteContractFiles:
    def _make_contracts(self) -> list[CuadContract]:
        return [
            CuadContract("Title A", "NDA", "Contract_A", "Text of contract A."),
            CuadContract("Title B", "MSA", "Contract_B", "Text of contract B."),
        ]

    def test_creates_output_directory(self, tmp_path):
        out = tmp_path / "legal"
        write_contract_files(self._make_contracts(), out)
        assert out.exists()

    def test_writes_one_file_per_contract(self, tmp_path):
        out = tmp_path / "legal"
        written = write_contract_files(self._make_contracts(), out)
        assert len(written) == 2
        assert all(p.exists() for p in written)

    def test_files_are_markdown(self, tmp_path):
        out = tmp_path / "legal"
        written = write_contract_files(self._make_contracts(), out)
        assert all(p.suffix == ".md" for p in written)

    def test_file_contains_title_heading(self, tmp_path):
        out = tmp_path / "legal"
        contracts = [CuadContract("My NDA", "NDA", "My_NDA", "Body text.")]
        written = write_contract_files(contracts, out)
        content = written[0].read_text(encoding="utf-8")
        assert content.startswith("# My NDA")

    def test_file_contains_contract_text(self, tmp_path):
        out = tmp_path / "legal"
        contracts = [CuadContract("T", "NDA", "T_id", "Unique contract body xyz.")]
        written = write_contract_files(contracts, out)
        content = written[0].read_text(encoding="utf-8")
        assert "Unique contract body xyz." in content

    def test_returns_list_of_paths(self, tmp_path):
        out = tmp_path / "legal"
        written = write_contract_files(self._make_contracts(), out)
        assert all(isinstance(p, Path) for p in written)


# ---------------------------------------------------------------------------
# save_eval_pairs
# ---------------------------------------------------------------------------


class TestSaveEvalPairs:
    def _make_contracts(self) -> list[CuadContract]:
        return [
            CuadContract(
                title="Contract A",
                contract_type="NDA",
                source_id="A",
                text="text",
                qas=[
                    CuadQA("Q about Parties", "Parties", ["Acme"], [0], False),
                    CuadQA("Q about Expiry", "Expiry", [], [], True),   # impossible
                ],
            ),
            CuadContract(
                title="Contract B",
                contract_type="MSA",
                source_id="B",
                text="text",
                qas=[
                    CuadQA("Q about Gov Law", "Governing Law", ["English law"], [10], False),
                ],
            ),
        ]

    def test_creates_output_file(self, tmp_path):
        out = tmp_path / "eval.json"
        save_eval_pairs(self._make_contracts(), out)
        assert out.exists()

    def test_excludes_impossible_qas(self, tmp_path):
        out = tmp_path / "eval.json"
        count = save_eval_pairs(self._make_contracts(), out)
        assert count == 2  # Parties + Governing Law; Expiry is impossible

    def test_json_structure(self, tmp_path):
        out = tmp_path / "eval.json"
        save_eval_pairs(self._make_contracts(), out)
        pairs = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(pairs, list)
        pair = pairs[0]
        assert "contract_title" in pair
        assert "contract_type" in pair
        assert "question_type" in pair
        assert "question" in pair
        assert "answers" in pair
        assert "answer_starts" in pair

    def test_answers_included(self, tmp_path):
        out = tmp_path / "eval.json"
        save_eval_pairs(self._make_contracts(), out)
        pairs = json.loads(out.read_text(encoding="utf-8"))
        parties_pair = next(p for p in pairs if p["question_type"] == "Parties")
        assert "Acme" in parties_pair["answers"]

    def test_empty_contracts_saves_empty_list(self, tmp_path):
        out = tmp_path / "eval.json"
        count = save_eval_pairs([], out)
        assert count == 0
        pairs = json.loads(out.read_text(encoding="utf-8"))
        assert pairs == []

    def test_creates_parent_directory(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "eval.json"
        save_eval_pairs(self._make_contracts(), out)
        assert out.exists()


# ---------------------------------------------------------------------------
# run_ingestion
# ---------------------------------------------------------------------------


class TestRunIngestion:
    def _mock_pipeline(self, results=None):
        if results is None:
            r = MagicMock()
            r.chunks_created = 10
            r.errors = []
            results = [r]
        pipeline = MagicMock()
        pipeline.initialize = AsyncMock()
        pipeline.ingest_documents = AsyncMock(return_value=results)
        pipeline.close = AsyncMock()
        return pipeline

    @pytest.mark.asyncio
    async def test_returns_summary_dict(self):
        pipeline = self._mock_pipeline()
        with patch("rag.ingestion.cuad_ingestion.create_pipeline", return_value=pipeline):
            result = await run_ingestion("rag/documents/legal")
        assert result["documents"] == 1
        assert result["chunks"] == 10
        assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_forwards_parameters(self):
        pipeline = self._mock_pipeline()
        with patch(
            "rag.ingestion.cuad_ingestion.create_pipeline", return_value=pipeline
        ) as mock_create:
            await run_ingestion(
                "custom/path", clean=True, chunk_size=500, max_tokens=256
            )
        mock_create.assert_called_once_with(
            documents_folder="custom/path",
            clean=True,
            chunk_size=500,
            max_tokens=256,
        )

    @pytest.mark.asyncio
    async def test_close_called_on_success(self):
        pipeline = self._mock_pipeline()
        with patch("rag.ingestion.cuad_ingestion.create_pipeline", return_value=pipeline):
            await run_ingestion("path")
        pipeline.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_called_on_error(self):
        pipeline = self._mock_pipeline()
        pipeline.ingest_documents.side_effect = RuntimeError("disk full")
        with patch("rag.ingestion.cuad_ingestion.create_pipeline", return_value=pipeline):
            with pytest.raises(RuntimeError, match="disk full"):
                await run_ingestion("path")
        pipeline.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_counts_errors_across_results(self):
        r1 = MagicMock()
        r1.chunks_created = 5
        r1.errors = ["parse error"]
        r2 = MagicMock()
        r2.chunks_created = 3
        r2.errors = []
        pipeline = self._mock_pipeline(results=[r1, r2])
        with patch("rag.ingestion.cuad_ingestion.create_pipeline", return_value=pipeline):
            result = await run_ingestion("path")
        assert result["documents"] == 2
        assert result["chunks"] == 8
        assert result["errors"] == 1

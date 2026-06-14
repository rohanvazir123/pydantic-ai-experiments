"""Unit tests for knowledge.config.settings.

No external services required. All tests instantiate Settings directly
to avoid polluting the load_settings() LRU cache.
"""

import json
import os
from unittest import mock

import pytest

from knowledge.config.settings import Settings, load_settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_ENV = {
    "DATABASE_URL": "postgresql://ragv2:test@localhost:5432/ragv2_test",
    "AGE_DATABASE_URL": "postgresql://age:test@localhost:5433/age_test",
}


def make_settings(**overrides: str) -> Settings:
    """Create an isolated Settings instance for testing.

    Kwargs are field names (lowercase or uppercase — both accepted).
    os.environ is temporarily replaced with only the values we specify,
    preventing conftest or other test fixtures from leaking in.
    """
    # Build final env: required fields + caller overrides (all uppercase for env)
    env: dict[str, str] = {**REQUIRED_ENV}
    # Accept both LLM_MODEL and llm_model as kwargs → normalise to UPPERCASE env
    env.update({k.upper(): str(v) for k, v in overrides.items()})

    # clear=True: wipe os.environ entirely for the duration of Settings()
    # so that conftest values don't contaminate field defaults.
    with mock.patch.dict(os.environ, env, clear=True):
        return Settings(_env_file=None)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------


class TestSettingsLoading:
    def test_loads_with_required_fields(self) -> None:
        s = make_settings()
        assert s.database_url == REQUIRED_ENV["DATABASE_URL"]
        assert s.age_database_url == REQUIRED_ENV["AGE_DATABASE_URL"]

    def test_defaults_applied(self) -> None:
        s = make_settings()
        assert s.redis_url == "redis://localhost:6379"
        assert s.llm_model == "llama3.2:3b"
        assert s.embedding_dimension == 768
        assert s.semantic_cache_enabled is True
        assert s.mem0_enabled is False
        assert s.langfuse_enabled is False
        assert s.cloud_models_enabled is False

    def test_missing_required_field_raises(self) -> None:
        # Only AGE_DATABASE_URL in env — DATABASE_URL absent → should raise
        with mock.patch.dict(os.environ, {"AGE_DATABASE_URL": "postgresql://x"}, clear=True):
            with pytest.raises(Exception):
                Settings(_env_file=None)  # type: ignore[call-arg]

    def test_overrides_applied(self) -> None:
        s = make_settings(LLM_MODEL="llama3.1:70b", EMBEDDING_DIMENSION="1536")
        assert s.llm_model == "llama3.1:70b"
        assert s.embedding_dimension == 1536


# ---------------------------------------------------------------------------
# Corpus config parsing
# ---------------------------------------------------------------------------


class TestCorpusConfigParsing:
    def test_empty_corpus_configs(self) -> None:
        s = make_settings(CORPUS_CONFIGS_JSON="[]")
        assert s.corpus_configs == []

    def test_single_corpus_config(self) -> None:
        corpus = [{"id": "hr", "display_name": "HR Docs", "source_folders": ["/mnt/hr"]}]
        s = make_settings(CORPUS_CONFIGS_JSON=json.dumps(corpus))
        assert len(s.corpus_configs) == 1
        c = s.corpus_configs[0]
        assert c.id == "hr"
        assert c.display_name == "HR Docs"
        assert len(c.source_folders) == 1
        assert str(c.source_folders[0]) == "/mnt/hr"

    def test_corpus_defaults(self) -> None:
        corpus = [{"id": "default"}]
        s = make_settings(CORPUS_CONFIGS_JSON=json.dumps(corpus))
        c = s.corpus_configs[0]
        assert c.allowed_roles == ["reader"]
        assert c.enable_graph_extraction is False
        assert c.graph_extraction_contract == "staged"
        assert c.graph_processing_mode == "many-to-one"
        assert c.graph_extraction_backend == "llm"

    def test_corpus_with_graph_extraction(self) -> None:
        corpus = [{
            "id": "legal",
            "enable_graph_extraction": True,
            "graph_ontology_path": "legal_contract.py",
            "graph_extraction_contract": "delta",
        }]
        s = make_settings(CORPUS_CONFIGS_JSON=json.dumps(corpus))
        c = s.corpus_configs[0]
        assert c.enable_graph_extraction is True
        assert c.graph_ontology_path == "legal_contract.py"
        assert c.graph_extraction_contract == "delta"

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(Exception):
            make_settings(CORPUS_CONFIGS_JSON="not-json")


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------


class TestDerivedHelpers:
    def test_age_graph_name_basic(self) -> None:
        s = make_settings(AGE_GRAPH_PREFIX="kg")
        name = s.age_graph_name("acme-corp", "hr-policies")
        assert name == "kg_acme_corp_hr_policies"

    def test_age_graph_name_colon_replaced(self) -> None:
        s = make_settings()
        name = s.age_graph_name("acme", "hr:policies")
        assert ":" not in name

    def test_masked_hides_password(self) -> None:
        s = make_settings(SMTP_PASSWORD="super_secret")
        masked = s.masked()
        assert masked["smtp_password"] == "***"

    def test_masked_hides_llm_api_key(self) -> None:
        s = make_settings(LLM_API_KEY="sk-real-key")
        masked = s.masked()
        assert masked["llm_api_key"] == "***"

    def test_masked_preserves_non_sensitive(self) -> None:
        s = make_settings()
        masked = s.masked()
        assert masked["llm_model"] == "llama3.2:3b"


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------


class TestConstraints:
    def test_embedding_dimension_min(self) -> None:
        with pytest.raises(Exception):
            make_settings(EMBEDDING_DIMENSION="10")  # below ge=64

    def test_confidence_threshold_range(self) -> None:
        with pytest.raises(Exception):
            make_settings(MIN_CONFIDENCE_SCORE="1.5")  # above le=1.0

    def test_max_retries_min(self) -> None:
        with pytest.raises(Exception):
            make_settings(MAX_RETRIES="0")  # below ge=1

    def test_valid_thresholds_accepted(self) -> None:
        s = make_settings(
            MIN_CONFIDENCE_SCORE="0.15",
            CONFIDENCE_WARN_THRESHOLD="0.50",
            RETRIEVAL_CONFIDENCE_THRESHOLD="2.0",
        )
        assert s.min_confidence_score == pytest.approx(0.15)
        assert s.confidence_warn_threshold == pytest.approx(0.50)
        assert s.retrieval_confidence_threshold == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# load_settings() singleton (isolated — clears cache before/after)
# ---------------------------------------------------------------------------


class TestLoadSettingsSingleton:
    def setup_method(self) -> None:
        load_settings.cache_clear()
        os.environ.setdefault("DATABASE_URL", REQUIRED_ENV["DATABASE_URL"])
        os.environ.setdefault("AGE_DATABASE_URL", REQUIRED_ENV["AGE_DATABASE_URL"])

    def teardown_method(self) -> None:
        load_settings.cache_clear()

    def test_returns_settings_instance(self) -> None:
        s = load_settings()
        assert isinstance(s, Settings)

    def test_singleton_same_object(self) -> None:
        s1 = load_settings()
        s2 = load_settings()
        assert s1 is s2

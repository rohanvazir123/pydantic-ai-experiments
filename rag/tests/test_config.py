"""Tests for configuration validation."""

from rag.config.settings import Settings, load_settings, mask_credential


class TestSettings:
    """Test Settings class."""

    def test_load_settings_returns_settings_instance(self):
        """Test that load_settings returns a Settings instance."""
        settings = load_settings()
        assert isinstance(settings, Settings)

    def test_settings_has_mongodb_uri(self):
        """Test that settings has mongodb_uri."""
        settings = load_settings()
        assert settings.mongodb_uri is not None
        assert len(settings.mongodb_uri) > 0

    def test_settings_has_mongodb_database(self):
        """Test that settings has mongodb_database."""
        settings = load_settings()
        assert settings.mongodb_database is not None
        assert (
            settings.mongodb_database == "rag_db" or len(settings.mongodb_database) > 0
        )

    def test_settings_has_llm_config(self):
        """Test that settings has LLM configuration."""
        settings = load_settings()
        assert settings.llm_provider is not None
        assert settings.llm_model is not None
        assert settings.llm_base_url is not None

    def test_settings_has_embedding_config(self):
        """Test that settings has embedding configuration."""
        settings = load_settings()
        assert settings.embedding_provider is not None
        assert settings.embedding_model is not None
        assert settings.embedding_base_url is not None
        assert settings.embedding_dimension > 0

    def test_settings_has_search_config(self):
        """Test that settings has search configuration."""
        settings = load_settings()
        assert settings.default_match_count > 0
        assert settings.max_match_count >= settings.default_match_count
        assert 0 <= settings.default_text_weight <= 1


class TestMaskCredential:
    """Test mask_credential function."""

    def test_mask_credential_short_string(self):
        """Test masking short credentials."""
        result = mask_credential("short")
        assert result == "***"

    def test_mask_credential_empty_string(self):
        """Test masking empty string."""
        result = mask_credential("")
        assert result == "***"

    def test_mask_credential_none(self):
        """Test masking None value."""
        result = mask_credential(None)
        assert result == "***"

    def test_mask_credential_long_string(self):
        """Test masking long credentials."""
        result = mask_credential("this-is-a-long-api-key")
        # First 4 chars + "..." + last 4 chars
        assert result == "this...-key"
        assert "is-a-long-api" not in result

    def test_mask_credential_exactly_8_chars(self):
        """Test masking exactly 8 character string."""
        result = mask_credential("12345678")
        assert result == "1234...5678"


class TestSettingsDefaults:
    """Test that settings have sensible defaults."""

    def test_default_embedding_dimension_for_ollama(self):
        """Test default embedding dimension for nomic-embed-text."""
        settings = load_settings()
        # If using nomic-embed-text, dimension should be 768
        if "nomic" in settings.embedding_model.lower():
            assert settings.embedding_dimension == 768

    def test_default_chunk_counts(self):
        """Test default chunk counts are reasonable."""
        settings = load_settings()
        assert settings.default_match_count >= 5
        assert settings.default_match_count <= 20
        assert settings.max_match_count <= 100

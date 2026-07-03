"""Deterministic unit tests for the shared config and knowledge-base runtime.

These touch no model and no network — they exercise the pure helpers in
``config`` and ``kb_tools`` directly.
"""

from __future__ import annotations

from pathlib import Path

import config
import kb_tools
import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.models.openai import OpenAIChatModel

KNOWLEDGE_DIR = (Path(__file__).parent.parent / "knowledge").resolve()


# --- config ---


def test_get_model_returns_ollama_backed_model() -> None:
    model = config.get_model("large")
    assert isinstance(model, OpenAIChatModel)
    # Model name is the resolved Ollama tag, not the tier name.
    assert model.model_name == config.MODEL_TIERS["large"]


def test_tiers_resolve_and_raw_tags_pass_through() -> None:
    assert config.resolve_model_name("nano") == config.MODEL_TIERS["nano"]
    assert config.resolve_model_name("small") == config.MODEL_TIERS["small"]
    # An unknown key is treated as a raw Ollama tag.
    assert config.resolve_model_name("mistral:7b") == "mistral:7b"


def test_get_model_does_no_network_io() -> None:
    # Constructing a model must not require a running daemon; it returns a model
    # whose name is a valid Ollama tag. (By default all tiers pin to `large` —
    # see test_tiers.py — so we assert on membership, not distinctness here.)
    m1 = config.get_model("nano")
    m2 = config.get_model("large")
    assert m1.model_name in config.MODEL_TIERS.values()
    assert m2.model_name == config.MODEL_TIERS["large"]


# --- kb_tools: listing ---


def test_list_files_finds_knowledge_base() -> None:
    listing = kb_tools.list_files_text(KNOWLEDGE_DIR)
    assert "policies/refund-policy.md" in listing
    assert "customers/cust_12345.md" in listing
    assert "templates/refund-confirmation.md" in listing


def test_list_files_glob_scopes_results() -> None:
    listing = kb_tools.list_files_text(KNOWLEDGE_DIR, "policies/*.md")
    assert "policies/refund-policy.md" in listing
    assert "customers/cust_12345.md" not in listing


def test_list_files_no_match_message() -> None:
    assert "No files match" in kb_tools.list_files_text(KNOWLEDGE_DIR, "nope/*.xyz")


# --- kb_tools: reading ---


def test_read_file_returns_content() -> None:
    text = kb_tools.read_file_text(KNOWLEDGE_DIR, "customers/cust_12345.md")
    assert "Sarah Johnson" in text
    assert "DUPLICATE" in text


def test_read_missing_file_raises_model_retry() -> None:
    with pytest.raises(ModelRetry):
        kb_tools.read_file_text(KNOWLEDGE_DIR, "customers/does-not-exist.md")


# --- kb_tools: path-traversal guard ---


@pytest.mark.parametrize(
    "escape",
    ["../config.py", "../../secrets.txt", "/etc/passwd", "policies/../../config.py"],
)
def test_safe_path_rejects_escapes(escape: str) -> None:
    with pytest.raises(ModelRetry):
        kb_tools.safe_path(KNOWLEDGE_DIR, escape)


def test_safe_path_allows_in_sandbox() -> None:
    p = kb_tools.safe_path(KNOWLEDGE_DIR, "policies/refund-policy.md")
    assert p.is_file()
    assert KNOWLEDGE_DIR in p.parents


def test_read_file_traversal_blocked() -> None:
    with pytest.raises(ModelRetry):
        kb_tools.read_file_text(KNOWLEDGE_DIR, "../config.py")


# --- kb_tools: search ---


def test_search_files_finds_term_with_location() -> None:
    hits = kb_tools.search_files_text(KNOWLEDGE_DIR, "manager approval")
    assert "refund-policy.md" in hits
    assert ":" in hits  # path:line format


def test_search_files_no_match_message() -> None:
    assert "No matches" in kb_tools.search_files_text(KNOWLEDGE_DIR, "zzz-nonexistent")


# --- kb_tools: simulated billing API ---


def test_payment_gateway_reports_eligibility() -> None:
    out = kb_tools.payment_gateway_text("2025-02-01", 49.99)
    assert "Refund eligible: YES" in out
    assert "49.99" in out


def test_refund_text_includes_details() -> None:
    out = kb_tools.refund_text(49.99, "duplicate charge", "cust_12345")
    assert "cust_12345" in out
    assert "49.99" in out
    assert "ref_" in out

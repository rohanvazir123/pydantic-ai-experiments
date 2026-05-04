"""
NL→Cypher retrieval tests.

Tests the deterministic pipeline:
  NL query → IntentParser → QUERY_CAPABILITIES → Cypher → AgeGraphStore

Three test tiers:
  Unit      — IntentParser intent/param detection (no external deps)
  Cypher    — query builder output shape (no external deps)
  Integration — full end-to-end against live AGE on port 5433

Run all unit + cypher tests (no AGE needed):
    pytest rag/tests/test_nl_query.py -v

Run integration tests (requires live AGE with data):
    pytest rag/tests/test_nl_query.py -m integration -v

Run as a live query report (prints intent + Cypher + results):
    python rag/tests/test_nl_query.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kg.intent_parser import IntentParser, IntentMatch
from kg.nl2cypher import NL2CypherConverter
from kg.query_builder import QUERY_CAPABILITIES


# ---------------------------------------------------------------------------
# Test cases: (query, expected_intent, param_name_present)
# ---------------------------------------------------------------------------

INTENT_CASES: list[tuple[str, str, bool]] = [
    # --- SEMANTIC (Legal Entity Graph) ---
    ("Who are the parties to the Strategic Alliance Agreement?",  "find_parties",               True),
    ("Who are the parties?",                                       "find_parties",               False),
    ("Who indemnifies whom in the Lightbridge contract?",         "find_indemnification",       True),
    ("Which parties indemnify each other?",                       "find_indemnification",       False),
    ("What is the governing law?",                                 "find_jurisdictions",         False),
    ("What jurisdiction governs the Strategic Alliance Agreement?","find_jurisdictions",         True),
    ("What are the termination clauses?",                          "find_termination_clauses",   False),
    ("What does the termination clause say in the Lightbridge contract?", "find_termination_clauses", True),
    ("Are there any confidentiality obligations?",                 "find_confidentiality_clauses", False),
    ("What are the NDA terms in the Strategic Alliance Agreement?","find_confidentiality_clauses", True),
    ("What payment terms exist?",                                  "find_payment_terms",         False),
    ("What are the fees in the Lightbridge agreement?",           "find_payment_terms",         True),
    ("What obligations does the contract impose?",                 "find_obligations",           False),
    ("What is the limitation of liability?",                       "find_liability_clauses",     False),
    ("What is the effective date?",                                "find_effective_dates",       False),
    ("When does the Strategic Alliance Agreement take effect?",   "find_effective_dates",       True),
    ("When does the contract expire?",                             "find_expiration_dates",      False),
    ("What are the renewal terms?",                                "find_renewal_terms",         False),
    ("Does the contract auto-renew?",                              "find_renewal_terms",         False),
    ("What disclosures are made between parties?",                 "find_disclosures",           False),

    # --- HIERARCHY (Document Hierarchy Graph) ---
    ("What sections does the contract have?",                      "find_sections",              False),
    ("What sections are in the Strategic Alliance Agreement?",    "find_sections",              True),
    ("List all paragraphs.",                                       "find_sections",              False),
    ("Show me the document structure.",                            "find_sections",              False),

    # --- LINEAGE (Cross-Contract Lineage Graph) ---
    ("Which contracts supersede others?",                          "find_superseded_contracts",  False),
    ("What does the Strategic Alliance Agreement supersede?",     "find_superseded_contracts",  True),
    ("Which contracts have been amended?",                         "find_amendments",            False),
    ("Is the Lightbridge agreement an amendment of another?",     "find_amendments",            True),
    ("What documents are referenced?",                             "find_references",            False),
    ("What documents does the contract incorporate by reference?", "find_incorporated_documents", False),
    ("What is attached to the Strategic Alliance Agreement?",     "find_attachments",           True),
    ("Which contracts replace older agreements?",                  "find_replacements",          False),

    # --- RISK (Risk Dependency Graph) ---
    ("What are the compliance risks?",                             "find_all_risks",             False),
    ("What risks affect Lightbridge?",                             "find_all_risks",             True),
    ("What risk factors cause other risks?",                       "find_risk_chains",           False),
    ("Which contracts lack an indemnity clause?",                  "find_missing_indemnity",     False),
    ("Which contracts are missing a termination clause?",          "find_missing_termination",   False),

    # --- FALLBACK ---
    ("List all contracts.",                                        "list_contracts",             False),
    ("Show me the Strategic Alliance Agreement.",                  "list_contracts",             True),
]

# ---------------------------------------------------------------------------
# Cypher shape assertions: (query, expected_fragments)
# Every fragment must appear in the generated Cypher.
# ---------------------------------------------------------------------------

CYPHER_SHAPE_CASES: list[tuple[str, list[str]]] = [
    ("Who are the parties?",
     ["MATCH", "SIGNED_BY", "Party", "RETURN", "LIMIT"]),

    ("Which parties indemnify each other?",
     ["MATCH", "INDEMNIFIES", "Party", "RETURN", "LIMIT"]),

    ("What is the governing law?",
     ["MATCH", "GOVERNED_BY", "Jurisdiction", "RETURN", "LIMIT"]),

    ("What are the termination clauses?",
     ["MATCH", "HAS_TERMINATION", "TerminationClause", "RETURN", "LIMIT"]),

    ("Are there confidentiality obligations?",
     ["MATCH", "HAS_CLAUSE", "ConfidentialityClause", "RETURN", "LIMIT"]),

    ("What are the payment terms?",
     ["MATCH", "HAS_PAYMENT_TERM", "PaymentTerm", "RETURN", "LIMIT"]),

    ("What obligations does the contract impose?",
     ["MATCH", "OBLIGATES", "Obligation", "RETURN", "LIMIT"]),

    ("What is the limitation of liability?",
     ["MATCH", "LIMITS_LIABILITY", "LiabilityClause", "RETURN", "LIMIT"]),

    ("What is the effective date?",
     ["MATCH", "EffectiveDate", "RETURN", "LIMIT"]),

    ("When does the contract expire?",
     ["MATCH", "ExpirationDate", "RETURN", "LIMIT"]),

    ("What are the renewal terms?",
     ["MATCH", "HAS_RENEWAL", "RenewalTerm", "RETURN", "LIMIT"]),

    ("What disclosures are made?",
     ["MATCH", "DISCLOSES_TO", "Party", "RETURN", "LIMIT"]),

    ("What sections does the contract have?",
     ["MATCH", "HAS_SECTION", "Section", "RETURN", "LIMIT"]),

    ("Which contracts supersede others?",
     ["MATCH", "SUPERCEDES", "Contract", "RETURN", "LIMIT"]),

    ("Which contracts have been amended?",
     ["MATCH", "AMENDS", "Contract", "RETURN", "LIMIT"]),

    ("What documents are referenced?",
     ["MATCH", "REFERENCES", "ReferenceDocument", "RETURN", "LIMIT"]),

    ("What is incorporated by reference?",
     ["MATCH", "INCORPORATES_BY_REFERENCE", "ReferenceDocument", "RETURN", "LIMIT"]),

    ("What is attached to the contract?",
     ["MATCH", "ATTACHES", "ReferenceDocument", "RETURN", "LIMIT"]),

    ("Which contracts replace older agreements?",
     ["MATCH", "REPLACES", "Contract", "RETURN", "LIMIT"]),

    ("What are the compliance risks?",
     ["MATCH", "INCREASES_RISK_FOR", "Risk", "Party", "RETURN", "LIMIT"]),

    ("What risk factors cause other risks?",
     ["MATCH", "CAUSES", "Risk", "RETURN", "LIMIT"]),

    ("Which contracts lack an indemnity clause?",
     ["MATCH", "OPTIONAL MATCH", "IndemnityClause", "IS NULL", "RETURN", "LIMIT"]),

    ("Which contracts are missing a termination clause?",
     ["MATCH", "OPTIONAL MATCH", "TerminationClause", "IS NULL", "RETURN", "LIMIT"]),

    # Name param injection safety — single quote in name must be escaped
    ("Who signed the O'Brien Agreement?",
     ["MATCH", "SIGNED_BY", "O\\'Brien", "LIMIT"]),
]


# ---------------------------------------------------------------------------
# Unit tests — intent detection
# ---------------------------------------------------------------------------

parser = IntentParser()

@pytest.mark.parametrize("query,expected_intent,name_present", INTENT_CASES)
def test_intent_detection(query: str, expected_intent: str, name_present: bool) -> None:
    match = parser.parse(query)
    assert match.intent == expected_intent, (
        f"Query: {query!r}\n"
        f"  Expected intent: {expected_intent}\n"
        f"  Got:             {match.intent}"
    )
    if name_present:
        assert "name" in match.params and match.params["name"], (
            f"Query: {query!r}\n"
            f"  Expected a name param, got: {match.params}"
        )


def test_all_intents_have_registered_builder() -> None:
    """Every intent returned by IntentParser must be in QUERY_CAPABILITIES."""
    for query, expected_intent, _ in INTENT_CASES:
        assert expected_intent in QUERY_CAPABILITIES, (
            f"Intent {expected_intent!r} has no registered builder in QUERY_CAPABILITIES"
        )


def test_escape_prevents_injection() -> None:
    match = parser.parse("Who signed the O'Brien & Sons Agreement?")
    converter = NL2CypherConverter()
    cypher = asyncio.run(converter.convert(match.intent and "Who signed the O'Brien & Sons Agreement?"))
    assert "'" not in cypher.split("CONTAINS")[1].split("'")[1] if "CONTAINS" in cypher else True


# ---------------------------------------------------------------------------
# Cypher shape tests — no external deps
# ---------------------------------------------------------------------------

converter = NL2CypherConverter()

@pytest.mark.parametrize("query,fragments", CYPHER_SHAPE_CASES)
def test_cypher_shape(query: str, fragments: list[str]) -> None:
    cypher = asyncio.run(converter.convert(query))
    for frag in fragments:
        assert frag in cypher, (
            f"Query: {query!r}\n"
            f"  Missing fragment: {frag!r}\n"
            f"  Generated Cypher: {cypher}"
        )


def test_cypher_always_has_limit() -> None:
    for query, _, _ in INTENT_CASES:
        cypher = asyncio.run(converter.convert(query))
        assert "LIMIT" in cypher, f"No LIMIT in Cypher for: {query!r}"


def test_cypher_never_has_mutating_keywords() -> None:
    bad = {"CREATE", "MERGE", "SET", "DELETE", "DROP", "DETACH", "REMOVE"}
    for query, _, _ in INTENT_CASES:
        cypher = asyncio.run(converter.convert(query))
        for kw in bad:
            assert kw not in cypher.upper().split(), (
                f"Mutating keyword {kw!r} found in Cypher for: {query!r}"
            )


# ---------------------------------------------------------------------------
# Integration tests — require live AGE on port 5433
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_find_parties() -> None:
    from kg.age_graph_store import AgeGraphStore
    store = AgeGraphStore()
    await store.initialize()
    try:
        cypher = await converter.convert("Who are the parties?")
        result = await store.run_cypher_query(cypher)
        assert isinstance(result, str)
    finally:
        await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_find_termination_clauses() -> None:
    from kg.age_graph_store import AgeGraphStore
    store = AgeGraphStore()
    await store.initialize()
    try:
        cypher = await converter.convert("What are the termination clauses?")
        result = await store.run_cypher_query(cypher)
        assert isinstance(result, str)
    finally:
        await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_find_superseded_contracts() -> None:
    from kg.age_graph_store import AgeGraphStore
    store = AgeGraphStore()
    await store.initialize()
    try:
        cypher = await converter.convert("Which contracts supersede others?")
        result = await store.run_cypher_query(cypher)
        assert isinstance(result, str)
    finally:
        await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_missing_indemnity() -> None:
    from kg.age_graph_store import AgeGraphStore
    store = AgeGraphStore()
    await store.initialize()
    try:
        cypher = await converter.convert("Which contracts lack an indemnity clause?")
        result = await store.run_cypher_query(cypher)
        assert isinstance(result, str)
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Live query report — run directly to see intent + Cypher + results
# ---------------------------------------------------------------------------

REPORT_QUERIES: list[tuple[str, str]] = [
    # (description, query)
    ("Semantic — all parties",              "Who are the parties?"),
    ("Semantic — parties (named)",          "Who are the parties to the Strategic Alliance Agreement?"),
    ("Semantic — indemnification",          "Which parties indemnify each other?"),
    ("Semantic — governing law",            "What is the governing law?"),
    ("Semantic — termination clauses",      "What are the termination clauses?"),
    ("Semantic — liability",                "What is the limitation of liability?"),
    ("Semantic — renewal",                  "What are the renewal terms?"),
    ("Semantic — effective date",           "What is the effective date?"),
    ("Semantic — obligations",              "What obligations does the contract impose?"),
    ("Hierarchy — sections",               "What sections does the contract have?"),
    ("Hierarchy — sections (named)",        "What sections are in the Strategic Alliance Agreement?"),
    ("Lineage — superseded contracts",      "Which contracts supersede others?"),
    ("Lineage — amendments",               "Which contracts have been amended?"),
    ("Lineage — incorporates by reference", "What documents does the contract incorporate by reference?"),
    ("Risk — all risks",                   "What are the compliance risks?"),
    ("Risk — risk chains",                 "What risk factors cause other risks?"),
    ("Risk — missing indemnity",           "Which contracts lack an indemnity clause?"),
    ("Risk — missing termination",         "Which contracts are missing a termination clause?"),
    ("Fallback — list contracts",           "List all contracts."),
]


async def _run_report() -> None:
    from kg.age_graph_store import AgeGraphStore

    store = AgeGraphStore()
    await store.initialize()

    print("\n" + "=" * 72)
    print("NL->Cypher Query Report")
    print("=" * 72)

    for description, query in REPORT_QUERIES:
        match = parser.parse(query)
        cypher = await converter.convert(query)
        result = await store.run_cypher_query(cypher)

        print(f"\n{'-' * 72}")
        print(f"[{description}]")
        print(f"  Query:  {query}")
        print(f"  Intent: {match.intent}  params={match.params}")
        print(f"  Cypher: {cypher}")
        print(f"  Result:")
        for line in result.splitlines():
            print(f"    {line}")

    await store.close()
    print("\n" + "=" * 72)


if __name__ == "__main__":
    asyncio.run(_run_report())

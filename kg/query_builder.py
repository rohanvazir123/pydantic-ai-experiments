"""
Deterministic Cypher query builders.

One builder per intent.  Each builder:
  - Accepts a params dict extracted by IntentParser.
  - Returns a raw openCypher MATCH..RETURN..LIMIT string.
  - Never calls an LLM.
  - Escapes every user-supplied value with _esc() before interpolation.

run_cypher_query() in AgeGraphStore wraps the returned string inside
ag_catalog.cypher('legal_graph', $$..$$) — builders do not include that.

QUERY_CAPABILITIES maps intent name → builder callable.
"""
from __future__ import annotations

from typing import Callable


def _esc(value: str) -> str:
    """Escape a string value for safe embedding in a Cypher single-quoted literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


# ---------------------------------------------------------------------------
# ENTITY graph builders
# ---------------------------------------------------------------------------

def build_parties_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (c:Contract)-[:SIGNED_BY]->(p:Party)"
            f" WHERE c.name CONTAINS '{_esc(name)}'"
            f" RETURN c.name AS contract, p.name AS party"
            f" LIMIT 20"
        )
    return (
        "MATCH (c:Contract)-[:SIGNED_BY]->(p:Party)"
        " RETURN c.name AS contract, p.name AS party"
        " LIMIT 50"
    )


def build_indemnification_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (p1:Party)-[:INDEMNIFIES]->(p2:Party)"
            f" WHERE p1.name CONTAINS '{_esc(name)}' OR p2.name CONTAINS '{_esc(name)}'"
            f" RETURN p1.name AS indemnifier, p2.name AS indemnified"
            f" LIMIT 20"
        )
    return (
        "MATCH (p1:Party)-[:INDEMNIFIES]->(p2:Party)"
        " RETURN p1.name AS indemnifier, p2.name AS indemnified"
        " LIMIT 50"
    )


def build_jurisdiction_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (c:Contract)-[:GOVERNED_BY]->(j:Jurisdiction)"
            f" WHERE c.name CONTAINS '{_esc(name)}'"
            f" RETURN c.name AS contract, j.name AS jurisdiction"
            f" LIMIT 20"
        )
    return (
        "MATCH (c:Contract)-[:GOVERNED_BY]->(j:Jurisdiction)"
        " RETURN c.name AS contract, j.name AS jurisdiction"
        " LIMIT 50"
    )


def build_termination_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (c:Contract)-[:HAS_TERMINATION]->(t:TerminationClause)"
            f" WHERE c.name CONTAINS '{_esc(name)}'"
            f" RETURN c.name AS contract, t.name AS clause"
            f" LIMIT 20"
        )
    return (
        "MATCH (c:Contract)-[:HAS_TERMINATION]->(t:TerminationClause)"
        " RETURN c.name AS contract, t.name AS clause"
        " LIMIT 50"
    )


def build_confidentiality_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (c:Contract)-[:HAS_CLAUSE]->(cl:ConfidentialityClause)"
            f" WHERE c.name CONTAINS '{_esc(name)}'"
            f" RETURN c.name AS contract, cl.name AS clause"
            f" LIMIT 20"
        )
    return (
        "MATCH (c:Contract)-[:HAS_CLAUSE]->(cl:ConfidentialityClause)"
        " RETURN c.name AS contract, cl.name AS clause"
        " LIMIT 50"
    )


def build_payment_terms_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (c:Contract)-[:HAS_PAYMENT_TERM]->(pt:PaymentTerm)"
            f" WHERE c.name CONTAINS '{_esc(name)}'"
            f" RETURN c.name AS contract, pt.name AS term"
            f" LIMIT 20"
        )
    return (
        "MATCH (c:Contract)-[:HAS_PAYMENT_TERM]->(pt:PaymentTerm)"
        " RETURN c.name AS contract, pt.name AS term"
        " LIMIT 50"
    )


def build_obligations_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (c:Contract)-[:OBLIGATES]->(o:Obligation)"
            f" WHERE c.name CONTAINS '{_esc(name)}'"
            f" RETURN c.name AS contract, o.name AS obligation"
            f" LIMIT 20"
        )
    return (
        "MATCH (c:Contract)-[:OBLIGATES]->(o:Obligation)"
        " RETURN c.name AS contract, o.name AS obligation"
        " LIMIT 50"
    )


def build_liability_query(intent: dict[str, str]) -> str:
    liability_type = intent.get("liability_type")
    name = intent.get("name")
    clauses: list[str] = []
    if name:
        clauses.append(f"c.name CONTAINS '{_esc(name)}'")
    if liability_type:
        clauses.append(f"l.subtype = '{_esc(liability_type)}'")
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return (
        f"MATCH (c:Contract)-[:LIMITS_LIABILITY]->(l:LiabilityClause)"
        f"{where}"
        f" RETURN c.name AS contract, l.name AS clause, l.subtype AS type"
        f" LIMIT {'20' if clauses else '50'}"
    )


def build_effective_dates_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (c:Contract)-[:HAS_CLAUSE]->(d:EffectiveDate)"
            f" WHERE c.name CONTAINS '{_esc(name)}'"
            f" RETURN c.name AS contract, d.name AS effective_date"
            f" LIMIT 20"
        )
    return (
        "MATCH (c:Contract)-[:HAS_CLAUSE]->(d:EffectiveDate)"
        " RETURN c.name AS contract, d.name AS effective_date"
        " LIMIT 50"
    )


def build_expiration_dates_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (c:Contract)-[:HAS_CLAUSE]->(d:ExpirationDate)"
            f" WHERE c.name CONTAINS '{_esc(name)}'"
            f" RETURN c.name AS contract, d.name AS expiration_date"
            f" LIMIT 20"
        )
    return (
        "MATCH (c:Contract)-[:HAS_CLAUSE]->(d:ExpirationDate)"
        " RETURN c.name AS contract, d.name AS expiration_date"
        " LIMIT 50"
    )


def build_renewal_terms_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (c:Contract)-[:HAS_RENEWAL]->(r:RenewalTerm)"
            f" WHERE c.name CONTAINS '{_esc(name)}'"
            f" RETURN c.name AS contract, r.name AS renewal_term"
            f" LIMIT 20"
        )
    return (
        "MATCH (c:Contract)-[:HAS_RENEWAL]->(r:RenewalTerm)"
        " RETURN c.name AS contract, r.name AS renewal_term"
        " LIMIT 50"
    )


def build_disclosures_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (p1:Party)-[:DISCLOSES_TO]->(p2:Party)"
            f" WHERE p1.name CONTAINS '{_esc(name)}' OR p2.name CONTAINS '{_esc(name)}'"
            f" RETURN p1.name AS discloser, p2.name AS recipient"
            f" LIMIT 20"
        )
    return (
        "MATCH (p1:Party)-[:DISCLOSES_TO]->(p2:Party)"
        " RETURN p1.name AS discloser, p2.name AS recipient"
        " LIMIT 50"
    )


# ---------------------------------------------------------------------------
# LINEAGE graph builders
# ---------------------------------------------------------------------------

def build_superseded_contracts_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (c1:Contract)-[:SUPERCEDES]->(c2:Contract)"
            f" WHERE c1.name CONTAINS '{_esc(name)}' OR c2.name CONTAINS '{_esc(name)}'"
            f" RETURN c1.name AS superseder, c2.name AS superseded"
            f" LIMIT 20"
        )
    return (
        "MATCH (c1:Contract)-[:SUPERCEDES]->(c2:Contract)"
        " RETURN c1.name AS superseder, c2.name AS superseded"
        " LIMIT 50"
    )


def build_amendment_chain_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (c1:Contract)-[:AMENDS]->(c2:Contract)"
            f" WHERE c1.name CONTAINS '{_esc(name)}' OR c2.name CONTAINS '{_esc(name)}'"
            f" RETURN c1.name AS amendment, c2.name AS original"
            f" LIMIT 20"
        )
    return (
        "MATCH (c1:Contract)-[:AMENDS]->(c2:Contract)"
        " RETURN c1.name AS amendment, c2.name AS original"
        " LIMIT 50"
    )


def build_references_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (c:Contract)-[:REFERENCES]->(r:ReferenceDocument)"
            f" WHERE c.name CONTAINS '{_esc(name)}'"
            f" RETURN c.name AS contract, r.name AS referenced_doc"
            f" LIMIT 20"
        )
    return (
        "MATCH (c:Contract)-[:REFERENCES]->(r:ReferenceDocument)"
        " RETURN c.name AS contract, r.name AS referenced_doc"
        " LIMIT 50"
    )


def build_incorporated_documents_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (c:Contract)-[:INCORPORATES_BY_REFERENCE]->(r:ReferenceDocument)"
            f" WHERE c.name CONTAINS '{_esc(name)}'"
            f" RETURN c.name AS contract, r.name AS incorporated_doc"
            f" LIMIT 20"
        )
    return (
        "MATCH (c:Contract)-[:INCORPORATES_BY_REFERENCE]->(r:ReferenceDocument)"
        " RETURN c.name AS contract, r.name AS incorporated_doc"
        " LIMIT 50"
    )


def build_attachments_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (c:Contract)-[:ATTACHES]->(r:ReferenceDocument)"
            f" WHERE c.name CONTAINS '{_esc(name)}'"
            f" RETURN c.name AS contract, r.name AS attachment"
            f" LIMIT 20"
        )
    return (
        "MATCH (c:Contract)-[:ATTACHES]->(r:ReferenceDocument)"
        " RETURN c.name AS contract, r.name AS attachment"
        " LIMIT 50"
    )


def build_replacements_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (c1:Contract)-[:REPLACES]->(c2:Contract)"
            f" WHERE c1.name CONTAINS '{_esc(name)}' OR c2.name CONTAINS '{_esc(name)}'"
            f" RETURN c1.name AS replacement, c2.name AS replaced"
            f" LIMIT 20"
        )
    return (
        "MATCH (c1:Contract)-[:REPLACES]->(c2:Contract)"
        " RETURN c1.name AS replacement, c2.name AS replaced"
        " LIMIT 50"
    )


# ---------------------------------------------------------------------------
# RISK graph builders
# ---------------------------------------------------------------------------

def build_all_risks_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (r:Risk)-[:INCREASES_RISK_FOR]->(p:Party)"
            f" WHERE p.name CONTAINS '{_esc(name)}'"
            f" RETURN r.name AS risk, r.risk_type AS type, r.severity AS severity, p.name AS party"
            f" LIMIT 20"
        )
    return (
        "MATCH (r:Risk)-[:INCREASES_RISK_FOR]->(p:Party)"
        " RETURN r.name AS risk, r.risk_type AS type, r.severity AS severity, p.name AS party"
        " LIMIT 50"
    )


def build_risk_chains_query(intent: dict[str, str]) -> str:
    return (
        "MATCH (r1:Risk)-[:CAUSES]->(r2:Risk)"
        " RETURN r1.name AS cause, r2.name AS effect"
        " LIMIT 50"
    )


def build_missing_indemnity_query(intent: dict[str, str]) -> str:
    # AGE does not support label filters inside WHERE NOT (pattern).
    # Use OPTIONAL MATCH + IS NULL instead.
    return (
        "MATCH (c:Contract)"
        " OPTIONAL MATCH (c)-[:HAS_CLAUSE]->(ic:IndemnityClause)"
        " WITH c, ic WHERE ic IS NULL"
        " RETURN c.name AS contract_without_indemnity, c.document_id AS document"
        " LIMIT 50"
    )


def build_missing_termination_query(intent: dict[str, str]) -> str:
    return (
        "MATCH (c:Contract)"
        " OPTIONAL MATCH (c)-[:HAS_TERMINATION]->(tc:TerminationClause)"
        " WITH c, tc WHERE tc IS NULL"
        " RETURN c.name AS contract_without_termination, c.document_id AS document"
        " LIMIT 50"
    )


# ---------------------------------------------------------------------------
# HIERARCHY graph builders
# ---------------------------------------------------------------------------

def build_sections_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (c:Contract)-[:HAS_SECTION]->(s:Section)"
            f" WHERE c.name CONTAINS '{_esc(name)}'"
            f" RETURN c.name AS contract, s.name AS section, s.sequence_number AS seq"
            f" LIMIT 50"
        )
    return (
        "MATCH (c:Contract)-[:HAS_SECTION]->(s:Section)"
        " RETURN c.name AS contract, s.name AS section, s.sequence_number AS seq"
        " LIMIT 50"
    )


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def build_list_contracts_query(intent: dict[str, str]) -> str:
    if name := intent.get("name"):
        return (
            f"MATCH (c:Contract)"
            f" WHERE c.name CONTAINS '{_esc(name)}'"
            f" RETURN c.name AS contract, c.document_id AS document"
            f" LIMIT 20"
        )
    return (
        "MATCH (c:Contract)"
        " RETURN c.name AS contract, c.document_id AS document"
        " LIMIT 50"
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

QueryBuilder = Callable[[dict[str, str]], str]

QUERY_CAPABILITIES: dict[str, QueryBuilder] = {
    # ENTITY
    "find_parties":                  build_parties_query,
    "find_indemnification":          build_indemnification_query,
    "find_jurisdictions":            build_jurisdiction_query,
    "find_termination_clauses":      build_termination_query,
    "find_confidentiality_clauses":  build_confidentiality_query,
    "find_payment_terms":            build_payment_terms_query,
    "find_obligations":              build_obligations_query,
    "find_liability_clauses":        build_liability_query,
    "find_effective_dates":          build_effective_dates_query,
    "find_expiration_dates":         build_expiration_dates_query,
    "find_renewal_terms":            build_renewal_terms_query,
    "find_disclosures":              build_disclosures_query,
    # LINEAGE
    "find_superseded_contracts":     build_superseded_contracts_query,
    "find_amendments":               build_amendment_chain_query,
    "find_references":               build_references_query,
    "find_incorporated_documents":   build_incorporated_documents_query,
    "find_attachments":              build_attachments_query,
    "find_replacements":             build_replacements_query,
    # RISK
    "find_all_risks":                build_all_risks_query,
    "find_risk_chains":              build_risk_chains_query,
    "find_missing_indemnity":        build_missing_indemnity_query,
    "find_missing_termination":      build_missing_termination_query,
    # HIERARCHY
    "find_sections":                 build_sections_query,
    # FALLBACK
    "list_contracts":                build_list_contracts_query,
}

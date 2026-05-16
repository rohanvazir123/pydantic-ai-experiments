"""
Compact graph schemas for NL→Cypher routing.

Each schema describes one logical subgraph of the physical Apache AGE graph
(legal_graph).  All four schemas share the same graph; they are distinguished
by vertex labels and relationship types, not by separate AGE graph objects.

Source of truth: docs/kg/KG_PIPELINE.md

Usage:
    from kg.legal.schemas import GraphType, get_schema

    schema = get_schema([GraphType.ENTITY, GraphType.LINEAGE])
    cypher = await converter.convert(question, schema)
"""

from __future__ import annotations

from enum import Enum


class GraphType(str, Enum):
    ENTITY    = "entity"
    HIERARCHY = "hierarchy"
    LINEAGE   = "lineage"
    RISK      = "risk"


# ---------------------------------------------------------------------------
# Compact schemas — token-bounded, no prose examples
# ---------------------------------------------------------------------------

ENTITY_SCHEMA = """\
=== Legal Entity Graph ===
All vertices carry: uuid, name, label, document_id, confidence

Vertex labels:
  (:Party)                 contracting party or organisation
  (:Contract)              the agreement itself
  (:Jurisdiction)          governing law state or country
  (:EffectiveDate)         date contract takes effect
  (:ExpirationDate)        date contract expires
  (:RenewalTerm)           automatic renewal period
  (:LiabilityClause)       limitation-of-liability provision
  (:IndemnityClause)       indemnification obligation
  (:PaymentTerm)           fees, royalties, revenue sharing
  (:ConfidentialityClause) NDA / confidentiality obligation
  (:TerminationClause)     termination-for-cause or convenience
  (:GoverningLawClause)    choice-of-law provision
  (:Obligation)            a duty a party must perform
  (:Clause)                any other contractual provision (fallback)

Edge types (source → target):
  SIGNED_BY         Contract → Party
  GOVERNED_BY       Contract → Jurisdiction
  INDEMNIFIES       Party → Party
  HAS_TERMINATION   Contract → TerminationClause
  HAS_RENEWAL       Contract → RenewalTerm
  HAS_PAYMENT_TERM  Contract → PaymentTerm
  OBLIGATES         Contract → Obligation
  LIMITS_LIABILITY  Contract → LiabilityClause
  DISCLOSES_TO      Party → Party
  HAS_CLAUSE        Contract → Clause"""

HIERARCHY_SCHEMA = """\
=== Document Hierarchy Graph ===
All vertices carry: uuid, name, label, document_id

Vertex labels:
  (:Contract) the agreement
  (:Section)  named section (also: sequence_number, title)
  (:Clause)   specific provision (also: sequence_number, title)

Edge types (source → target):
  HAS_SECTION  Contract → Section
  HAS_CLAUSE   Section → Clause
  HAS_CHUNK    Clause → Chunk"""

LINEAGE_SCHEMA = """\
=== Cross-Contract Lineage Graph ===
All vertices carry: uuid, name, label, document_id

Vertex labels:
  (:Contract)          the agreement
  (:ReferenceDocument) external document referenced by the contract

Edge types (source → target):
  REFERENCES                Contract → ReferenceDocument
  AMENDS                    Contract → Contract
  SUPERCEDES                Contract → Contract
  REPLACES                  Contract → Contract
  ATTACHES                  Contract → ReferenceDocument
  INCORPORATES_BY_REFERENCE Contract → ReferenceDocument"""

RISK_SCHEMA = """\
=== Risk Dependency Graph ===
All vertices carry: uuid, name, label, document_id

Vertex labels:
  (:Risk)  risk factor or compliance gap (also: risk_type, severity)
  (:Party) contracting party (shared label with Legal Entity Graph)

Edge types (source → target):
  INCREASES_RISK_FOR  Risk → Party
  CAUSES              Risk → Risk"""


_SCHEMA_MAP: dict[GraphType, str] = {
    GraphType.ENTITY:    ENTITY_SCHEMA,
    GraphType.HIERARCHY: HIERARCHY_SCHEMA,
    GraphType.LINEAGE:   LINEAGE_SCHEMA,
    GraphType.RISK:      RISK_SCHEMA,
}


def get_schema(graph_types: list[GraphType]) -> str:
    """Return combined compact schema string for the given graph types."""
    parts = [_SCHEMA_MAP[gt] for gt in graph_types if gt in _SCHEMA_MAP]
    return "\n\n".join(parts)

"""
Single source of truth for KG ontology constants.

All other modules import VALID_LABELS, VALID_REL_TYPES, ENTITY_TYPE_MAP,
and RELATIONSHIP_MAP from here.  Do not redefine them elsewhere.
"""

# ---------------------------------------------------------------------------
# Vertex labels (18 canonical types)
# ---------------------------------------------------------------------------

VALID_LABELS: frozenset[str] = frozenset({
    "Contract",
    "Section",
    "Clause",
    "Party",
    "Jurisdiction",
    "EffectiveDate",
    "ExpirationDate",
    "RenewalTerm",
    "LiabilityClause",
    "IndemnityClause",
    "PaymentTerm",
    "ConfidentialityClause",
    "TerminationClause",
    "GoverningLawClause",
    "Obligation",
    "Risk",
    "Amendment",
    "ReferenceDocument",
})

# ---------------------------------------------------------------------------
# Relationship types (all paths: legal entity, hierarchy, lineage, risk)
# ---------------------------------------------------------------------------

VALID_REL_TYPES: frozenset[str] = frozenset({
    # --- legal entity graph (semantic) ---
    "PARTY_TO",                   # entity → Contract  (CUAD annotation path)
    "SIGNED_BY",                  # Contract → Party   (LLM extraction path)
    "GOVERNED_BY",                # Contract → Jurisdiction
    "GOVERNED_BY_LAW",            # entity → Contract  (CUAD annotation path)
    "INDEMNIFIES",                # Party → Party
    "HAS_TERMINATION",            # Contract → TerminationClause
    "HAS_RENEWAL",                # Contract → RenewalTerm
    "HAS_PAYMENT_TERM",           # Contract → PaymentTerm
    "HAS_LICENSE",                # entity → Contract  (CUAD); Contract → LicenseClause (LLM)
    "HAS_RESTRICTION",            # entity → Contract  (CUAD); Contract → RestrictionClause (LLM)
    "HAS_IP_CLAUSE",              # entity → Contract  (CUAD); Contract → IPClause (LLM)
    "HAS_LIABILITY",              # entity → Contract  (CUAD); Contract → LiabilityClause (LLM)
    "HAS_PAYMENT",                # Contract → PaymentTerm (alt)
    "HAS_OBLIGATION",             # Contract → Obligation
    "HAS_CLAUSE",                 # Contract → Clause  (generic / fallback)
    "HAS_DATE",                   # Contract → Date entity
    "EFFECTIVE_ON",               # Contract → EffectiveDate
    "EXPIRES_ON",                 # Contract → ExpirationDate
    "OBLIGATES",                  # Contract → Obligation
    "LIMITS_LIABILITY",           # Contract → LiabilityClause
    "DISCLOSES_TO",               # Party → Party
    "GRANTS_LICENSE_TO",          # Party → Party (via license)
    "OWES_OBLIGATION_TO",         # Party → Party
    "ASSIGNS_IP_TO",              # Party → Party
    "CAN_TERMINATE",              # Party → Contract
    # --- document hierarchy ---
    "HAS_SECTION",                # Contract → Section
    "HAS_CHUNK",                  # Clause → Chunk
    # --- cross-contract lineage ---
    "REFERENCES",                 # Contract → ReferenceDocument
    "AMENDS",                     # Contract → Contract
    "SUPERCEDES",                 # Contract → Contract
    "REPLACES",                   # Contract → Contract
    "ATTACHES",                   # Contract → ReferenceDocument
    "INCORPORATES_BY_REFERENCE",  # Contract → ReferenceDocument
    # --- risk dependency ---
    "INCREASES_RISK_FOR",         # Risk → Party
    "CAUSES",                     # Risk → Risk
})

# ---------------------------------------------------------------------------
# CUAD question-type → vertex label mapping
# ---------------------------------------------------------------------------

ENTITY_TYPE_MAP: dict[str, str] = {
    "Parties":                              "Party",
    "Governing Law":                        "Jurisdiction",
    "Agreement Date":                       "Date",
    "Effective Date":                       "Date",
    "Expiration Date":                      "Date",
    "Renewal Term":                         "Date",
    "License Grant":                        "LicenseClause",
    "Non-Transferable License":             "LicenseClause",
    "Irrevocable or Perpetual License":     "LicenseClause",
    "Unlimited/All-You-Can-Eat License":    "LicenseClause",
    "Affiliate License-Licensor":           "LicenseClause",
    "Affiliate License-Licensee":           "LicenseClause",
    "Termination for Convenience":          "TerminationClause",
    "Termination for Cause":               "TerminationClause",
    "Non-Compete":                          "RestrictionClause",
    "No-Solicit of Customers":             "RestrictionClause",
    "No-Solicit of Employees":             "RestrictionClause",
    "Exclusivity":                          "RestrictionClause",
    "Competitive Restriction Exception":    "RestrictionClause",
    "IP Ownership Assignment":             "IPClause",
    "Joint IP Ownership":                  "IPClause",
    "Source Code Escrow":                  "IPClause",
    "Cap on Liability":                    "LiabilityClause",
    "Uncapped Liability":                  "LiabilityClause",
    "Liquidated Damages":                  "LiabilityClause",
    "Change of Control":                   "Clause",
    "Anti-Assignment":                     "Clause",
    "Revenue/Profit Sharing":              "Clause",
    "Price Restrictions":                  "Clause",
    "Minimum Commitment":                  "Clause",
    "Volume Restriction":                  "Clause",
    "Post-Termination Services":           "Clause",
    "Audit Rights":                        "Clause",
    "Warranty Duration":                   "Clause",
    "Insurance":                           "Clause",
    "Covenant Not to Sue":                 "Clause",
    "Third Party Beneficiary":             "Clause",
    "Most Favored Nation":                 "Clause",
    "Non-Disparagement":                   "Clause",
    "Notice Period to Terminate Renewal":  "Clause",
}

# vertex label → relationship type (for CUAD annotation path)
RELATIONSHIP_MAP: dict[str, str] = {
    "Party":             "PARTY_TO",
    "Jurisdiction":      "GOVERNED_BY_LAW",
    "Date":              "HAS_DATE",
    "LicenseClause":     "HAS_LICENSE",
    "TerminationClause": "HAS_TERMINATION",
    "RestrictionClause": "HAS_RESTRICTION",
    "IPClause":          "HAS_IP_CLAUSE",
    "LiabilityClause":   "HAS_LIABILITY",
    "Clause":            "HAS_CLAUSE",
    "Contract":          "HAS_CLAUSE",  # fallback for Contract self-references
}


def entity_type_for(question_type: str) -> str:
    """Map a CUAD question type string to a vertex label. Defaults to 'Clause'."""
    return ENTITY_TYPE_MAP.get(question_type, "Clause")


def relationship_type_for(entity_type: str) -> str:
    """Map a vertex label to its canonical relationship type. Defaults to 'HAS_CLAUSE'."""
    return RELATIONSHIP_MAP.get(entity_type, "HAS_CLAUSE")

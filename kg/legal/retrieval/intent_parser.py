"""
Intent parser — maps natural-language queries to (intent, params) pairs.

No LLM.  Pure regex with named capture groups.
Tries patterns in registration order; first match wins.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class IntentMatch:
    intent: str
    params: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Name extraction helpers
# ---------------------------------------------------------------------------

_QUOTED = re.compile(r'"([^"]+)"|\'([^\']+)\'')

# Two-or-more consecutive title-case words (e.g. "Strategic Alliance Agreement").
_TITLED = re.compile(
    r"\b([A-Z][A-Za-z0-9&',.\\-]*(?:\s+[A-Z][A-Za-z0-9&',.\\-]*)+)\b"
)

# Single proper noun (must start uppercase) before contract-related words or
# end of sentence — catches "Lightbridge" in "fees in the Lightbridge agreement".
# No re.I — must be a genuinely uppercase-initial word.
_SINGLE_PROPER = re.compile(
    r"\b([A-Z][A-Za-z0-9&',.\\-]{2,})"
    r"(?=\s*(?:[?.!]|\s*$"
    r"|\s+(?:contract|agreement|corp|corporation|inc|ltd|llc|company)))",
)

# Interrogative words that should never be treated as entity names.
_INTERROGATIVES = frozenset({"Which", "What", "Who", "Whose", "Whom", "How", "When", "Where"})


def _extract_name(query: str) -> str | None:
    """Extract an entity name — quoted > multi-word title-case > single proper noun."""
    m = _QUOTED.search(query)
    if m:
        return (m.group(1) or m.group(2)).strip()
    m = _TITLED.search(query)
    if m:
        words = m.group(1).split()
        while words and words[0] in _INTERROGATIVES:
            words.pop(0)
        name = " ".join(words).strip()
        return name or None
    m = _SINGLE_PROPER.search(query)
    if m:
        name = m.group(1).strip()
        return None if name in _INTERROGATIVES else name
    return None


# ---------------------------------------------------------------------------
# Intent patterns — (compiled_pattern, intent_name)
# More specific patterns first; catch-all last.
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[re.Pattern[str], str]] = [

    # LINEAGE
    (re.compile(r"\bsupersed(?:e|es|ed|ing)?\b|\bsupercede[sd]?\b", re.I),   "find_superseded_contracts"),
    (re.compile(r"\bamend(?:s|ment|ed|ing)?\b", re.I),                         "find_amendments"),
    (re.compile(r"\bincorporat\w+\s+by\s+reference\b", re.I),                  "find_incorporated_documents"),
    (re.compile(r"\battache[sd]?\b|\battachment\b", re.I),                     "find_attachments"),
    (re.compile(r"\breplace[sd]?\b|\breplacements?\b", re.I),                  "find_replacements"),
    # "documents referenced", "are referenced", "referenced document"
    (re.compile(
        r"\breference(?:s|d)?\s+document\b"
        r"|\bexternal\s+document\b"
        r"|\bdocuments?\s+reference[sd]?\b"
        r"|\bare\s+referenced\b",
        re.I,
    ), "find_references"),

    # RISK — negative-clause checks before generic risk
    # Allow article ("a", "an") between keyword and noun
    (re.compile(r"\b(?:missing|no|lacks?|without)\b\s+(?:an?\s+)?\bindemnit\w+", re.I), "find_missing_indemnity"),
    (re.compile(r"\b(?:missing|no|lacks?|without)\b\s+(?:an?\s+)?\bterminat\w+", re.I), "find_missing_termination"),
    (re.compile(r"\bcause[sd]?\b.*\brisk\w*|\brisk\w*.*\bcause[sd]?\b", re.I), "find_risk_chains"),
    (re.compile(r"\brisk(?:s)?\b|\bcompliance\s+gap\b|\bexposure\b|\bvulnerabilit\w+\b", re.I), "find_all_risks"),

    # ENTITY
    (re.compile(r"\bindemnif\w+\b|\bindemnit(?:y|ies)\b", re.I),               "find_indemnification"),
    (re.compile(r"\bgoverning\s+law\b|\bjurisdiction\b|\bchoice\s+of\s+law\b", re.I), "find_jurisdictions"),
    (re.compile(r"\bterminat(?:e|ion|ing|ed)\b", re.I),                        "find_termination_clauses"),
    (re.compile(r"\bconfidential(?:ity)?\b|\bnda\b|\bnon.?disclosure\b", re.I), "find_confidentiality_clauses"),
    (re.compile(r"\bpayment\s+terms?\b|\bfee(?:s)?\b|\broyalt(?:y|ies)\b|\brevenue\s+shar\w+\b", re.I), "find_payment_terms"),
    (re.compile(r"\bobligations?\b|\bduties\b|\bduty\b", re.I),                "find_obligations"),
    (re.compile(r"\blimit(?:ation)?\s+(?:of\s+)?liabilit\w+\b|\bliabilit(?:y|ies)\b", re.I), "find_liability_clauses"),
    (re.compile(r"\beffective\s+date\b|\bcommences?\b|\btakes?\s+effect\b", re.I), "find_effective_dates"),
    (re.compile(r"\bexpir(?:ation|y)\s+date\b|\bexpire[sd]?\b|\bexpiry\b", re.I), "find_expiration_dates"),
    (re.compile(r"\brenewal\b|\bauto.?renew\w*\b", re.I),                      "find_renewal_terms"),
    (re.compile(r"\bdisclose[sd]?\b|\bdisclosures?\b", re.I),                  "find_disclosures"),
    # "signed by", "who signed", "parties", "signatories"
    (re.compile(r"\bsigned\b|\bsign(?:ed|s)?\s+by\b|\bpart(?:y|ies)\b|\bsignator(?:y|ies)\b", re.I), "find_parties"),

    # HIERARCHY — includes "structure" and "document structure"
    (re.compile(
        r"\bsections?\b|\bparagraphs?\b|\bheadings?\b|\bsubsections?\b"
        r"|\bdocument\s+structure\b|\bstructur(?:e|al)\b",
        re.I,
    ), "find_sections"),

    # CATCH-ALL
    (re.compile(r".*", re.I), "list_contracts"),
]


class IntentParser:
    """Parse a natural-language query into a structured IntentMatch."""

    def parse(self, query: str) -> IntentMatch:
        for pattern, intent in _PATTERNS:
            if pattern.search(query):
                name = _extract_name(query)
                params: dict[str, str] = {"name": name} if name else {}
                return IntentMatch(intent=intent, params=params)
        return IntentMatch(intent="list_contracts")

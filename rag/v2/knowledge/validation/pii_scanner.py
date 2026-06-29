"""PII scanner — post-generation output auditing via Presidio.

Wraps presidio-analyzer with:
  - lazy singleton initialisation (spaCy model loaded once per process)
  - explicit en_core_web_sm config (already downloaded for normalizer)
  - restricted entity list — only high-specificity PII types that
    warrant rejection; generic NER types (PERSON, ORG, DATE) are
    excluded because en_core_web_sm produces too many false positives
  - async interface via asyncio.to_thread (Presidio is sync/CPU-bound)

Usage:
  types = await scan_pii("Call me at 555-123-4567")
  # → ["PHONE_NUMBER"]

  clean = await has_pii("No PII here.")
  # → False
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Entity types to flag. Deliberately excludes PERSON, ORGANIZATION, LOCATION,
# DATE_TIME — these produce too many false positives with en_core_web_sm and
# are not reliably "sensitive" in a business knowledge-base context.
_SENSITIVE_ENTITIES: list[str] = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IBAN_CODE",
    "US_BANK_NUMBER",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "MEDICAL_LICENSE",
    "IP_ADDRESS",
    "CRYPTO",
]

_analyzer: Any | None = None


def _get_analyzer() -> Any:
    global _analyzer
    if _analyzer is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        # Use the already-downloaded en_core_web_sm (base dep via spaCy normalizer).
        # Presidio defaults to en_core_web_lg; this prevents a surprise model download.
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        })
        _analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
        logger.info("Presidio AnalyzerEngine initialised (en_core_web_sm)")
    return _analyzer


async def scan_pii(text: str) -> list[str]:
    """Return detected sensitive PII entity types. Empty list means clean."""
    if not text:
        return []
    analyzer = _get_analyzer()
    try:
        results = await asyncio.to_thread(
            analyzer.analyze,
            text=text,
            language="en",
            entities=_SENSITIVE_ENTITIES,
            score_threshold=0.7,    # ignore low-confidence matches (e.g. "Q4" → US_DRIVER_LICENSE at 0.3)
        )
        return [r.entity_type for r in results]
    except Exception as exc:
        logger.warning("PII scan failed (%s) — treating as clean", exc)
        return []


async def has_pii(text: str) -> bool:
    return bool(await scan_pii(text))

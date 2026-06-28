"""Query normalizer for cache keying.

Uses spaCy en_core_web_sm for tokenization and lemmatization.
Lazy-loaded once per process.

normalize_query("What industries does NeuralFlow AI serve?")
→ "what industry do neuralflow ai serve"

normalize_query("  What's the PTO policy??  ")
→ "what be the pto policy"
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_nlp: Any = None


def _get_nlp() -> Any:
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _nlp


def normalize_query(query: str) -> str:
    """Canonical query form for cache keying.

    - Tokenizes with spaCy (handles punctuation, contractions, unicode)
    - Lemmatizes every token ("industries" → "industry", "does" → "do")
    - Lowercases
    - Drops pure punctuation and whitespace tokens
    - Joins with single spaces
    """
    try:
        nlp = _get_nlp()
        doc = nlp(query)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_punct and not token.is_space
        ]
        return " ".join(tokens)
    except Exception as exc:
        # If spaCy fails for any reason, fall back to basic normalization
        logger.warning("spaCy normalization failed (%s) — falling back to basic", exc)
        return " ".join(query.lower().split())

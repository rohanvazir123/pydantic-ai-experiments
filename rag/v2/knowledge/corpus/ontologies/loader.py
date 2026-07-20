# Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
# See LICENSE file in the project root for details.

"""Ontology template loader with LRU cache.

Loads a Python file from the ontologies directory and returns the root
Pydantic BaseModel subclass (the last BaseModel subclass defined in the file,
by convention).

Cache: LRU(maxsize=32) — one entry per ontology path per worker process.
Call `load_ontology.cache_clear()` in tests to avoid cross-test pollution.
"""

import importlib.util
import logging
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_ONTOLOGIES_DIR = Path(__file__).parent


@lru_cache(maxsize=32)
def load_ontology(ontology_path: str | None) -> type[BaseModel]:
    """Return the root Pydantic class for the given ontology path.

    Args:
        ontology_path: Path relative to knowledge/corpus/ontologies/.
                       None → use the generic default ontology.

    Returns:
        The root BaseModel subclass (last one defined in the file).

    Raises:
        FileNotFoundError: if the ontology file does not exist.
        ValueError: if the file contains no BaseModel subclass.
    """
    if ontology_path is None:
        from knowledge.corpus.ontologies.generic import GenericDocument
        return GenericDocument

    full_path = _ONTOLOGIES_DIR / ontology_path
    if not full_path.exists():
        raise FileNotFoundError(f"Ontology not found: {full_path}")

    spec = importlib.util.spec_from_file_location("_ontology_module", full_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load spec from {full_path}")

    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ValueError(f"No loader for spec at {full_path}")
    spec.loader.exec_module(module)

    # Root class = last BaseModel subclass defined (by file convention)
    root_class: type[BaseModel] | None = None
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseModel)
            and obj is not BaseModel
            and obj.__module__ == module.__name__
        ):
            root_class = obj

    if root_class is None:
        raise ValueError(f"No BaseModel subclass found in {full_path}")

    logger.info("Loaded ontology '%s' → root class: %s", ontology_path, root_class.__name__)
    return root_class

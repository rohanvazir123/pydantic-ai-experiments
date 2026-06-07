# knowledge/corpus/ontologies/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)
- [Writing an Ontology](#writing-an-ontology)
- [Uploading via API](#uploading-via-api)

---

## What This Is

Pydantic ontology templates for docling-graph knowledge graph extraction. Each `.py` file defines the entity types and relationships that docling-graph will extract from documents ingested into a corpus.

---

## Files

| File | Purpose |
|------|---------|
| `loader.py` | `load_ontology(path)`: loads a Python file, returns the root `BaseModel` class; LRU-cached per worker |
| `generic.py` | Default ontology: `GenericDocument` + `GenericEntity` — extracts named entities without domain specifics |
| `<corpus_id>.py` | User-uploaded domain ontologies (e.g. `hr_policy.py`, `legal_contract.py`) |

---

## Writing an Ontology

An ontology is a plain Python file containing Pydantic `BaseModel` subclasses. The **last** class defined is the root (what `PipelineConfig.template` receives).

Minimal structure every ontology must follow:

```python
from typing import Any, List
from pydantic import BaseModel, ConfigDict, Field

def edge(label: str, **kwargs: Any) -> Any:
    """Required helper — marks a field as a directed graph edge."""
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

class MyEntity(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])  # stable node ID
    name: str = Field(description="Entity name. LOOK FOR: ... EXAMPLES: ...")

class MyDocument(BaseModel):                             # root class — last in file
    model_config = ConfigDict(graph_id_fields=["title"])
    title: str = Field(description="Document title")
    entities: List[MyEntity] = edge(label="MENTIONS", default_factory=list,
                                     description="Entities mentioned")

MyDocument.model_rebuild()  # required when using forward references
```

Key rules:
- `edge()` helper **must** be defined identically in every file
- Entities have `graph_id_fields` for stable node deduplication across chunks
- Components (value objects) have `is_entity=False` in `ConfigDict`
- Field `description` follows `LOOK FOR / EXTRACT / EXAMPLES` — this is the LLM prompt

For a full worked example see `basics/rag/memory/MEMORY_DESIGN.md` and the docling-graph docs at `basics/rag/docling-graph/faq.md`.

---

## Uploading via API

```bash
# Upload a new ontology for a corpus
curl -X POST http://localhost:8000/api/v1/corpus/my-corpus/ontology \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@my_ontology.py"

# Revert to generic default
curl -X DELETE http://localhost:8000/api/v1/corpus/my-corpus/ontology \
  -H "Authorization: Bearer $TOKEN"
```

The API validates the file (checks for root `BaseModel` subclass and `edge()` helper) before saving.

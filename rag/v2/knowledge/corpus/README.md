# knowledge/corpus/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)
- [Ontologies](#ontologies)

---

## What This Is

Corpus registry and ontology management. A corpus is a named, isolated namespace for documents — each has its own `corpus_id`, RBAC roles, and optional graph extraction ontology.

---

## Files

| File | Purpose |
|------|---------|
| `registry.py` | `CorpusRegistry`: loads corpus configs from settings, enforces RBAC at query time |

---

## Ontologies

The `ontologies/` subdirectory holds Pydantic template files for docling-graph knowledge graph extraction. See `ontologies/README.md` for details.

Corpus configs are defined in `CORPUS_CONFIGS_JSON` (`.env`). Each corpus can reference an ontology file:

```json
{
  "id": "legal",
  "enable_graph_extraction": true,
  "graph_ontology_path": "legal_contract.py"
}
```

If `graph_ontology_path` is `null`, the generic default ontology (`ontologies/generic.py`) is used, which extracts named entities and relationships without domain specifics.

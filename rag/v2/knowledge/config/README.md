# knowledge/config/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)
- [Usage](#usage)
- [Adding a New Setting](#adding-a-new-setting)

---

## What This Is

All runtime configuration lives here. Every module reads settings through `load_settings()` rather than calling `os.environ` directly. This makes settings testable (no env pollution) and centrally documented.

---

## Files

| File | Purpose |
|------|---------|
| `settings.py` | `Settings` (Pydantic BaseSettings) + `CorpusConfig` + `load_settings()` LRU singleton |

---

## Usage

```python
from knowledge.config.settings import load_settings

settings = load_settings()
print(settings.database_url)
print(settings.age_graph_name("acme", "hr-policies"))  # → "kg_acme_hr_policies"
```

In tests, always instantiate `Settings` directly (bypassing the LRU cache) with a clean env:

```python
from unittest import mock
from knowledge.config.settings import Settings

with mock.patch.dict(os.environ, {"DATABASE_URL": "...", "AGE_DATABASE_URL": "..."}, clear=True):
    s = Settings(_env_file=None)
```

---

## Adding a New Setting

1. Add a field to `Settings` in `settings.py` with a default and `Field(description=...)`.
2. Add the corresponding env var (uppercase) to `.env.example` with a comment.
3. Add a test in `tests/unit/test_settings.py` verifying the default and at least one override.

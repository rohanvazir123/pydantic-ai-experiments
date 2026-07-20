# Plugin Architecture — Design Doc

A modular system for **hardware simulation models** — battery, motor, rotor,
sensor models, one eVTOL variant vs. another — so a developer adds a new one
by writing a new file, not by editing the core simulation engine. This
document compares three ways to wire that up; nothing is implemented yet.

## Table of Contents

- [Where this fits](#where-this-fits)
- [Requirements](#requirements)
- [The plugin contract](#the-plugin-contract)
- [Approach A — Protocol + explicit registry](#approach-a--protocol--explicit-registry)
- [Approach B — Directory auto-discovery](#approach-b--directory-auto-discovery)
- [Approach C — Installable packages via entry points](#approach-c--installable-packages-via-entry-points)
- [Comparison](#comparison)
- [Recommendation](#recommendation)
- [Open questions / named limits](#open-questions--named-limits)
- [Proposed file layout](#proposed-file-layout)
- [Status](#status)

## Where this fits

This is the same open/closed problem `worker_queues/base.py`'s `Job` already
solves, one level up: `Job.process()` lets "new job types = new subclasses,
no worker change" because the worker calls one method it never has to
branch on (Command pattern). A hardware model is the same shape — the engine
should call one method on whatever model is configured, and never contain a
growing `if model_type == "battery_li_ion": ... elif ...` ladder.

`anomaly/events.py`'s `EvtolSensorSource` is the "before" picture: it
*is* a hardcoded hardware model (fixed battery-temp/rotor-vibration ranges,
baked into the simulation loop). Adding a second aircraft type today means
editing that class. This doc is about not doing that again.

## Requirements

- A developer adds a new hardware model by adding code, not editing the
  engine's loop, dispatch logic, or any existing model.
- The engine can run several models at once (a fleet mixing aircraft types,
  or one aircraft's battery + motor + rotor models together) without knowing
  their concrete types.
- A model owns its own internal state; the engine only owns the tick loop
  that advances every model forward — same "worker owns its resequencer, isn't
  one" composition rule as `worker_queues`.
- One model misbehaving (raises, returns garbage) shouldn't be free to take
  down every other model's simulation.

## The plugin contract

Sketch, not final — the shape every approach below shares:

```python
class HardwareModel(Protocol):
    model_id: str  # e.g. "battery_li_ion_v1"

    def initial_state(self) -> dict: ...
    def step(self, state: dict, dt_seconds: float) -> dict: ...
```

`step` takes the model's own last state and returns its next state (and,
implicitly, the reading to emit) — pull-based, the engine's tick loop calls
it, the same "engine pulls every model forward" shape as the moving
average's "compute on read" rather than a model pushing updates on its own
schedule (parent README's Question 3). The three approaches below differ
only in **how a model gets registered**, not in this contract.

## Approach A — Protocol + explicit registry

A module-level dict (`_REGISTRY: dict[str, type[HardwareModel]]`) and a
decorator:

```python
@register_model("battery_li_ion_v1")
class LiIonBatteryModel:
    ...
```

The engine resolves a model by name at startup (`_REGISTRY["battery_li_ion_v1"]()`)
and calls only `initial_state()`/`step()` on it — it never imports a concrete
model class. New models: add a file, import it once wherever registration
needs to run (e.g., a `plugins/__init__.py` that imports every model file),
decorate the class, done.

**Trade-offs**
- \+ Simplest possible version of the contract; a few lines of registry code,
  no new dependency, no packaging.
- \+ Failures are local and visible: a broken plugin fails at its own
  `import`, not silently at runtime.
- − Still requires one line somewhere (the `plugins/__init__.py` import
  list) to change per new model — "no editing the engine" holds, but "zero
  files touched anywhere" doesn't.

## Approach B — Directory auto-discovery

Same registry and decorator as A, but the engine itself scans a
`models/` directory at startup (`pkgutil.iter_modules` + `importlib.import_module`
over every `.py` file found) instead of a hand-maintained import list. Drop a
file in the folder; it self-registers on import; nothing else changes.

**Trade-offs**
- \+ Closest to "just add a file" — no import list to maintain at all.
- − Import-time magic: a typo'd or broken plugin file can break the scan for
  everyone unless the scanner wraps each import in its own try/except and
  logs-and-skips the failure (worth doing, not free).
- − Harder to answer "which models are actually active" by reading code —
  it's now "whatever happens to be in the folder," not an explicit list.

## Approach C — Installable packages via entry points

`importlib.metadata.entry_points()`, the mechanism pytest/Flask use: each
plugin is its own installable package declaring an entry point
(`[project.entry-points."telemetry.hardware_models"]` in its `pyproject.toml`).
The engine discovers every installed plugin without importing or even
knowing about its package name.

**Trade-offs**
- \+ The only approach where a plugin can be developed and distributed
  **outside this repo entirely** — a third party ships a model as its own
  pip package.
- \+ Real prior art for exactly this problem (pytest's plugin system).
- − Every plugin needs its own installable package and `pyproject.toml` —
  real packaging overhead for what is, in this repo, a teaching exercise
  with no external distribution need.
- − Slowest edit-test loop: a plugin change means reinstalling that package,
  not just re-running the engine.

## Comparison

| | A: Explicit registry | B: Directory auto-discovery | C: Entry points |
|---|---|---|---|
| To add a model | New file + one import line + decorator | New file + decorator, nothing else | New installable package + entry point declaration |
| "No engine edits" holds? | Yes | Yes | Yes |
| "Zero files touched anywhere" | No (import list) | Yes | Yes (after initial packaging setup) |
| Failure isolation | Import fails loudly, locally | Needs explicit per-file try/except in the scanner | Per-package install failures, isolated by nature |
| External/third-party plugins | No | No | Yes |
| Setup cost | Lowest | Low (a scanner function) | Highest (packaging per plugin) |
| Prior art here | `worker_queues/base.py`'s `Job` | — | pytest, Flask (external) |

## Recommendation

**Build Approach A first.** It's the same registry-plus-protocol shape as B
and C underneath — nothing in the plugin contract or the registry changes if
this later grows into B (add a directory scanner that calls the same
decorator) or C (add entry-point discovery that populates the same
registry). The one-line import-list cost is real but small, and it keeps
"which models exist" explicit and greppable while this stays a
single-repo exercise.

**Move to B** only once enough models exist that hand-maintaining the import
list is itself the annoying part.

**Reach for C** only if a plugin ever needs to live and ship outside this
repo — not a need this exercise has today.

## Open questions / named limits

- **Failure isolation during a tick is undecided.** If one model's `step()`
  raises, does the engine skip that model for the tick, halt the whole run,
  or something else? Not picked yet.
- **Config validation timing is undecided.** If a configured model name
  isn't in the registry, should that fail at startup (fail fast) or only
  when the engine tries to resolve it mid-run? Fail-fast is the obvious
  answer but isn't designed here.
- **Interface versioning isn't addressed.** If `HardwareModel`'s contract
  changes later (a new required method), every existing plugin breaks with
  no compatibility story named.
- **State typing is loose in the sketch above** (`dict`, not a pydantic
  model) — this repo's own type-safety rule argues for a typed state object
  per model; left as `dict` here only because the concrete state shape is
  model-specific and this doc isn't picking models yet.

## Proposed file layout

Not created yet:

- `plugin_architecture/README.md` — this document
- `plugin_architecture/registry.py` — `HardwareModel` protocol, `register_model` decorator, `_REGISTRY`
- `plugin_architecture/engine.py` — the tick loop; resolves models by name via the registry, never imports a concrete model
- `plugin_architecture/models/` — one file per hardware model (e.g. `battery_li_ion.py`, `rotor_basic.py`)

## Status

**Designed, not built.**

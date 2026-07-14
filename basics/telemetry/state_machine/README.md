# Flight State Machine (JD 3b)

A finite state machine for a vehicle's flight lifecycle
(`PREFLIGHT → TAXI → TAKEOFF → CRUISE → LANDING → …`) where **invalid transitions
throw precise exceptions**. Hand-rolled on purpose — the exercise is about the
design, not about wiring up a library.

## Files

| File | Purpose |
|------|---------|
| `flight_state_machine.py` | Hand-rolled: transition table, guards, precise exceptions, demo. |
| `flight_state_machine_lib.py` | Alternate using the `python-statemachine` library (`StateChart`). |
| `../tests/test_flight_state_machine.py` | Pytest suite for the hand-rolled version (no services). |

## Two implementations

The hand-rolled version is the interview answer (the exercise says *implement*).
`flight_state_machine_lib.py` shows the same graph via `python-statemachine` for
contrast. Two things it taught us, both real interview talking points:

- **`StateChart` follows SCXML semantics** — an event with no enabled transition
  is *silently discarded* by default, which violates "throw precise exceptions".
  Set `allow_event_without_transition = False` to make it raise.
- The library raises **one generic `TransitionNotAllowed`** for *both* an illegal
  edge and a guard veto (`Can't Take off when in Taxi.`). The hand-rolled version
  distinguishes them (`InvalidTransition` vs `GuardRejected`). That's the trade:
  much less code, but coarser errors — and it also validates the graph at
  class-definition time (trap states raise `InvalidDefinition`).

Run it with the dependency (not part of the repo's deps):

```bash
uv run --with python-statemachine basics/telemetry/state_machine/flight_state_machine_lib.py
```

## How It Works

**Event-driven.** The whole machine is one nested table you read like a map:

```
_TRANSITIONS[current_state][event] -> Transition(target_state, guard)
```

You `fire(event)`; the lookup *is* the control flow — index by current state, then
by event. There are no branches on state anywhere else, and the guard (condition)
sits right next to the edge it governs.

Two failure modes → two precise exceptions (both subclass `StateMachineError`):

- `InvalidTransition(current, event)` — no table entry: the event doesn't apply
  here. Carries the state, the event, and which events *were* valid.
- `GuardRejected(current, event, reason)` — the entry exists but its guard vetoed
  it now (e.g. `take_off` before clearance).

`fire()` checks structure (is there an entry?) **before** policy (does the guard
allow it?), and mutates state **only after every check passes** — a rejected event
leaves the machine exactly where it was.

```python
from flight_state_machine import FlightEvent, FlightStateMachine, InvalidTransition

fsm = FlightStateMachine()             # starts PREFLIGHT
fsm.fire(FlightEvent.TAXI_OUT)
fsm.fire(FlightEvent.CLIMB)            # InvalidTransition: no such event from TAXI

fsm.cleared_for_takeoff = True         # guard input the take_off edge reads
fsm.fire(FlightEvent.TAKE_OFF)
```

Because events are just enum values, string dispatch drops straight in:
`fsm.fire(FlightEvent(payload["event"]))`.

**Import-time validation.** `_validate()` runs once when the module is imported
and rejects a malformed table immediately (raising `InvalidStateMachine`, not
`assert` — so it survives `python -O`). It checks three invariants: every state
has a table entry, every transition target exists, and every state is reachable
from the initial state (no dead states). This is the cheap replica of the
library's definition-time validation.

## Editing the table

The table is per-instance. To change the graph, either pass a custom `transitions`
dict to the constructor ("delete + create a whole new table") or edit one edge at
a time:

```python
fsm.add_transition(FlightState.PREFLIGHT, FlightEvent.SHUT_DOWN, FlightState.SHUTDOWN)
fsm.remove_transition(FlightState.LANDING, FlightEvent.GO_AROUND)
```

Both apply the change to a copy, run `_validate()`, and only swap it in if valid —
so a broken edit leaves the live table untouched (never a half-applied graph), and
a removal that would strand a state raises `InvalidStateMachine`.

## Versioning a running workflow

`add_transition` / `remove_transition` mutate **one in-memory instance**. Changing
the rules of many workflows that are **already in flight** is a *versioning*
problem, not an editing one — and for something safety-critical like flight state
you almost never want an aircraft's allowed transitions to change under it
mid-flight. Standard ways to handle it:

- **Pin the version per instance (recommended default).** Each run records the
  definition version it started on and runs to completion on that version; new
  runs get the new version. No in-flight instance ever changes rules under its
  feet. This is how Temporal (workflow versioning) and AWS Step Functions
  (versions + aliases) default. Cost: you retain old definitions until their runs
  drain.
- **Additive / expand–contract.** Only *add* states/edges to a live version;
  never remove or repurpose. To retire an edge: stop new runs from using it, wait
  for old runs to drain, then delete. Removals are always safe because nothing
  in flight depended on the new shape.
- **Drain + cutover (blue/green).** Stop starting runs on v1, let v1 drain, flip
  to v2. Simple, but needs a drain window — impractical for very long-lived runs.
- **Patch/branch by version.** Guard new behavior behind a version check so old
  in-flight runs deterministically take the old path and new runs take the new one
  (Temporal's `patched()` / `GetVersion`).
- **Live migration (last resort).** Pause the instance, map its current state to a
  legal state in the new table, validate, resume. Powerful but error-prone —
  reserve for critical hotfixes, and audit it.

The invariant behind all of them: **an in-flight instance's current state (and the
path it still needs) must remain valid under whatever table you move it to.**
Remove the edge it's about to take, or the state it's sitting in, and you've
bricked the run. That's why version pinning is the safe default and live mutation
is the exception.

## Hand-rolled vs `python-statemachine`

For a small, flat FSM where clear errors matter, hand-rolled wins. Cross into
hierarchy, persistence, or a large graph and the library pulls ahead. Knowing
*when* each is right is the interview answer — not picking a permanent winner.

**Where the hand-rolled version wins:**

- **Readability** — the whole machine is one table read top-to-bottom; nothing
  hidden in a metaclass.
- **Precise errors** — two distinct exceptions (`InvalidTransition` vs
  `GuardRejected`) vs the library's single generic `TransitionNotAllowed`.
- **No surprises** — no SCXML silent-discard footgun, no
  `allow_event_without_transition` flag to remember.
- **Zero dependency** — the logic is a table plus a ~20-line `fire()` you fully
  own (the file is ~300 lines, mostly comments), and `_validate()` recovers the
  library's structural checks.

**Where the library wins:**

- **Hierarchical / parallel / nested states** — once you need substates
  (`Cruise → {Climbing, Level, Descending}`), a flat dict stops scaling and
  `StateChart` earns its keep.
- **Batteries** — enter/exit callbacks, async, persistence, diagram export, i18n
  messages.
- **Maturity** — a well-tested engine vs code you maintain.

## Running the Demo

```bash
.venv/bin/python basics/telemetry/state_machine/flight_state_machine.py
```

Walks a full flight (with a guarded takeoff), shows a guard veto and an illegal
transition being rejected, then prints the transition history.

## Running the Tests

```bash
.venv/bin/python -m pytest basics/telemetry/tests/test_flight_state_machine.py -v
```

Dependencies: `pytest` only.

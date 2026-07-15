"""Flight-state machine -- event-driven transition table + precise exceptions.

Models a vehicle's flight lifecycle as a finite state machine. The prompt asks
for a *robust* machine where *invalid transitions throw precise exceptions*.

The whole machine is ONE nested table you read like a map:

    _TRANSITIONS[current_state][event] -> Transition(target_state, guard)

To fire an event: index by the current state, then by the event. That lookup is
the entire control flow -- there are no branches on state anywhere else.

  * A MISSING (state, event) entry means the event doesn't apply here → raise
    ``InvalidTransition`` (structure: "that's not a real transition").
  * A PRESENT entry with a guard that vetoes means the event applies but not now
    → raise ``GuardRejected`` (policy: "not yet"). Two distinct failures, two
    distinct, precise exceptions.

Event-driven (fire an EVENT, the table decides the next state) rather than
target-driven (name the next state yourself): it matches how a real system is
driven -- a command arrives ("take_off") and the machine computes the result --
and it maps 1:1 onto a string dispatch like ``fire(FlightEvent(payload))``.

Why not a library (``transitions`` / ``python-statemachine``)? For a graph this
small a table is less code and less risk; reach for a library when you need
hierarchical/parallel states, persistence, or diagram export. The one library
feature worth stealing -- definition-time validation -- is replicated cheaply by
``_validate()``, which runs at import and rejects a malformed table immediately.

Transition graph -- ``STATE --event[guard]--> STATE`` (keep in sync with the table):

    PREFLIGHT --taxi_out--------------> TAXI
    TAXI      --take_off[cleared]-----> TAKEOFF
    TAXI      --abort----------------->  PREFLIGHT
    TAXI      --shut_down------------->  SHUTDOWN
    TAKEOFF   --climb----------------->  CRUISE
    TAKEOFF   --emergency_return------>  LANDING
    CRUISE    --descend--------------->  LANDING
    LANDING   --roll_out------------->   TAXI     (rollout)
    LANDING   --go_around------------>   TAKEOFF  (go-around)
    SHUTDOWN  (terminal -- no events)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import NamedTuple


class FlightState(StrEnum):
    """The vehicle's flight states. ``StrEnum`` → clean logging / JSON."""

    PREFLIGHT = "preflight"
    TAXI = "taxi"
    TAKEOFF = "takeoff"
    CRUISE = "cruise"
    LANDING = "landing"
    SHUTDOWN = "shutdown"


class FlightEvent(StrEnum):
    """The commands that drive the machine. ``fire(event)`` looks these up."""

    TAXI_OUT = "taxi_out"
    TAKE_OFF = "take_off"
    ABORT = "abort"
    SHUT_DOWN = "shut_down"
    CLIMB = "climb"
    EMERGENCY_RETURN = "emergency_return"
    DESCEND = "descend"
    ROLL_OUT = "roll_out"
    GO_AROUND = "go_around"


# A guard answers "is this transition safe RIGHT NOW?" given the machine -- return
# None to allow, or a string reason to veto. It lives in the table entry, so the
# condition sits right next to the edge it governs.
Guard = Callable[["FlightStateMachine"], "str | None"]


class Transition(NamedTuple):
    """One table entry: where the event leads, and an optional guard."""

    target: FlightState
    guard: Guard | None = None


# The state the machine boots into, and the root for the reachability check.
_INITIAL = FlightState.PREFLIGHT

# The entire machine: current state -> event -> Transition. This nested dict is
# the single source of truth for BOTH shape (which events exist per state) and
# policy (the guard on each). Terminal states map to an empty dict.
_TRANSITIONS: dict[FlightState, dict[FlightEvent, Transition]] = {
    FlightState.PREFLIGHT: {
        FlightEvent.TAXI_OUT: Transition(FlightState.TAXI),
    },
    FlightState.TAXI: {
        FlightEvent.TAKE_OFF: Transition(
            FlightState.TAKEOFF,
            guard=lambda m: None if m.cleared_for_takeoff else "not cleared for takeoff",
        ),
        FlightEvent.ABORT: Transition(FlightState.PREFLIGHT),
        FlightEvent.SHUT_DOWN: Transition(FlightState.SHUTDOWN),
    },
    FlightState.TAKEOFF: {
        FlightEvent.CLIMB: Transition(FlightState.CRUISE),
        FlightEvent.EMERGENCY_RETURN: Transition(FlightState.LANDING),
    },
    FlightState.CRUISE: {
        FlightEvent.DESCEND: Transition(FlightState.LANDING),
    },
    FlightState.LANDING: {
        FlightEvent.ROLL_OUT: Transition(FlightState.TAXI),
        FlightEvent.GO_AROUND: Transition(FlightState.TAKEOFF),
    },
    FlightState.SHUTDOWN: {},  # terminal -- no outgoing events
}


class StateMachineError(Exception):
    """Base for every transition failure -- catch this to handle any of them."""


class InvalidTransition(StateMachineError):
    """The event doesn't apply in the current state (no table entry).

    Carries the state, the event, and which events *were* valid, so logs and
    callers get a precise message instead of a generic error.
    """

    def __init__(
        self, current: FlightState, event: FlightEvent, allowed: Iterable[FlightEvent]
    ) -> None:
        self.current = current
        self.event = event
        allowed_str = ", ".join(sorted(e.value for e in allowed)) or "<none: terminal>"
        super().__init__(
            f"event {event.value!r} not valid in {current.value}; allowed: {allowed_str}"
        )


class GuardRejected(StateMachineError):
    """The event applies, but its guard vetoed it right now.

    Structure is fine; policy said no (e.g. take_off before clearance). Distinct
    from :class:`InvalidTransition` so callers can tell an impossible event from a
    legal-but-not-yet one.
    """

    def __init__(self, current: FlightState, event: FlightEvent, reason: str) -> None:
        self.current = current
        self.event = event
        self.reason = reason
        super().__init__(f"event {event.value!r} in {current.value} blocked: {reason}")


class InvalidStateMachine(StateMachineError):
    """The transition table itself is malformed. Raised at import by _validate()."""


def _validate(
    transitions: dict[FlightState, dict[FlightEvent, Transition]] = _TRANSITIONS,
    initial: FlightState = _INITIAL,
) -> None:
    """Structural self-check for a transition table. Runs at import on the default,
    and again in ``__init__`` on any custom table a caller supplies.

    This is the cheap version of the library's definition-time validation: a
    malformed table fails loudly and immediately instead of misbehaving at
    runtime. We deliberately raise (not ``assert``) so the check survives
    ``python -O``, which strips asserts. Three invariants:

      1. Completeness  -- every FlightState has a table entry (none left out).
      2. Valid targets -- every transition points at a state that has an entry.
      3. Reachability  -- every state is reachable from ``initial`` (no dead
         states stranded off the graph).
    """
    missing = set(FlightState) - set(transitions)
    if missing:
        raise InvalidStateMachine(
            f"states missing from the table: {sorted(s.value for s in missing)}"
        )

    for state, events in transitions.items():
        for event, transition in events.items():
            if transition.target not in transitions:
                raise InvalidStateMachine(
                    f"{state.value} --{event.value}--> unknown target {transition.target!r}"
                )

    # BFS from the initial state over the transition edges.
    seen = {initial}
    stack = [initial]
    while stack:
        for transition in transitions[stack.pop()].values():
            if transition.target not in seen:
                seen.add(transition.target)
                stack.append(transition.target)
    unreachable = set(FlightState) - seen
    if unreachable:
        raise InvalidStateMachine(
            f"states unreachable from {initial.value}: "
            f"{sorted(s.value for s in unreachable)}"
        )


_validate()  # fail fast at import if the default table is malformed


class FlightStateMachine:
    """A finite state machine driven by events over a transition table.

    Uses the module ``_TRANSITIONS`` by default, but a caller can pass their own
    ``transitions`` to add/remove/rewire edges as needed -- a custom table is
    validated on construction. Each instance carries its own current state,
    history, and guard inputs (here, ``cleared_for_takeoff``).
    """

    def __init__(
        self,
        initial: FlightState = _INITIAL,
        *,
        transitions: dict[FlightState, dict[FlightEvent, Transition]] | None = None,
        cleared_for_takeoff: bool = False,
    ) -> None:
        # Default to the shared module table; a custom one is validated first so a
        # malformed override fails at construction, not mid-flight.
        if transitions is None:
            self._transitions = _TRANSITIONS
        else:
            _validate(transitions, initial)
            self._transitions = transitions
        self._initial = initial  # root for reachability re-checks on edits
        self._state = initial
        self._history: list[FlightState] = [initial]
        # Guard input read by the take_off transition's guard. Flip it to model
        # ground control clearing the vehicle.
        self.cleared_for_takeoff = cleared_for_takeoff

    @property
    def state(self) -> FlightState:
        return self._state

    @property
    def history(self) -> list[FlightState]:
        return list(self._history)  # copy -- callers can't mutate our log

    @property
    def is_terminal(self) -> bool:
        """True if no event leaves the current state (e.g. SHUTDOWN)."""
        return not self._transitions[self._state]

    # -- Introspection helpers ---------------------------------------------
    # NOTE: not used by fire() or the demo -- fire() + catch is the normal path.
    # Kept deliberately for CALLERS doing look-before-you-leap validation, e.g.
    # a REST layer returning 409 for an inapplicable event, or a dashboard
    # graying out buttons that can't fire from the current state. Both read the
    # table without mutating, so they're safe to call anytime.

    def allowed_events(self) -> set[FlightEvent]:
        """The events that can fire from the current state (ignores guards)."""
        return set(self._transitions[self._state])

    def can_fire(self, event: FlightEvent) -> bool:
        """Is ``event`` a valid edge from the current state? (Ignores guards.)"""
        return event in self._transitions[self._state]

    def fire(self, event: FlightEvent) -> FlightState:
        """Apply ``event`` or raise a precise exception; state is unchanged on failure.

        The lookup IS the logic: index the table by current state, then by event.
        Structure is checked first (is there an entry?), then policy (does the
        guard allow it now?) -- so an impossible event always raises
        :class:`InvalidTransition`, never masked by a guard.
        """
        table = self._transitions[self._state]
        transition = table.get(event)
        if transition is None:
            raise InvalidTransition(self._state, event, allowed=table)

        if transition.guard is not None:
            reason = transition.guard(self)
            if reason is not None:
                raise GuardRejected(self._state, event, reason)

        # Mutate only after every check passes -- a rejected event leaves the
        # machine exactly where it was (no half-applied state).
        previous = self._state
        self._state = transition.target
        self._history.append(self._state)
        # Log the transition with all three pieces (from / event / to) so the
        # trail is greppable; failures are already surfaced via the exceptions.
        print(f"[fsm] {previous.value} --{event.value}--> {self._state.value}")
        return self._state

    # -- Editing the table at runtime --------------------------------------
    # Conveniences over "delete + create a whole new table". Each edit is applied
    # to a COPY, validated, and only then swapped in -- so a rejected edit leaves
    # the live table untouched (never a half-applied graph). NOTE: this mutates
    # ONE in-memory instance. Changing the rules of many *already-running*
    # workflows in real time is a versioning problem, not an editing one -- see
    # the "Versioning a running workflow" note in the README.

    def add_transition(
        self,
        state: FlightState,
        event: FlightEvent,
        target: FlightState,
        guard: Guard | None = None,
    ) -> None:
        """Add or replace one edge, re-validating the whole table first."""
        updated = {s: dict(events) for s, events in self._transitions.items()}
        updated.setdefault(state, {})[event] = Transition(target, guard)
        _validate(updated, self._initial)  # reject an edit that breaks the graph
        self._transitions = updated

    def remove_transition(self, state: FlightState, event: FlightEvent) -> None:
        """Remove one edge, re-validating the whole table first.

        Raises ``InvalidStateMachine`` if the removal would strand a state
        (unreachable from the initial state).
        """
        updated = {s: dict(events) for s, events in self._transitions.items()}
        updated.get(state, {}).pop(event, None)
        _validate(updated, self._initial)
        self._transitions = updated


if __name__ == "__main__":
    fsm = FlightStateMachine()

    fsm.fire(FlightEvent.TAXI_OUT)
    try:
        fsm.fire(FlightEvent.TAKE_OFF)  # valid event, but guard vetoes
    except GuardRejected as exc:
        print(f"guard   : {exc}")

    fsm.cleared_for_takeoff = True  # ground control clears us
    fsm.fire(FlightEvent.TAKE_OFF)
    fsm.fire(FlightEvent.CLIMB)
    fsm.fire(FlightEvent.DESCEND)  # cruise -> landing

    try:
        fsm.fire(FlightEvent.CLIMB)  # no such event from LANDING
    except InvalidTransition as exc:
        print(f"invalid : {exc}")

    fsm.fire(FlightEvent.ROLL_OUT)   # landing -> taxi
    fsm.fire(FlightEvent.SHUT_DOWN)  # terminal

    print("history :", " -> ".join(s.value for s in fsm.history))
    print("terminal:", fsm.is_terminal)

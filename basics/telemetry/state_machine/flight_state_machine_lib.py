"""Flight-state machine (JD 3b) -- library version using `python-statemachine`.

Same graph and intent as ``flight_state_machine.py``, but declared with the
``python-statemachine`` library instead of a hand-rolled transition table. This
is the "don't reinvent the wheel" answer -- useful to contrast in an interview.

What the library buys you:
  * States and transitions are declarative; ``preflight.to(taxi)`` reads like the
    diagram, and events are first-class (``machine.take_off()``).
  * Definition-time validation: a non-final state with no outgoing transition is
    a "trap state" and raises ``InvalidDefinition`` at class creation -- a whole
    class of bugs caught before you ever run.
  * Guards (``cond=``), enter/exit callbacks, and (with StateChart) nested/parallel
    states and SCXML semantics come for free.

Two gotchas worth knowing (both handled below):
  1. ``StateChart`` follows SCXML semantics, so by default an event with no
     enabled transition is SILENTLY DISCARDED -- which violates the prompt's
     "invalid transitions throw precise exceptions". We set
     ``allow_event_without_transition = False`` to make it raise instead.
  2. The library raises ONE generic ``TransitionNotAllowed`` for BOTH an illegal
     edge AND a guard veto -- e.g. "Can't Take off when in Taxi." It does not
     distinguish structure from policy the way the hand-rolled version does with
     ``InvalidTransition`` vs ``GuardRejected``. That's the real trade: far less
     code, but coarser errors. If you need to tell "not a real transition" from
     "not yet", the hand-rolled version wins.

Transition graph (identical to the hand-rolled version):

    PREFLIGHT → TAXI
    TAXI      → TAKEOFF | PREFLIGHT (abort) | SHUTDOWN
    TAKEOFF   → CRUISE  | LANDING  (abort)
    CRUISE    → LANDING
    LANDING   → TAXI (rollout) | TAKEOFF (go-around)
    SHUTDOWN  → ⊥ (terminal)
"""

from __future__ import annotations

from statemachine import State, StateChart
from statemachine.exceptions import TransitionNotAllowed

__all__ = ["FlightMachine", "TransitionNotAllowed"]


class FlightMachine(StateChart):
    """Flight lifecycle as a declarative state chart.

    ``allow_event_without_transition = False`` overrides SCXML's silent-discard so
    an illegal event (or a guard-vetoed one) raises ``TransitionNotAllowed`` --
    the behavior the exercise asks for.
    """

    allow_event_without_transition = False

    # --- States (one is initial, one is final) ---
    preflight = State(initial=True)
    taxi = State()
    takeoff = State()
    cruise = State()
    landing = State()
    shutdown = State(final=True)

    # --- Events: each reads like an edge in the diagram. ``|`` unions the
    #     transitions that share an event name; ``cond=`` attaches a guard. ---
    taxi_out = preflight.to(taxi)
    take_off = taxi.to(takeoff, cond="cleared_for_takeoff")  # guarded
    abort = taxi.to(preflight)
    shut_down = taxi.to(shutdown)
    climb = takeoff.to(cruise)
    emergency_return = takeoff.to(landing)
    descend = cruise.to(landing)
    roll_out = landing.to(taxi)
    go_around = landing.to(takeoff)

    def __init__(self) -> None:
        # Guard state for take_off. Flip via `clear_for_takeoff()`; the `cond` on
        # the take_off event reads this attribute by name.
        self.cleared_for_takeoff = False
        self.history: list[str] = []
        super().__init__()  # enters the initial state (fires on_enter_state)

    # Generic enter hook: the library calls it on every state entry and injects
    # `target` (the State) by name. We use it to record history, like the
    # hand-rolled machine's _history list.
    def on_enter_state(self, target: State) -> None:
        self.history.append(target.id)

    def clear_for_takeoff(self) -> None:
        self.cleared_for_takeoff = True

    @property
    def current(self) -> str:
        """Current state id (flat chart → exactly one active state)."""
        return next(iter(self.configuration)).id

    @property
    def is_terminal(self) -> bool:
        return next(iter(self.configuration)).final


if __name__ == "__main__":
    # Dispatch events by NAME with send(). This is the realistic entry point: a
    # flight command usually arrives as data -- a REST payload {"event": "take_off"}
    # or a message off a bus -- so string dispatch maps 1:1 onto it, and one
    # try/except around send() guards every externally-driven transition. We catch
    # fsm.TransitionNotAllowed (exposed on the instance) so no separate import is
    # needed at the call site.
    fsm = FlightMachine()

    fsm.send("taxi_out")
    try:
        fsm.send("take_off")  # legal edge, but guard (cleared_for_takeoff) vetoes
    except fsm.TransitionNotAllowed as exc:
        print(f"guard   : {exc}")

    fsm.clear_for_takeoff()
    for event in ("take_off", "climb", "descend"):  # -> takeoff -> cruise -> landing
        fsm.send(event)

    try:
        fsm.send("climb")  # illegal from landing (also: an unknown name raises here)
    except fsm.TransitionNotAllowed as exc:
        print(f"invalid : {exc}")

    fsm.send("roll_out")   # landing -> taxi
    fsm.send("shut_down")  # terminal

    print("history :", " -> ".join(fsm.history))
    print("terminal:", fsm.is_terminal)

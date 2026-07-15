"""Tests for the event-driven flight-state machine.

Run from the repo root with the project venv:

    .venv/bin/python -m pytest basics/telemetry/tests/test_flight_state_machine.py -v

``conftest.py`` puts the ``state_machine/`` source dir on ``sys.path`` so the
import below is a bare module import.
"""

import flight_state_machine as fsm_mod
import pytest
from flight_state_machine import (
    FlightEvent,
    FlightState,
    FlightStateMachine,
    GuardRejected,
    InvalidStateMachine,
    InvalidTransition,
    Transition,
)


def _copy_table() -> dict:
    """A deep-enough copy of the default table (inner dicts copied; Transitions
    are immutable so sharing them is fine)."""
    return {state: dict(events) for state, events in fsm_mod._TRANSITIONS.items()}


def test_happy_path() -> None:
    fsm = FlightStateMachine(cleared_for_takeoff=True)
    for event in (
        FlightEvent.TAXI_OUT,
        FlightEvent.TAKE_OFF,
        FlightEvent.CLIMB,
        FlightEvent.DESCEND,
    ):
        fsm.fire(event)
    assert fsm.state is FlightState.LANDING
    assert fsm.history[0] is FlightState.PREFLIGHT
    assert fsm.history[-1] is FlightState.LANDING


def test_invalid_event_is_precise() -> None:
    fsm = FlightStateMachine()  # PREFLIGHT
    with pytest.raises(InvalidTransition) as exc:
        fsm.fire(FlightEvent.CLIMB)  # no such event from PREFLIGHT
    # the exception carries the state and the offending event
    assert exc.value.current is FlightState.PREFLIGHT
    assert exc.value.event is FlightEvent.CLIMB
    # a rejected event must NOT change state
    assert fsm.state is FlightState.PREFLIGHT


def test_guard_vetoes_a_valid_event() -> None:
    fsm = FlightStateMachine(cleared_for_takeoff=False)
    fsm.fire(FlightEvent.TAXI_OUT)
    with pytest.raises(GuardRejected) as exc:
        fsm.fire(FlightEvent.TAKE_OFF)  # valid event, guard says no
    assert exc.value.reason == "not cleared for takeoff"
    assert exc.value.event is FlightEvent.TAKE_OFF
    assert fsm.state is FlightState.TAXI  # still taxiing, unchanged


def test_guard_clears_then_allows() -> None:
    fsm = FlightStateMachine()
    fsm.fire(FlightEvent.TAXI_OUT)
    fsm.cleared_for_takeoff = True  # ground control clears us
    fsm.fire(FlightEvent.TAKE_OFF)
    assert fsm.state is FlightState.TAKEOFF


def test_missing_event_raises_invalid_not_guard() -> None:
    # An event with no table entry is InvalidTransition -- guards only exist on
    # real entries, so structure is always decided before policy.
    fsm = FlightStateMachine()  # PREFLIGHT, where only TAXI_OUT is valid
    with pytest.raises(InvalidTransition):
        fsm.fire(FlightEvent.SHUT_DOWN)


def test_allowed_events_and_can_fire() -> None:
    # Introspection helpers for callers doing validation (not used by fire()).
    fsm = FlightStateMachine()  # PREFLIGHT
    assert fsm.allowed_events() == {FlightEvent.TAXI_OUT}
    assert fsm.can_fire(FlightEvent.TAXI_OUT)
    assert not fsm.can_fire(FlightEvent.CLIMB)


def test_terminal_state_has_no_events() -> None:
    fsm = FlightStateMachine(initial=FlightState.SHUTDOWN)
    assert fsm.is_terminal
    assert fsm.allowed_events() == set()
    with pytest.raises(InvalidTransition):
        fsm.fire(FlightEvent.TAXI_OUT)


def test_real_table_is_valid() -> None:
    # The shipped table must pass its own import-time check.
    fsm_mod._validate()  # should not raise


def test_validate_catches_unreachable_state() -> None:
    broken = _copy_table()
    broken[FlightState.PREFLIGHT] = {}  # nothing is reachable from the start now
    with pytest.raises(InvalidStateMachine):
        fsm_mod._validate(broken)


def test_validate_catches_missing_state() -> None:
    broken = _copy_table()
    del broken[FlightState.SHUTDOWN]  # a state with no table entry
    with pytest.raises(InvalidStateMachine):
        fsm_mod._validate(broken)


def test_custom_transitions_table() -> None:
    # "Update = delete + create": build a variant table and inject it. Here we add
    # a new edge -- PREFLIGHT can now shut down directly.
    custom = _copy_table()
    custom[FlightState.PREFLIGHT] = {
        **custom[FlightState.PREFLIGHT],
        FlightEvent.SHUT_DOWN: Transition(FlightState.SHUTDOWN),
    }
    fsm = FlightStateMachine(transitions=custom)
    fsm.fire(FlightEvent.SHUT_DOWN)  # not legal on the default table; is here
    assert fsm.state is FlightState.SHUTDOWN


def test_custom_table_validated_on_construction() -> None:
    broken = _copy_table()
    broken[FlightState.PREFLIGHT] = {}  # strands the rest of the graph
    with pytest.raises(InvalidStateMachine):
        FlightStateMachine(transitions=broken)


def test_add_and_remove_transition() -> None:
    fsm = FlightStateMachine()
    fsm.add_transition(FlightState.PREFLIGHT, FlightEvent.SHUT_DOWN, FlightState.SHUTDOWN)
    assert fsm.can_fire(FlightEvent.SHUT_DOWN)  # newly added edge
    fsm.remove_transition(FlightState.PREFLIGHT, FlightEvent.SHUT_DOWN)
    assert not fsm.can_fire(FlightEvent.SHUT_DOWN)


def test_edit_does_not_leak_into_module_default() -> None:
    fsm = FlightStateMachine()  # default table
    fsm.add_transition(FlightState.PREFLIGHT, FlightEvent.SHUT_DOWN, FlightState.SHUTDOWN)
    # a fresh default machine must not see the first machine's edit
    other = FlightStateMachine()
    assert not other.can_fire(FlightEvent.SHUT_DOWN)


def test_remove_that_strands_a_state_is_rejected() -> None:
    fsm = FlightStateMachine()
    with pytest.raises(InvalidStateMachine):
        fsm.remove_transition(FlightState.PREFLIGHT, FlightEvent.TAXI_OUT)


def test_go_around_and_history() -> None:
    fsm = FlightStateMachine(cleared_for_takeoff=True)
    for event in (
        FlightEvent.TAXI_OUT,
        FlightEvent.TAKE_OFF,
        FlightEvent.EMERGENCY_RETURN,  # takeoff -> landing
        FlightEvent.GO_AROUND,         # landing -> takeoff
    ):
        fsm.fire(event)
    assert fsm.state is FlightState.TAKEOFF
    assert fsm.history.count(FlightState.TAKEOFF) == 2

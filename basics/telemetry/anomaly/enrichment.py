"""Real-time metadata enrichment: aircraft metadata, maintenance history,
weather, firmware, operator, and flight plan -- see anomaly/README.md
"Real-Time Metadata Enrichment" for why each source gets a join pattern, and
stores.py for which real backing store each simulated update stands in for.

Two patterns are used, not one-per-source, because PyFlink's Python
DataStream API doesn't expose Flink's Async I/O operator (Java/Scala-only) --
weather was redesigned from "async lookup + cache" to the same broadcast
mechanism as everything else keyed by aircraft_id, just keyed by grid cell
instead and refreshed periodically instead of on request:

- **Broadcast state** -- aircraft metadata, maintenance history, weather,
  firmware, operator. All small enough to replicate to every task and either
  aircraft-keyed or low-cardinality (grid cells, operators). See
  :class:`AircraftContextEnrichment`.
- **Temporal (versioned) join** -- flight plan. Must reflect the plan
  version active *at event time*, not the latest one, so broadcast state
  (which only ever holds "the current value") is the wrong shape. See
  :class:`FlightPlanEnrichment`.
"""

import json

from pyflink.common import Types
from pyflink.datastream.functions import (
    KeyedBroadcastProcessFunction,
    KeyedCoProcessFunction,
    MapFunction,
)
from pyflink.datastream.state import MapStateDescriptor

from stores import (
    ConfigServiceStore,
    MaintenanceHistoryStore,
    PostgresAircraftStore,
    PostgresFlightPlanStore,
    WeatherStore,
)

AIRCRAFT_METADATA_DESCRIPTOR = MapStateDescriptor(
    "aircraft_metadata", Types.STRING(), Types.STRING()
)
MAINTENANCE_HISTORY_DESCRIPTOR = MapStateDescriptor(
    "maintenance_history", Types.STRING(), Types.STRING()
)
WEATHER_DESCRIPTOR = MapStateDescriptor("weather", Types.STRING(), Types.STRING())
FIRMWARE_DESCRIPTOR = MapStateDescriptor("firmware", Types.STRING(), Types.STRING())
OPERATOR_DESCRIPTOR = MapStateDescriptor("operator", Types.STRING(), Types.STRING())


def _synthetic_grid_cell(aircraft_id: str) -> str:
    """Stand-in for a GPS-derived weather grid cell.

    Real weather enrichment keys off (lat, lon) carried on the telemetry
    event itself. The simulated source in events.py has no location field
    and is intentionally left untouched (see README "Where this fits"), so
    this demo derives a fixed cell per aircraft instead of a real one.
    """
    return f"grid-{hash(aircraft_id) % 3}"


# -------------------------------------------------------------------------
# Broadcast lane: aircraft metadata, maintenance history, weather,
# firmware, operator
# -------------------------------------------------------------------------


class SimulatedContextUpdateSource(MapFunction):
    """Fabricates updates for every broadcast source, tagged by "kind" --
    matching how they are pushed through a single ``.broadcast(...)`` call
    carrying five MapStateDescriptors (see :class:`AircraftContextEnrichment`).
    Each kind delegates to the store in stores.py it stands in for.
    """

    KINDS = ["aircraft_metadata", "maintenance_history", "weather", "firmware", "operator"]

    def map(self, value):
        kind = self.KINDS[value % len(self.KINDS)]
        aircraft_id = f"eVTOL-{(value % 3) + 1}"

        if kind == "aircraft_metadata":
            payload = PostgresAircraftStore.aircraft_metadata_update(aircraft_id)
        elif kind == "maintenance_history":
            payload = MaintenanceHistoryStore.maintenance_update(aircraft_id)
        elif kind == "firmware":
            payload = ConfigServiceStore.firmware_update(aircraft_id)
        elif kind == "operator":
            payload = ConfigServiceStore.operator_update(
                PostgresAircraftStore.OPERATORS[value % 3]
            )
        else:
            payload = WeatherStore.weather_update(_synthetic_grid_cell(aircraft_id))

        return json.dumps(payload)


class AircraftContextEnrichment(KeyedBroadcastProcessFunction):
    """Attaches aircraft metadata, maintenance history, weather, firmware,
    and operator to each event.

    ``process_broadcast_element`` writes updates into the matching broadcast
    state; ``process_element`` reads all of them and merges them into the
    event under an "enrichment" key. Operator is resolved one hop indirect --
    via the aircraft metadata's ``operator_id`` -- since telemetry itself
    carries no operator field. If an aircraft's telemetry arrives before its
    first broadcast update, the corresponding field is enriched as ``None``
    rather than blocking -- the "enrich with nulls until the broadcast
    catches up" answer named in the README's open questions.
    """

    def process_broadcast_element(self, value, ctx):
        update = json.loads(value)
        kind = update["kind"]

        if kind == "aircraft_metadata":
            ctx.get_broadcast_state(AIRCRAFT_METADATA_DESCRIPTOR).put(
                update["aircraft_id"], value
            )
        elif kind == "maintenance_history":
            ctx.get_broadcast_state(MAINTENANCE_HISTORY_DESCRIPTOR).put(
                update["aircraft_id"], value
            )
        elif kind == "weather":
            ctx.get_broadcast_state(WEATHER_DESCRIPTOR).put(update["grid_cell"], value)
        elif kind == "firmware":
            ctx.get_broadcast_state(FIRMWARE_DESCRIPTOR).put(update["aircraft_id"], value)
        elif kind == "operator":
            ctx.get_broadcast_state(OPERATOR_DESCRIPTOR).put(update["operator_id"], value)

    def process_element(self, value, ctx):
        event = json.loads(value)
        aircraft_id = event["aircraft_id"]
        grid_cell = _synthetic_grid_cell(aircraft_id)

        metadata_state = ctx.get_broadcast_state(AIRCRAFT_METADATA_DESCRIPTOR)
        maintenance_state = ctx.get_broadcast_state(MAINTENANCE_HISTORY_DESCRIPTOR)
        weather_state = ctx.get_broadcast_state(WEATHER_DESCRIPTOR)
        firmware_state = ctx.get_broadcast_state(FIRMWARE_DESCRIPTOR)
        operator_state = ctx.get_broadcast_state(OPERATOR_DESCRIPTOR)

        metadata = (
            json.loads(metadata_state.get(aircraft_id))
            if metadata_state.contains(aircraft_id)
            else None
        )
        operator_id = metadata["operator_id"] if metadata else None

        event["enrichment"] = {
            "aircraft_metadata": metadata,
            "maintenance_history": json.loads(maintenance_state.get(aircraft_id))
            if maintenance_state.contains(aircraft_id)
            else None,
            "weather": json.loads(weather_state.get(grid_cell))
            if weather_state.contains(grid_cell)
            else None,
            "firmware": json.loads(firmware_state.get(aircraft_id))
            if firmware_state.contains(aircraft_id)
            else None,
            "operator": json.loads(operator_state.get(operator_id))
            if operator_id and operator_state.contains(operator_id)
            else None,
        }
        yield json.dumps(event)


# -------------------------------------------------------------------------
# Temporal join lane: flight plan
# -------------------------------------------------------------------------


class SimulatedFlightPlanUpdateSource(MapFunction):
    """Fabricates flight plan version updates, simulating mid-flight amendments."""

    def map(self, value):
        aircraft_id = f"eVTOL-{(value % 3) + 1}"
        return json.dumps(PostgresFlightPlanStore.flight_plan_update(aircraft_id))


class FlightPlanEnrichment(KeyedCoProcessFunction):
    """As-of-event-time flight plan join.

    Keeps every known plan version per aircraft in keyed state (not
    broadcast -- see README "Flight plan" section for why) and picks the
    latest version whose ``valid_from_ms`` is at or before the telemetry
    event's own timestamp, so a plan amendment only affects events at or
    after it, never events already in flight ahead of it.

    Version pruning is an open question named in the README, not solved
    here -- state grows with every amendment.
    """

    def __init__(self) -> None:
        self.plan_versions_state = None  # set in open(); MapState[valid_from_ms, plan_json]

    def open(self, runtime_context) -> None:
        self.plan_versions_state = runtime_context.get_map_state(
            MapStateDescriptor("flight_plan_versions", Types.STRING(), Types.STRING())
        )

    def process_element1(self, value, ctx):
        # main enriched telemetry stream
        event = json.loads(value)
        event_ts = event["timestamp"]

        # PROD GOTCHA: O(versions) scan on every single event. Fine while a
        # flight has a handful of amendments; without the pruning policy
        # named as an open question in the README, a long-lived aircraft_id
        # with years of accumulated plan versions turns this into an
        # unbounded-latency hot path, not just an unbounded-memory one.
        best_version_json = None
        best_valid_from = -1
        for valid_from_str, plan_json in self.plan_versions_state.items():
            valid_from = int(valid_from_str)
            if valid_from <= event_ts and valid_from > best_valid_from:
                best_valid_from = valid_from
                best_version_json = plan_json

        event.setdefault("enrichment", {})["flight_plan"] = (
            json.loads(best_version_json) if best_version_json else None
        )
        yield json.dumps(event)

    def process_element2(self, value, ctx):
        # flight plan version update
        update = json.loads(value)
        self.plan_versions_state.put(str(update["valid_from_ms"]), value)

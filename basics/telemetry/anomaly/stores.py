"""Fake clients standing in for the real metadata stores named in the design
doc (see anomaly/README.md "Field -> store -> pattern"). Telemetry itself
(vehicle_id, timestamp, motor_temp, rpm, ...) arrives over Kafka; everything
here is data an ML model or rules engine needs that Kafka never carries.

Each class models one store's *shape*, not its wire protocol -- an in-memory
dict standing in for a table/collection, so the enrichment code that calls
these can be swapped to a real Postgres/DynamoDB/Redis/config-service client
later without changing the Flink DAG, the same "swap the client, keep the
pipeline" idiom `worker_queues/io_workloads_fixed.py` already uses for its
SQLAlchemy sink.

Deliberately NOT here: battery age. It's `event_time - battery_install_date`,
computed from a field this module *does* provide -- not looked up as its own
value, because a stored "age" is stale the instant it's written. Computing it
is feature engineering's job, not enrichment's (see README "Open questions").

Deliberately NOT here: a feature store client. Precomputed features (rolling
stats, historical failure rates) are a feature-engineering concern, scoped
out of this round -- see README "Status".

PROD GOTCHAS, not modeled by any fake client below:
- No retry/circuit-breaker around any of these calls. `rag/v2/knowledge/bus/`
  already has one for exactly this "external dependency in a hot path"
  problem -- a real CDC/poll source feeding the broadcast stream should sit
  behind it, not call out unprotected.
- No connection pooling. A real Postgres/DynamoDB/Redis client is shared
  across the source's parallel instances, not one-per-call.
- No schema evolution story. A field rename/type change in the real
  aircraft table breaks every consumer of `aircraft_metadata_update` with no
  compatibility layer -- worth an Avro/Protobuf schema registry in front of
  the broadcast stream before this goes anywhere real.
"""

import random
import time


class PostgresAircraftStore:
    """Models a Postgres `aircraft` dimension table.

    Static, relational, low write volume -- manufacturing specs and the
    aircraft's operator, keyed by aircraft_id. Real deployment: a plain
    `SELECT ... WHERE aircraft_id = ...`, or more likely a CDC feed (e.g.
    Debezium) turning row changes into the broadcast update stream this
    demo fabricates directly.
    """

    OPERATORS = ["SkyLift Air", "Metro Aviation", "Coastal eVTOL"]

    @staticmethod
    def aircraft_metadata_update(aircraft_id: str) -> dict:
        return {
            "kind": "aircraft_metadata",
            "aircraft_id": aircraft_id,
            "model": "eVTOL-X1",
            "battery_chemistry": "LiFePO4",
            "motor_model": "MX-200",
            "manufacturing_date": "2023-06-01",
            "battery_install_date": "2025-01-15",
            "operator_id": PostgresAircraftStore.OPERATORS[hash(aircraft_id) % 3],
            "max_battery_temp_c": 60.0,
            "max_rotor_vibration_hz": 35.0,
        }


class ConfigServiceStore:
    """Models a small reference/configuration service.

    Firmware and operator reference data are grouped here (not in Postgres)
    because they're typically owned by a different system than the aircraft
    dimension table -- a fleet-management/OTA service for firmware, an
    org-directory service for operators. Kept as two separate lookups rather
    than one blob so an outage or schema change in one doesn't require
    touching the other (same reasoning as aircraft metadata vs. maintenance
    history in README).
    """

    @staticmethod
    def firmware_update(aircraft_id: str) -> dict:
        return {
            "kind": "firmware",
            "aircraft_id": aircraft_id,
            "firmware_version": f"2.{hash(aircraft_id) % 5}.0",
            "released_at": int(time.time() * 1000) - 7 * 86_400_000,
        }

    @staticmethod
    def operator_update(operator_id: str) -> dict:
        return {
            "kind": "operator",
            "operator_id": operator_id,
            "operator_name": operator_id,
            "country": "US",
        }


class MaintenanceHistoryStore:
    """Models a DynamoDB/Cassandra maintenance-events table.

    Append-heavy, one partition per aircraft_id, queried for "recent
    history" rather than a single row -- the write pattern wide-column
    stores are built for and Postgres isn't (see README comparison table).

    PROD GOTCHA: aircraft_id as the sole partition key means one aircraft's
    maintenance events all land on one partition -- fine at "dozens of
    events per aircraft," a hot-partition risk if any single aircraft
    accumulates enough history to concentrate read/write load. A real
    schema would likely add a time-bucket to the partition key.
    """

    @staticmethod
    def maintenance_update(aircraft_id: str) -> dict:
        return {
            "kind": "maintenance_history",
            "aircraft_id": aircraft_id,
            "last_service_ts": int(time.time() * 1000) - 86_400_000,
            "hours_since_service": round(random.uniform(5.0, 200.0), 1),
        }


class WeatherStore:
    """Models a Redis cache in front of an external weather API.

    Redis's role is exactly its usual one: absorb repeated reads for the
    same key (grid cell) so the slow/rate-limited external call happens on
    a refresh interval, not per event. The demo skips the API call and
    fabricates a reading directly -- the periodic *refresh* behavior is what
    this models, not the HTTP round trip.

    PROD GOTCHA: no TTL/eviction is modeled here -- this always returns a
    fresh reading. A real Redis-backed cache needs an explicit TTL matching
    the refresh interval, or a stale grid cell silently reports
    weeks-old weather forever once the refresh job stops running.
    """

    @staticmethod
    def weather_update(grid_cell: str) -> dict:
        return {
            "kind": "weather",
            "grid_cell": grid_cell,
            "ambient_temp_c": round(random.uniform(-5.0, 35.0), 1),
            "wind_speed_kph": round(random.uniform(0.0, 60.0), 1),
        }


class PostgresFlightPlanStore:
    """Models a Postgres `flight_plan` table with `valid_from`/`valid_to` columns.

    Versioned rows map directly onto the temporal join in
    :class:`enrichment.FlightPlanEnrichment` -- amendments are new rows, not
    updates in place, so "the version active at event time" is a plain
    `WHERE valid_from <= event_time ORDER BY valid_from DESC LIMIT 1` in the
    real system.
    """

    AIRPORTS = ["KJFK", "KLAX", "KORD"]

    @staticmethod
    def flight_plan_update(aircraft_id: str) -> dict:
        return {
            "kind": "flight_plan",
            "aircraft_id": aircraft_id,
            "valid_from_ms": int(time.time() * 1000),
            "planned_altitude_m": random.choice([150, 300, 450]),
            "flight_phase": random.choice(["climb", "cruise", "descent", "hover"]),
            "departure_airport": random.choice(PostgresFlightPlanStore.AIRPORTS),
            "arrival_airport": random.choice(PostgresFlightPlanStore.AIRPORTS),
        }

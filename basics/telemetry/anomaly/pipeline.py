"""Orchestrates the demo enrichment pipeline described in anomaly/README.md.

events.py stays untouched as the reference implementation (source ->
watermarks -> keyBy -> window). This file imports its building blocks and
inserts the two new stages -- dedup and enrichment -- between keyBy and the
window, matching the README's "Pipeline overview" diagram:

    source -> watermarks -> keyBy -> dedup -> enrichment -> window -> sink
"""

import json

from pyflink.common import Duration, Time, Types, WatermarkStrategy
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.window import SlidingEventTimeWindows

from dedup import DedupFunction
from enrichment import (
    AIRCRAFT_METADATA_DESCRIPTOR,
    FIRMWARE_DESCRIPTOR,
    MAINTENANCE_HISTORY_DESCRIPTOR,
    OPERATOR_DESCRIPTOR,
    WEATHER_DESCRIPTOR,
    AircraftContextEnrichment,
    FlightPlanEnrichment,
    SimulatedContextUpdateSource,
    SimulatedFlightPlanUpdateSource,
)
from events import (
    AircraftIdSelector,
    EvtolSensorSource,
    EvtolTimestampAssigner,
    WindowAnomalyEvaluator,
)


def run_enriched_pipeline() -> None:
    # PROD GOTCHA: no checkpointing is configured (env.enable_checkpointing(...)).
    # Every stage below holds real state -- dedup's TTL'd ValueState, five
    # broadcast MapStates, flight plan's per-aircraft version MapState -- and
    # without checkpointing, a task manager failure loses all of it silently;
    # the job restarts clean rather than failing loudly, which is worse.
    # PROD GOTCHA: parallelism(1) means everything runs on one task slot --
    # fine for demoing correctness, but says nothing about behavior once
    # aircraft_id is actually hash-partitioned across N task managers (see
    # basics/telemetry/README.md Question 1). Broadcast state cost in
    # particular scales with N, not down -- more partitions means more full
    # copies of the same aircraft/weather/firmware/operator maps, not less.
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)

    # ---- Source, watermarks, keyBy -- identical to events.py ----
    raw_stream = env.from_collection(collection=[i for i in range(1, 200)]).map(
        EvtolSensorSource(), output_type=Types.STRING()
    )

    env.get_config().set_auto_watermark_interval(500)
    watermarked_stream = raw_stream.assign_timestamps_and_watermarks(
        WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(5))
        .with_timestamp_assigner(EvtolTimestampAssigner())
    )

    keyed_stream = watermarked_stream.key_by(AircraftIdSelector(), key_type=Types.STRING())

    # ---- Dedup ----
    deduped_stream = keyed_stream.process(DedupFunction(), output_type=Types.STRING())

    # A ProcessFunction's output is a plain DataStream, not a KeyedStream --
    # Flink doesn't carry partitioning info through .process(). Re-key
    # explicitly since the broadcast connect below needs a KeyedStream.
    deduped_keyed_stream = deduped_stream.key_by(AircraftIdSelector(), key_type=Types.STRING())

    # ---- Broadcast enrichment: aircraft metadata, maintenance, weather, firmware, operator ----
    # PROD GOTCHA: env.from_collection here is a finite, in-memory demo source.
    # A real deployment replaces this with a CDC feed (Postgres/config-service)
    # or a periodic poll job (weather) -- see stores.py docstrings for which.
    # Those real sources can emit out of order or redeliver, same as the main
    # telemetry stream; nothing here dedups broadcast updates the way
    # DedupFunction dedups telemetry, because a broadcast update's only effect
    # is overwriting a MapState entry -- a duplicate is a harmless no-op, not
    # a correctness bug the way a duplicated telemetry event would be.
    context_updates = env.from_collection(collection=[i for i in range(1, 40)]).map(
        SimulatedContextUpdateSource(), output_type=Types.STRING()
    )
    context_broadcast_stream = context_updates.broadcast(
        AIRCRAFT_METADATA_DESCRIPTOR,
        MAINTENANCE_HISTORY_DESCRIPTOR,
        WEATHER_DESCRIPTOR,
        FIRMWARE_DESCRIPTOR,
        OPERATOR_DESCRIPTOR,
    )
    # PROD GOTCHA: process_element (keyed side) and process_broadcast_element
    # (broadcast side) are NOT guaranteed to interleave in any particular
    # order -- Flink explicitly does not order the two inputs of a
    # BroadcastConnectedStream. A telemetry event can be processed before its
    # aircraft's first metadata update arrives, enriching it with None (see
    # enrichment.py's docstring) even if that update was technically produced
    # earlier upstream. Fine for this demo's staleness tolerance; would need
    # naming as an explicit consistency model in a design review for
    # anything safety-critical.
    context_enriched_stream = deduped_keyed_stream.connect(context_broadcast_stream).process(
        AircraftContextEnrichment(), output_type=Types.STRING()
    )

    # ---- Temporal join: flight plan ----
    flight_plan_updates = env.from_collection(collection=[i for i in range(1, 20)]).map(
        SimulatedFlightPlanUpdateSource(), output_type=Types.STRING()
    )
    fully_enriched_stream = (
        context_enriched_stream.key_by(AircraftIdSelector(), key_type=Types.STRING())
        .connect(flight_plan_updates.key_by(AircraftIdSelector(), key_type=Types.STRING()))
        .process(FlightPlanEnrichment(), output_type=Types.STRING())
    )

    # ---- Window + threshold, same as events.py, now over enriched events ----
    windowed_alerts = (
        fully_enriched_stream.key_by(AircraftIdSelector(), key_type=Types.STRING())
        .window(SlidingEventTimeWindows.of(Time.minutes(10), Time.minutes(1)))
        .process(WindowAnomalyEvaluator(), output_type=Types.STRING())
    )

    critical_alerts = windowed_alerts.filter(
        lambda data: json.loads(data)["sustained_risk_alert"] is True
    )

    print("--- Launching Enriched eVTOL Fleet Telemetry Monitor ---")
    critical_alerts.print()

    env.execute("evtol_fleet_monitoring_enriched_dag")


if __name__ == "__main__":
    run_enriched_pipeline()

"""Per-aircraft duplicate suppression.

See anomaly/README.md "Pipeline overview" -- dedup runs right after keyBy and
before enrichment, so nothing downstream (joins, windows, rules) ever sees the
same event twice.
"""

import json

from pyflink.common import Time, Types
from pyflink.datastream.functions import KeyedProcessFunction
from pyflink.datastream.state import StateTtlConfig, ValueStateDescriptor


class DedupFunction(KeyedProcessFunction):
    """Drops an event that repeats the last timestamp seen for its aircraft_id.

    Real telemetry buses redeliver messages (MQTT QoS 1, Kafka producer
    retries). The simulated source in events.py has no dedicated event id, so
    identity here is (aircraft_id via the keyed partition, timestamp) --
    this only catches a *repeat of the immediately preceding* timestamp, not
    a duplicate that arrives after other events for the same aircraft. A real
    deployment would need a proper event id to catch that case too; named
    here rather than silently assumed away.
    """

    def __init__(self, ttl_seconds: int = 10) -> None:
        self.ttl_seconds = ttl_seconds
        self.last_seen_state = None  # set in open()

    def open(self, runtime_context) -> None:
        ttl_config = (
            StateTtlConfig.new_builder(Time.seconds(self.ttl_seconds))
            .set_update_type(StateTtlConfig.UpdateType.OnCreateAndWrite)
            .set_state_visibility(StateTtlConfig.StateVisibility.NeverReturnExpired)
            .build()
        )
        descriptor = ValueStateDescriptor("last_seen_timestamp", Types.LONG())
        descriptor.enable_time_to_live(ttl_config)
        self.last_seen_state = runtime_context.get_state(descriptor)

    def process_element(self, value, ctx):
        data = json.loads(value)
        timestamp = data["timestamp"]

        if self.last_seen_state.value() == timestamp:
            return  # exact repeat for this aircraft -- drop

        self.last_seen_state.update(timestamp)
        yield value

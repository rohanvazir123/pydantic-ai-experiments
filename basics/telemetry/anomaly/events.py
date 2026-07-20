import json
import random
import time
from pyflink.common import WatermarkStrategy, Duration, Types, Time
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import MapFunction, KeySelector, ProcessWindowFunction
from pyflink.datastream.window import SlidingEventTimeWindows
from pyflink.common.watermark_strategy import TimestampAssigner


# Design skeleton
'''
Source: MQTT => Kafka

Real Time Event Processing:
[Deserialize =>Assign TimeStamps => Watermarks (5s, freq 500ms)
=> KeyBy(vehicle_id) => Sliding Window => deduplication
=> Rules engine
=> ML Inference

Real Time Metadata Enrichment:Flink => [join with] Aircraft metadata, Weather, Flight Plan, Maintenance History

Sink: Kafka, TSDB

Rule Engine/CEP: RPM > X [and/or] TEMP > Y [and/or] …
ML inference: Feature Extraction => ML model => Anomaly score => [Alert, retraining feedback]

===============================

Data store and queries: Raw Telemetry => TSDB => [aggregation queries] => UI Dashboard
TimescaleDB: [ Raw hypertable (100 ms data), Continuous aggregates (1s, 1 min, 1 hr), Retention & compression policies ]
Dashboards / ML / Predictive Maintenance


'''

# TODO metada 



# Skeleton code
'''
import: [StreamExecEnv, 
TimeCharacteristic, StreamTableEnv, Schema, Kafka, Json, 
FlinkKafkaConsumer, FlinkKafkaProducer.FlinkPattern, CEP, 
PatternProcessFunction, ProcessFunction, timedelta, json, SimpleStringSchema]

# ML
model = joblib.load(“anomaly_model.pkl”)

env =  StreamExecutionEnv.get_exec_env(); 
env. set_stream_time_characteristic(TimeCharactistic.EventTime); 
env.set_parallelism(4); 
t_env = StreamTableEnv.create(env)

events_source(): 
return env.add_source(
    FlinkKafkaConsumer(topics=”evtol_telemetry”, properties=<sources>, 
    deserialization_schem=Json()).assign_timestamps_and_watermarks(lambda event: event[‘timestemp’]))

_def_pattern(): 
return (Pattern.begin(<some pattern>)
    .where(lambda event: <?>).followed_by(<some pat>)
    .where (lambda event: do something with event)
    .within(timedelta(min=6)))

apply_pattern (stream): 
    pattern = _def_pattern(); 
    pattern_stream = CEP.pattern(stream, pattern)

dedup_events(anomaly_stream): class DedupFunc(ProcessFunction): 
    def open(self, runtime_context), def process_element(self, value, ctx) -> value, 
    return anomaly_stream,process(DedupFunc()))

apply_ml_model(anomaly_stream): 
    class MLModelAnomalyDetection(ProcessFunction): 
        def process_element(self, value, ctx):
            features = np.array(value[‘this’], value[‘that’], …).reshape(1, -1), 
            anomaly_score = model.predict(features[0,1]); 
            yield ({anomaly_score > 0.8, anomaly_score, alert})
; return anomaly_stream.process(MLModelAnomalyDetection)

sink_results(anomaly_stream): anomaly_stream,map(lambda event: json.dumps(event)).add_sink(FlinkKafkaProducer(topic=”anomaly_detection”, serialization_schema=SimpleStringSchema, properties=<sinks>); anomaly_schema.map(lambda event: f”INSERT into anomaly_table VALUES ({event[‘this], event[‘that’], ….})”)

main: events_stream = events_source(), anomaly_event_stream = apply_pattern(events_stream); deduped_anomaly_events = dedup_events(anomaly_event_stream);sink_results(deduped_anomaly_events); 

anomaly_events_from_ml = apply_ml_model(anomaly_event_stream)sink_results(deduped_anomaly_events_from_ml)

env.execute(“flink cep anomaly detection”)

'''

# -------------------------------------------------------------------------
# 1. Custom Core Functions & Selectors
# -------------------------------------------------------------------------

class EvtolSensorSource(MapFunction):
    """Simulates real-time eVTOL telemetry data."""
    def map(self, value):
        # Normal ranges: Temp 30-45C. Occasional anomalies injected.
        is_anomaly = random.random() < 0.15
        return json.dumps({
            "aircraft_id": f"eVTOL-{random.randint(1, 3)}",
            "timestamp": int(time.time() * 1000), # Unix time in milliseconds
            "battery_temp": random.uniform(65.0, 95.0) if is_anomaly else random.uniform(30.0, 45.0),
            "rotor_vibration": random.uniform(40.0, 60.0) if is_anomaly else random.uniform(10.0, 20.0)
        })

class EvtolTimestampAssigner(TimestampAssigner):
    """Extracts millisecond timestamp from the JSON string payload."""
    def extract_timestamp(self, element, record_timestamp):
        data = json.loads(element)
        return int(data["timestamp"])

class AircraftIdSelector(KeySelector):
    """Partitions the data stream by aircraft identifier."""
    def getKey(self, value):
        return json.loads(value)["aircraft_id"]

class WindowAnomalyEvaluator(ProcessWindowFunction):
    """Analyzes metrics inside the sliding window for sustained risks."""
    def process(self, key, context, elements):
        total_temp = 0.0
        count = 0
        
        for element in elements:
            data = json.loads(element)
            total_temp += data["battery_temp"]
            count += 1
            
        avg_temp = total_temp / count if count > 0 else 0.0
        is_sustained_anomaly = avg_temp > 55.0 
        
        result = {
            "aircraft_id": key,
            "window_start_ms": context.window().get_start(),
            "window_end_ms": context.window().get_end(),
            "avg_battery_temp": round(avg_temp, 2),
            "data_points_analyzed": count,
            "sustained_risk_alert": is_sustained_anomaly
        }
        return [json.dumps(result)]

# -------------------------------------------------------------------------
# 2. Main Execution Pipeline
# -------------------------------------------------------------------------

def run_evtol_pipeline():
    # Set up environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)


    # Step 1: Ingest simulated live data
    raw_stream = env.from_collection(
        collection=[i for i in range(1, 200)]
    ).map(EvtolSensorSource(), output_type=Types.STRING())

    # Step 2: Apply 5-second Bounded Out-of-Orderness Watermarks
    # By default, Flink checks and emits a new watermark every 200 milliseconds.
    # Change the periodic watermark interval to 500 milliseconds
    env.get_config().set_auto_watermark_interval(500)
    watermarked_stream = raw_stream.assign_timestamps_and_watermarks(
        WatermarkStrategy
        .for_bounded_out_of_orderness(Duration.of_seconds(5))
        .with_timestamp_assigner(EvtolTimestampAssigner())
    )

    # Step 3: Partition stream by Aircraft ID
    keyed_stream = watermarked_stream.key_by(
        AircraftIdSelector(),
        key_type=Types.STRING()
    )

    # Step 4: Evaluate 10-minute windows sliding every 1 minute
    windowed_alerts = keyed_stream \
        .window(SlidingEventTimeWindows.of(Time.minutes(10), Time.minutes(1))) \
        .process(WindowAnomalyEvaluator(), output_type=Types.STRING())

    # Step 5: Filter for sustained high risks and print to console
    critical_alerts = windowed_alerts.filter(
        lambda data: json.loads(data)["sustained_risk_alert"] == True
    )

    print("--- Launching Live eVTOL Fleet Telemetry Monitor ---")
    critical_alerts.print()

    env.execute("evtol_fleet_monitoring_dag")

if __name__ == "__main__":
    run_evtol_pipeline()

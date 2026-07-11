# read kafka message true topic, schema, return python dict for the realtime process

import io
from pathlib import Path

import avro.schema
from avro.io import BinaryDecoder, DatumReader
from kafka import KafkaConsumer

# find and load avro schema
def _resolve_schema_path(schema_path: str) -> Path:
    path = Path(schema_path)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    parents = Path(__file__).resolve().parents
    if len(parents) > 4:
        repo_root = parents[4]
        repo_path = repo_root / path
        if repo_path.exists():
            return repo_path

    return cwd_path



def load_avro_schema(schema_path: str) -> avro.schema.Schema:
    resolved = _resolve_schema_path(schema_path)
    with resolved.open("r", encoding="utf-8") as f:
        return avro.schema.parse(f.read())

# decode kafka bytes to dict
def deserialize_avro(schema: avro.schema.Schema, raw_bytes: bytes) -> dict:
    reader = DatumReader(schema)
    buf = io.BytesIO(raw_bytes)
    decoder = BinaryDecoder(buf)
    return reader.read(decoder)


class StreamingKafkaConsumer:
    def __init__(
        self,
        bootstrap_servers: str = "localhost:29092",
        topic: str = "reid_output",
        group_id: str = "streaming_consumer_group", # use group id to manage consumer group
        schema_path: str = "src/contracts/reid_output.avsc",
    ):
        self.topic = topic
        self.schema = load_avro_schema(schema_path)
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers.split(","),
            group_id=group_id,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            max_partition_fetch_bytes=100 * 1024 * 1024,
            fetch_min_bytes=1,
            session_timeout_ms=30000,
            heartbeat_interval_ms=3000,
        )

    def poll(self, timeout_ms: int = 1000, max_records: int = 50) -> list[dict]:
        messages = []
        raw = self.consumer.poll(timeout_ms=timeout_ms, max_records=max_records)
        for _, records in raw.items():
            for record in records:
                messages.append(deserialize_avro(self.schema, record.value))
        return messages

    def close(self):
        self.consumer.close()

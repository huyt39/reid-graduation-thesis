# read message from kafka and decode avro bytes -> python dict

import io
from pathlib import Path

import avro.schema
from avro.io import BinaryDecoder, DatumReader
from kafka import KafkaConsumer


def _resolve_schema_path(schema_path: str) -> Path:
    path = Path(schema_path)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    repo_root = Path(__file__).resolve().parents[4]
    repo_candidate = repo_root / path
    if repo_candidate.exists():
        return repo_candidate
    return cwd_candidate


def load_avro_schema(schema_path: str) -> avro.schema.Schema:
    resolved = _resolve_schema_path(schema_path)
    with resolved.open("r", encoding="utf-8") as f:
        return avro.schema.parse(f.read())


def deserialize_avro(schema: avro.schema.Schema, raw_bytes: bytes) -> dict:
    reader = DatumReader(schema)
    buf = io.BytesIO(raw_bytes)
    decoder = BinaryDecoder(buf)
    return reader.read(decoder)


class WorkerKafkaConsumer:
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "reid_input",
        group_id: str = "reid_worker_group",
        schema_path: str = "src/contracts/reid_input.avsc",
    ):
        self.topic = topic
        self.schema = load_avro_schema(schema_path)
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers.split(","),
            group_id=group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            max_partition_fetch_bytes=20 * 1024 * 1024,
        )

    def poll(self, timeout_ms: int = 1000) -> list[dict]:
        messages = []
        raw = self.consumer.poll(timeout_ms=timeout_ms)
        for _, records in raw.items():
            for record in records:
                messages.append(deserialize_avro(self.schema, record.value))
        return messages

    def close(self):
        self.consumer.close()

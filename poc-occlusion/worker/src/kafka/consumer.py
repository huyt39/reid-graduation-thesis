import io

import avro.schema
from avro.io import BinaryDecoder, DatumReader
from kafka import KafkaConsumer


def load_avro_schema(schema_path: str) -> avro.schema.Schema:
    with open(schema_path, "r") as f:
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
        schema_path: str = "contracts/reid_input_v2.avsc",
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
        for tp, records in raw.items():
            for record in records:
                msg = deserialize_avro(self.schema, record.value)
                messages.append(msg)
        return messages

    def close(self):
        self.consumer.close()

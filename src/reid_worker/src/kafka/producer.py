from kafka import KafkaProducer

from src.kafka.serialization import load_avro_schema, serialize_avro

_STRING_ATTRIBUTE_FIELDS = (
    "gender",
    "age_child",
    "backpack",
    "sidebag",
    "hat",
    "glasses",
    "sleeve",
    "lower",
)


class WorkerKafkaProducer:
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "reid_output",
        schema_path: str = "src/contracts/reid_output.avsc",
    ):
        self.topic = topic
        self.schema = load_avro_schema(schema_path)
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers.split(","),
            max_request_size=20 * 1024 * 1024,
            linger_ms=10,
            batch_size=16384,
            acks=1,
        )

    def _normalize_tracked_person(self, person: dict) -> dict:
        quality = person.get("quality")
        if quality is not None:
            quality = {
                "v_avg": float(quality.get("v_avg", 0.0)),
                "embedding_consistency": float(quality.get("embedding_consistency", 0.0)),
                "overall_consistency": float(quality.get("overall_consistency", 0.0)),
                "good_frame_ratio": float(quality.get("good_frame_ratio", 0.0)),
            }

        attributes = person.get("attributes")
        if attributes is not None:
            attributes = {str(k): str(v) for k, v in attributes.items()}

        normalized = {
            "person_id": int(person["person_id"]),
            "bbox": [float(v) for v in person["bbox"]],
            "confidence": float(person["confidence"]),
            "tracklet_id": person.get("tracklet_id"),
            "tracklet_state": person.get("tracklet_state"),
            "snapshot_key": person.get("snapshot_key"),
            "visibility_score": float(person.get("visibility_score", 0.0)),
            "quality": quality,
            "attributes": attributes,
        }
        for field in _STRING_ATTRIBUTE_FIELDS:
            value = person.get(field)
            normalized[field] = "unknown" if value in (None, "") else str(value)
            normalized[f"{field}_confidence"] = float(person.get(f"{field}_confidence", 0.0) or 0.0)
        return normalized

    def send(
        self,
        device_id: str,
        frame_number: int,
        tracked_persons: list[dict],
        image_data: bytes,
        timestamp_ns: int,
    ):
        normalized_persons = [
            self._normalize_tracked_person(person)
            for person in tracked_persons
        ]

        datum = {
            "device_id": str(device_id),
            "frame_number": int(frame_number),
            "tracked_persons": normalized_persons,
            "created_at": int(timestamp_ns),
            "image_data": image_data,
            "schema_version": 2,
        }
        msg_bytes = serialize_avro(self.schema, datum)
        self.producer.send(self.topic, value=msg_bytes)

    def close(self):
        self.producer.flush()
        self.producer.close()

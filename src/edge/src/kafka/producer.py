from kafka import KafkaProducer
from src.kafka.serialization import load_avro_schema, serialize_avro


class EdgeKafkaProducer:
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "reid_input",
        schema_path: str = "src/contracts/reid_input.avsc",
    ):
        self.topic = topic
        self.schema = load_avro_schema(schema_path)
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers.split(","),
            max_request_size=20*1024*1024,
            linger_ms=10,
            batch_size=16384,
            acks=1,
        )

    def send(
        self,
        device_id: str,
        frame_number: int,
        detections: list[dict],
        image_data: bytes,
        timestamp_ns: int,
    ):
        datum = {
            "device_id": device_id,
            "frame_number": frame_number,
            "detections": [
                {
                    "bbox": d["bbox"],
                    "confidence": d["confidence"],
                    "class_id": d["class_id"],
                    "visibility_score": d["visibility_score"],
                    "overlap_ratio": d["overlap_ratio"],
                    "visibility_tag": d.get("visibility_tag", "mid"),
                }
                for d in detections
            ],
            "created_at": timestamp_ns,
            "image_data": image_data,
        }
        msg_bytes = serialize_avro(self.schema, datum)
        self.producer.send(self.topic, value=msg_bytes)

    def send_end_of_stream(
        self,
        device_id: str,
        frame_number: int,
        timestamp_ns: int,
    ):
        datum = {
            "device_id": device_id,
            "frame_number": -1,
            "detections": [],
            "created_at": timestamp_ns,
            "image_data": b"",
        }
        msg_bytes = serialize_avro(self.schema, datum)
        self.producer.send(self.topic, value=msg_bytes)
        self.producer.flush()

    def close(self):
        self.producer.flush()
        self.producer.close()

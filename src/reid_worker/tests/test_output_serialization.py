import src.kafka.producer as producer_module
from src.kafka.producer import WorkerKafkaProducer
from src.kafka.serialization import load_avro_schema, serialize_avro


def test_reid_output_serialization():
    schema = load_avro_schema("src/contracts/reid_output.avsc")
    payload = {
        "device_id": "cam-1",
        "frame_number": 12,
        "tracked_persons": [
            {
                "person_id": 7,
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "confidence": 0.95,
                "gender": "male",
                "gender_confidence": 0.91,
                "age_child": "adult",
                "age_child_confidence": 0.77,
                "backpack": "no",
                "backpack_confidence": 0.88,
                "sidebag": "no",
                "sidebag_confidence": 0.86,
                "hat": "no",
                "hat_confidence": 0.9,
                "glasses": "no",
                "glasses_confidence": 0.92,
                "sleeve": "long",
                "sleeve_confidence": 0.83,
                "lower": "pants",
                "lower_confidence": 0.8,
                "tracklet_id": "tracklet-123",
                "tracklet_state": "matched",
                "snapshot_key": "persons/7/best.jpg",
                "visibility_score": 0.87,
                "quality": {
                    "v_avg": 0.87,
                    "embedding_consistency": 0.93,
                    "overall_consistency": 0.89,
                    "good_frame_ratio": 0.80,
                },
                "attributes": {
                    "gender": "male",
                },
            }
        ],
        "created_at": 123456789,
        "image_data": b"jpeg-bytes",
        "schema_version": 2,
    }

    encoded = serialize_avro(schema, payload)

    assert isinstance(encoded, bytes)
    assert len(encoded) > 0


def test_normalize_tracked_person_defaults_and_types():
    producer = WorkerKafkaProducer.__new__(WorkerKafkaProducer)

    normalized = producer._normalize_tracked_person(
        {
            "person_id": "7",
            "bbox": [1, 2, 3, 4],
            "confidence": "0.95",
            "attributes": {"gender": "male", "age": 25},
        }
    )

    assert normalized["person_id"] == 7
    assert normalized["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert normalized["confidence"] == 0.95
    assert normalized["gender"] == "unknown"
    assert normalized["gender_confidence"] == 0.0
    assert normalized["tracklet_id"] is None
    assert normalized["tracklet_state"] is None
    assert normalized["snapshot_key"] is None
    assert normalized["visibility_score"] == 0.0
    assert normalized["quality"] is None
    assert normalized["attributes"] == {"gender": "male", "age": "25"}
    assert normalized["age_child"] == "unknown"
    assert normalized["age_child_confidence"] == 0.0
    assert normalized["backpack"] == "unknown"
    assert normalized["backpack_confidence"] == 0.0
    assert normalized["sidebag"] == "unknown"
    assert normalized["sidebag_confidence"] == 0.0
    assert normalized["hat"] == "unknown"
    assert normalized["hat_confidence"] == 0.0
    assert normalized["glasses"] == "unknown"
    assert normalized["glasses_confidence"] == 0.0
    assert normalized["sleeve"] == "unknown"
    assert normalized["sleeve_confidence"] == 0.0
    assert normalized["lower"] == "unknown"
    assert normalized["lower_confidence"] == 0.0


def test_normalize_tracked_person_coerces_none_attribute_fields():
    producer = WorkerKafkaProducer.__new__(WorkerKafkaProducer)

    normalized = producer._normalize_tracked_person(
        {
            "person_id": 9,
            "bbox": [1, 2, 3, 4],
            "confidence": 0.8,
            "age_child": None,
            "age_child_confidence": None,
            "hat": "",
            "hat_confidence": None,
        }
    )

    assert normalized["age_child"] == "unknown"
    assert normalized["age_child_confidence"] == 0.0
    assert normalized["hat"] == "unknown"
    assert normalized["hat_confidence"] == 0.0


def test_send_uses_normalized_tracked_persons(monkeypatch):
    captured = {}

    class DummyKafkaProducer:
        def send(self, topic, value):
            captured["topic"] = topic
            captured["value"] = value

    def fake_serialize(schema, datum):
        captured["schema"] = schema
        captured["datum"] = datum
        return b"encoded-message"

    producer = WorkerKafkaProducer.__new__(WorkerKafkaProducer)
    producer.topic = "reid_output"
    producer.schema = object()
    producer.producer = DummyKafkaProducer()

    monkeypatch.setattr(producer_module, "serialize_avro", fake_serialize)

    producer.send(
        device_id="cam-1",
        frame_number=12,
        tracked_persons=[
            {
                "person_id": "7",
                "bbox": [1, 2, 3, 4],
                "confidence": "0.95",
                "attributes": {"gender": "male", "age": 25},
            }
        ],
        image_data=b"jpeg-bytes",
        timestamp_ns=123456789,
    )

    person = captured["datum"]["tracked_persons"][0]

    assert captured["topic"] == "reid_output"
    assert captured["value"] == b"encoded-message"
    assert captured["datum"]["device_id"] == "cam-1"
    assert captured["datum"]["frame_number"] == 12
    assert captured["datum"]["created_at"] == 123456789
    assert captured["datum"]["schema_version"] == 2
    assert person["person_id"] == 7
    assert person["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert person["confidence"] == 0.95
    assert person["gender"] == "unknown"
    assert person["gender_confidence"] == 0.0
    assert person["tracklet_id"] is None
    assert person["tracklet_state"] is None
    assert person["snapshot_key"] is None
    assert person["visibility_score"] == 0.0
    assert person["quality"] is None
    assert person["attributes"] == {"gender": "male", "age": "25"}

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
                "tracklet_id": "tracklet-123",
                "tracklet_state": "matched",
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

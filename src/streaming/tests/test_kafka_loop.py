from src.workers.kafka_loop import _with_snapshot_urls


class StubMinIO:
    def presigned_url(self, object_key: str | None) -> str | None:
        if not object_key:
            return None
        return f"https://example.com/{object_key}"


def test_with_snapshot_urls_enriches_without_mutating_input():
    tracked_persons = [
        {"person_id": 1, "snapshot_key": "persons/1/best.jpg"},
        {"person_id": 2, "snapshot_key": None},
    ]

    enriched = _with_snapshot_urls(tracked_persons, StubMinIO())

    assert tracked_persons[0].get("snapshot_url") is None
    assert enriched[0]["snapshot_url"] == "https://example.com/persons/1/best.jpg"
    assert enriched[1]["snapshot_url"] is None

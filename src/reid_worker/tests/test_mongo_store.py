from src.persistence.mongo_store import MongoPersonStore


def test_attribute_conflict_single_accessory_flip_is_not_strong_under_occlusion():
    attrs_a = {
        "gender": "male",
        "gender_confidence": 0.96,
        "backpack": "no_backpack",
        "backpack_confidence": 0.87,
        "sleeve": "short_sleeve",
        "sleeve_confidence": 0.92,
    }
    attrs_b = {
        "gender": "unknown",
        "gender_confidence": 0.0,
        "backpack": "backpack",
        "backpack_confidence": 0.96,
        "sleeve": "unknown",
        "sleeve_confidence": 0.0,
    }

    assert not MongoPersonStore.attributes_have_strong_conflict(attrs_a, attrs_b)


def test_attribute_conflict_gender_remains_hard_conflict():
    attrs_a = {"gender": "male", "gender_confidence": 0.95}
    attrs_b = {"gender": "female", "gender_confidence": 0.94}

    assert MongoPersonStore.attributes_have_strong_conflict(attrs_a, attrs_b)


def test_attribute_conflict_multiple_non_gender_conflicts_are_strong():
    attrs_a = {
        "backpack": "no_backpack",
        "backpack_confidence": 0.9,
        "sleeve": "short_sleeve",
        "sleeve_confidence": 0.91,
    }
    attrs_b = {
        "backpack": "backpack",
        "backpack_confidence": 0.93,
        "sleeve": "long_sleeve",
        "sleeve_confidence": 0.9,
    }

    assert MongoPersonStore.attributes_have_strong_conflict(attrs_a, attrs_b)

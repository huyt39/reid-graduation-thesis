from src.attributes.attribute_voter import AttributeVoter


def _attrs(**kwargs) -> dict:
    """Helper: build the inference-engine `/attributes/classify` response shape."""
    return {task: {"label": label, "confidence": conf} for task, (label, conf) in kwargs.items()}


def test_tracklet_majority_vote():
    voter = AttributeVoter()
    voter.vote_frame(1, _attrs(gender=("male", 0.9)))
    voter.vote_frame(1, _attrs(gender=("male", 0.8)))
    voter.vote_frame(1, _attrs(gender=("female", 0.7)))
    out = voter.resolve_tracklet(1)
    assert out["gender"][0] == "male"
    assert out["gender"][1] > 0.0


def test_tracklet_empty_returns_empty_dict():
    voter = AttributeVoter()
    assert voter.resolve_tracklet(99) == {}


def test_person_first_assignment():
    voter = AttributeVoter(person_threshold=0.6)
    out = voter.resolve_person(1, {"gender": ("male", 0.8)})
    assert out["gender"] == ("male", 0.8)


def test_person_hysteresis_blocks_flip():
    voter = AttributeVoter(person_threshold=0.7)
    voter.resolve_person(1, {"gender": ("male", 0.9)})
    # One contradicting tracklet — should NOT flip yet.
    out = voter.resolve_person(1, {"gender": ("female", 0.95)})
    assert out["gender"][0] == "male"


def test_person_hysteresis_flips_after_two_consecutive():
    voter = AttributeVoter(person_threshold=0.7)
    voter.resolve_person(1, {"gender": ("male", 0.9)})
    voter.resolve_person(1, {"gender": ("female", 0.85)})
    out = voter.resolve_person(1, {"gender": ("female", 0.90)})
    assert out["gender"][0] == "female"


def test_gender_can_use_task_specific_lower_flip_threshold():
    voter = AttributeVoter(
        person_threshold=0.7,
        flip_threshold=0.85,
        task_flip_thresholds={"gender": 0.65},
    )
    voter.resolve_person(1, {"gender": ("male", 0.97)})
    voter.resolve_person(1, {"gender": ("female", 0.66)})
    out = voter.resolve_person(1, {"gender": ("female", 0.74)})
    assert out["gender"][0] == "female"


def test_non_gender_tasks_keep_conservative_flip_threshold():
    voter = AttributeVoter(
        person_threshold=0.7,
        flip_threshold=0.85,
        task_flip_thresholds={"gender": 0.65},
    )
    voter.resolve_person(1, {"sleeve": ("long_sleeve", 0.92)})
    voter.resolve_person(1, {"sleeve": ("short_sleeve", 0.72)})
    out = voter.resolve_person(1, {"sleeve": ("short_sleeve", 0.74)})
    assert out["sleeve"][0] == "long_sleeve"


def test_person_same_label_reinforces():
    voter = AttributeVoter(person_threshold=0.7)
    voter.resolve_person(1, {"gender": ("male", 0.8)})
    out = voter.resolve_person(1, {"gender": ("male", 1.0)})
    assert out["gender"][0] == "male"
    # Confidence updates via weighted average, so it should rise above the original 0.8.
    assert out["gender"][1] > 0.8


def test_person_low_confidence_blocks_first_assignment():
    voter = AttributeVoter(person_threshold=0.7)
    out = voter.resolve_person(1, {"gender": ("male", 0.3)})
    assert out["gender"] == ("unknown", 0.0)


def test_per_task_state_is_independent():
    """A flip on one task shouldn't reset hysteresis on another."""
    voter = AttributeVoter(person_threshold=0.7)
    voter.resolve_person(1, {"gender": ("male", 0.9),    "lower": ("trousers", 0.9)})
    voter.resolve_person(1, {"gender": ("female", 0.95), "lower": ("trousers", 0.95)})
    out = voter.resolve_person(1, {"gender": ("female", 0.95), "lower": ("trousers", 0.95)})
    # Gender flips after two consecutive "female" agreements above threshold.
    assert out["gender"][0] == "female"
    # Lower stays trousers — never disagreed.
    assert out["lower"][0] == "trousers"


def test_person_snapshot_returns_current_state():
    voter = AttributeVoter(person_threshold=0.5)
    voter.resolve_person(1, {"gender": ("female", 0.8), "hat": ("hat", 0.6)})
    snap = voter.person_snapshot(1)
    assert snap["gender"][0] == "female"
    assert snap["hat"][0] == "hat"


def test_person_task_stable_support_counts_reinforcing_tracklets():
    voter = AttributeVoter(person_threshold=0.7)
    voter.resolve_person(1, {"gender": ("male", 0.76)})
    voter.resolve_person(1, {"gender": ("male", 0.77)})
    voter.resolve_person(1, {"gender": ("male", 0.75)})
    assert voter.person_task_stable_support(1, "gender") == 3


def test_vote_frame_skips_invalid_payloads():
    voter = AttributeVoter()
    voter.vote_frame(1, {"gender": {"label": "male", "confidence": 0.9}})
    voter.vote_frame(1, {"gender": {"confidence": 0.9}})   # missing label
    voter.vote_frame(1, {"gender": "not_a_dict"})           # wrong shape
    out = voter.resolve_tracklet(1)
    assert out["gender"][0] == "male"

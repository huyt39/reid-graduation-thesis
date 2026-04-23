from src.attributes.gender_voter import GenderVoter


def test_tracklet_majority_vote():
    voter = GenderVoter()
    voter.vote_frame(1, "male", 0.9)
    voter.vote_frame(1, "male", 0.8)
    voter.vote_frame(1, "female", 0.7)
    gender, conf = voter.resolve_tracklet(1)
    assert gender == "male"
    assert conf > 0.0


def test_tracklet_empty_returns_unknown():
    voter = GenderVoter()
    gender, conf = voter.resolve_tracklet(99)
    assert gender == "unknown"
    assert conf == 0.0


def test_person_first_assignment():
    voter = GenderVoter(person_threshold=0.6)
    g, c = voter.resolve_person(1, "male", 0.8)
    assert g == "male"
    assert c == 0.8


def test_person_hysteresis_blocks_flip():
    voter = GenderVoter(person_threshold=0.7)
    # Initial
    voter.resolve_person(1, "male", 0.9)
    # One contradicting tracklet — should NOT flip
    g, _ = voter.resolve_person(1, "female", 0.95)
    assert g == "male"


def test_person_hysteresis_flips_after_two_consecutive():
    voter = GenderVoter(person_threshold=0.7)
    voter.resolve_person(1, "male", 0.9)
    # Two consecutive "female" with high confidence
    voter.resolve_person(1, "female", 0.85)
    g, _ = voter.resolve_person(1, "female", 0.90)
    assert g == "female"


def test_person_same_gender_reinforces():
    voter = GenderVoter(person_threshold=0.7)
    voter.resolve_person(1, "male", 0.8)
    g, c = voter.resolve_person(1, "male", 1.0)
    assert g == "male"
    # Confidence should update (weighted average)
    assert c > 0.8


def test_person_low_confidence_blocks_first_assignment():
    voter = GenderVoter(person_threshold=0.7)
    g, c = voter.resolve_person(1, "male", 0.3)
    assert g == "unknown"

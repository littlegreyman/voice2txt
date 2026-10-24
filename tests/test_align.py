from voice2txt.core.align import assign_speakers


def test_assign_majority_overlap():
    segs = [
        {"start": 0.0, "end": 2.0, "text": "a"},
        {"start": 2.0, "end": 4.0, "text": "b"},
    ]
    turns = [
        {"start": 0.0, "end": 1.5, "spk": "SPEAKER_00"},
        {"start": 1.5, "end": 4.0, "spk": "SPEAKER_01"},
    ]
    assign_speakers(segs, turns)
    assert segs[0]["speaker"] == "SPEAKER_00"
    assert segs[1]["speaker"] == "SPEAKER_01"


def test_assign_no_overlap():
    segs = [{"start": 10.0, "end": 12.0, "text": "x"}]
    turns = [{"start": 0.0, "end": 1.0, "spk": "SPEAKER_00"}]
    assign_speakers(segs, turns)
    assert segs[0]["speaker"] is None

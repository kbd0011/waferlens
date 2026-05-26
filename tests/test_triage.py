from waferlens.triage.rules import all_patterns, triage


def test_triage_single_pattern():
    t = triage(["Center"])
    assert t["entries"]
    assert t["entries"][0]["pattern"] == "Center"
    assert t["entries"][0]["candidate_causes"]
    assert t["entries"][0]["recommended_checks"]


def test_triage_none():
    t = triage([])
    assert t["entries"] == []
    assert "none" in t["summary"].lower()


def test_triage_compound_hint():
    t = triage(["Center", "Edge-Ring"])
    assert len(t["entries"]) == 2
    assert t["compound_hints"]   # the Center+Edge-Ring hint should fire


def test_all_patterns_count():
    assert len(all_patterns()) == 8

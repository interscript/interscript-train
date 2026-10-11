import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from oracle_min import oracle_score  # noqa: E402


def test_two_arm_disjoint_strengths_oracle_zero():
    refs = ["اَلْبَيْتُ/اَلْبَيْتَ كَبِيرٌ"]
    a = ["اَلْبَيْتُ خطأ"]   # arm A right on word1
    b = ["خطأ كَبِيرٌ"]      # arm B right on word2
    r = oracle_score(a, b, refs=refs)
    assert r["WER_1"] == 50.0 and r["WER_2"] == 50.0
    assert r["WER_oracle"] == 0.0 and r["DER_oracle"] == 0.0


def test_n_arm_abc_disjoint():
    refs = ["اَلْبَيْتُ/اَلْبَيْتَ كَبِيرٌ"]
    a = ["اَلْبَيْتُ كَبِيرٌ"]
    b = ["اَلْبَيْتِ كَبِيرٌ"]
    c = ["خطأ كَبِيرٌ"]
    r = oracle_score(a, b, c, refs=refs)
    assert r["WER_1"] == 0.0 and r["WER_2"] == 50.0 and r["WER_3"] == 50.0
    assert r["WER_oracle"] == 0.0


def test_length_mismatch_penalizes():
    refs = ["اَلْبَيْتُ/اَلْبَيْتَ كَبِيرٌ"]
    r = oracle_score(["اَلْبَيْتُ"], ["x"], refs=refs)
    assert r["WER_oracle"] == 50.0

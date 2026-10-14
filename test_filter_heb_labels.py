"""WO42 unit tests — label-filter pure logic."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import filter_heb_labels as F  # noqa: E402


def test_phone_lev():
    assert F.phone_lev(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert F.phone_lev([], ["a"]) == 1
    assert F.phone_lev(["a"], ["a"]) == 0


def test_accept_threshold():
    # 1 error in 10 phones = 0.10 < 0.15 -> keep
    assert F.consistent(["a"] * 9 + ["b"], ["a"] * 10, 0.15) is True
    # 3 errors in 10 = 0.30 -> drop
    assert F.consistent(["a", "b", "c"] + ["a"] * 7, ["a"] * 10, 0.15) is False
    # empty decode vs nonempty label -> drop
    assert F.consistent([], ["a", "b"], 0.15) is False


def test_ratio_band():
    rows = [{"ipa": "a b c", "text": "abc"}] * 298 + [
        {"ipa": "z", "text": "verylongtext" * 5} for _ in range(2)
    ]
    lo, hi = F.ratio_band(rows, 0.01, 0.99)
    assert lo <= hi
    # the outliers fall outside the 1-99 band
    out = F.ratio_ok("z", "verylongtext" * 5, lo, hi)
    assert out is False
    assert F.ratio_ok("a b c", "abc", lo, hi) is True


def test_stretch_factors():
    assert F.stretch_factors() == (0.9, 1.1)

"""Tests for the Hebrew nikud plane decomposition (byte-exact inverse)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nikud_planes import NIKUD_MARKS, canon_combo, plane_classes, render, split_planes


class TestNikudPlanes(unittest.TestCase):
    def test_split_render_round_trip(self):
        text = "שָׁלוֹם עֲלֵיכֶם"
        skel, classes = split_planes(text)
        self.assertEqual(skel, "שלום עליכם")
        self.assertEqual(len(classes), len(skel))
        self.assertEqual(render(skel, classes), text)

    def test_marks_close_into_previous_letter(self):
        skel, classes = split_planes("בְּרֵאשִׁית")
        self.assertEqual(skel, "בראשית")
        # dagesh+sheva on bet, tsere on alef... each cluster one class
        self.assertTrue(classes[0])
        self.assertEqual(render(skel, classes), "בְּרֵאשִׁית")

    def test_bare_text_is_all_empty_classes(self):
        skel, classes = split_planes("שלום עליכם")
        self.assertEqual(classes, [""] * len(skel))

    def test_canon_preserves_written_order(self):
        # cluster order is preserved as written (corpus convention)
        self.assertEqual(canon_combo("ְּ"), "ְּ")
        self.assertEqual(canon_combo("ָּ"), "ָּ")

    def test_maqqaf_stays_in_skeleton(self):
        skel, classes = split_planes("וְכָל־הָעָם")
        self.assertIn("־", skel)
        self.assertEqual(render(skel, classes), "וְכָל־הָעָם")

    def test_plane_classes_is_stable_inventory(self):
        self.assertIn("ָ", plane_classes(["שָׁלוֹם", "דָּבָר"]))

    def test_real_corpus_round_trip(self):
        path = Path(__file__).resolve().parent / "data/hebrew-distilled/train.txt"
        if not path.exists():
            self.skipTest("corpus not present")
        n = 0
        for line in path.read_text(encoding="utf-8").splitlines()[:500]:
            line = line.strip()
            if not line:
                continue
            skel, classes = split_planes(line)
            self.assertEqual(render(skel, classes), line, line[:40])
            n += 1
        self.assertGreater(n, 100)


if __name__ == "__main__":
    unittest.main()

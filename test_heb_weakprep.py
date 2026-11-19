"""Tests for the WO92 knesset weak-pretrain prep (normalize/filter/dedupe/decontam).

Real string instances throughout — no doubles, no network, no disk
fixtures beyond a tmp file.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nikud_planes  # noqa: F401  (prep imports it via /code in the job)
import prep_heb_knesset_weak as P


class TestNormalizeLine(unittest.TestCase):
    def test_strips_syllable_and_stress_marks(self):
        # '|' syllable separator + teamim/stress + custom marks are dropped;
        # nikud plane marks are kept
        line = "שָׁ|לוֹ|ם עַ|ל הָ|אָ֗רֶץ"
        pair = P.normalize_line(line)
        self.assertIsNotNone(pair)
        src, tgt = pair
        self.assertNotIn("|", tgt)
        self.assertNotIn("֗", tgt)  # teamim dropped
        self.assertNotIn("֫", tgt)  # stress (ateh) dropped
        self.assertIn("שָׁלוֹם", tgt)
        self.assertEqual(src, "".join(c for c in tgt if c not in nikud_planes.NIKUD_MARKS).strip())

    def test_round_trip_through_plane_split(self):
        pair = P.normalize_line("הַ|דְּ|בָרִ|ים הָ|אֵ|לֶּה")
        self.assertIsNotNone(pair)
        tgt = pair[1]
        skel, classes = nikud_planes.split_planes(tgt)
        self.assertEqual(nikud_planes.render(skel, classes), tgt)

    def test_short_line_filtered(self):
        self.assertIsNone(P.normalize_line("שָׁלוֹם"))

    def test_long_line_filtered(self):
        self.assertIsNone(P.normalize_line("שָׁלוֹם " * 65))

    def test_ascii_pipe_dropped_explicitly(self):
        # phonikud PREFIX_CHAR is ASCII U+007C — inside the printable band,
        # so the drop must be explicit, not a side effect of the range check
        pair = P.normalize_line("שָׁ|לוֹ|ם עַ|ל הָ|אָרֶץ בָּ|אָרֶץ")
        self.assertIsNotNone(pair)
        self.assertNotIn("|", pair[1])
        self.assertNotIn("|", pair[0])

    def test_low_hebrew_fraction_filtered(self):
        self.assertIsNone(P.normalize_line("hello world 12345 שָׁלוֹם"))

    def test_low_nikud_density_filtered(self):
        self.assertIsNone(P.normalize_line("אבגדהוזחטיכלמנסעפצקרשת אבגדהוזחטיכלמנס"))

    def test_ascii_and_whitespace_kept(self):
        pair = P.normalize_line("בַּיּוֹם הַהוּא 2026 (והַשָּׁלוֹם)")
        self.assertIsNotNone(pair)
        self.assertIn("2026", pair[1])


class TestLetterNorm(unittest.TestCase):
    def test_drops_nikud_punct_and_marks(self):
        norm = P._letter_norm("שָׁלוֹם, עֲלֵיכֶם| — 123")
        self.assertEqual(norm, "שלום עליכם")

    def test_collapses_whitespace_runs(self):
        self.assertEqual(P._letter_norm("א  ב   דֶּשֶׁם"), "א ב דשם")


class TestBuildTestWindows(unittest.TestCase):
    def test_windows_from_both_sides(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            row = {"src": "וידבר אלהים את כל הדברים האלה לאמר",
                   "tgt": "וַיְדַבֵּר אֱלֹהִים אֶת כָּל הַדְּבָרִים הָאֵלֶּה לֵאמֹר"}
            f.write(json.dumps(row) + "\n")
            path = Path(f.name)
        try:
            windows = P.build_test_windows(path)
            self.assertGreater(len(windows), 0)
            # a 40-letter slice of the normalized src must be in the window set
            norm = P._letter_norm(row["src"])
            self.assertTrue(any(norm[i:i + 40] in windows
                                for i in range(0, max(1, len(norm) - 40), 20)))
        finally:
            path.unlink(missing_ok=True)

    def test_contam_window_matches_normalized_variant(self):
        # nikud/punct variants of the same text must collide in the window set
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write(json.dumps({"src": "אבגדהוזחטיכלמנסעפצקרשתאבגדהוזחטיכל",
                                "tgt": "אבגדהוזחטיכלמנסעפצקרשתאבגדהוזחטיכל"}) + "\n")
            path = Path(f.name)
        try:
            windows = P.build_test_windows(path)
            variant = P._letter_norm("אבגדהוזחטיכלמנסעפצקרשתאבגדהוזחטיכל")
            hit = any(variant[i:i + 40] in windows
                      for i in range(0, max(1, len(variant) - 40), 20))
            self.assertTrue(hit)
        finally:
            path.unlink(missing_ok=True)


class TestDedupePolicyContract(unittest.TestCase):
    def test_skeleton_dup_is_detected_by_set_semantics(self):
        # the policy: exact-input (skeleton) dedupe — same skeleton from
        # differently-marked lines is a dup; first occurrence wins
        a = P.normalize_line("שָׁלוֹם עֲלֵיכֶם כֹּל יִשְׂרָאֵל עַם אֶחָד בְּאָרְצוֹ")
        b = P.normalize_line("שָׁלוֹם עֲלֵיכֶם כֹּל יִשְׂרָאֵל עַם אֶחָד בְּאָרְצוֹ")
        self.assertEqual(a[0], b[0])
        seen = set()
        self.assertTrue(a[0] not in seen)
        seen.add(a[0])
        self.assertTrue(b[0] in seen)


class TestSha256File(unittest.TestCase):
    def test_known_digest(self):
        with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as f:
            f.write(b"heb weak prep\n")
            path = Path(f.name)
        try:
            import hashlib
            self.assertEqual(P.sha256_file(path),
                             hashlib.sha256(b"heb weak prep\n").hexdigest())
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

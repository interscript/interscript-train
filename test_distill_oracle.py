"""WO39 stage-1 corpus-fidelity tests — distill_oracle_hf.build_rows
must reproduce run-047's qcri-only stripped corpus flow exactly.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import distill_oracle_hf as D  # noqa: E402

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")


@pytest.fixture()
def tiny_corpus(tmp_path, monkeypatch):
    arts = []
    for a in range(4):
        paras = []
        for p in range(3):
            word = "بٍچًثَ" + "عibal".replace("ibal", "ٌلَم") + str(a) + str(p)
            paras.append(" ".join(f"{word}{k}/بٍچًثَ{k}" for k in range(8)))
        arts.append({"text": "\n".join(paras)})
    q = tmp_path / "qcri-wiki.jsonl"
    q.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in arts),
                 encoding="utf-8")
    monkeypatch.setattr(D, "DATA", tmp_path)
    monkeypatch.setattr(D, "PSEUDO_CAP", 40)
    return q


def test_strip_alt_applied(tiny_corpus):
    rows = D.build_rows()
    assert rows
    for _src, tgt in rows:
        assert "/" not in tgt


def test_cap_and_determinism(tiny_corpus, monkeypatch):
    r1 = D.build_rows()
    r2 = D.build_rows()
    assert r1 == r2
    assert len(r1) <= D.PSEUDO_CAP
    monkeypatch.setattr(D, "PSEUDO_CAP", 5)
    assert len(D.build_rows()) == 5


def test_rows_match_trainer_flow(tiny_corpus):
    """Independent re-derivation of the trainer's flow (seed 7
    article shuffle -> window -> silver -> strip -> seed 42 shuffle).
    """
    articles = tiny_corpus.read_text(encoding="utf-8").splitlines()
    rng2 = random.Random(7)
    rng2.shuffle(articles)
    pseudo: list[tuple[str, str]] = []

    def add_silver(tgt):
        tgt = tgt.strip()
        src = DIACRITICS_RE.sub("", tgt).strip()
        if len(src) >= 5 and len(src.encode()) <= D.WINDOW - 50:
            pseudo.append((src, tgt))

    for aline in articles:
        if len(pseudo) >= D.PSEUDO_CAP:
            break
        art = json.loads(aline).get("text", "")
        for para in art.split("\n"):
            para = para.strip()
            if not para:
                continue
            if len(para.encode()) <= D.WINDOW - 50:
                add_silver(para)
            else:
                for w in D.split_windows(para, D.WINDOW - 50):
                    add_silver(w)
            if len(pseudo) >= D.PSEUDO_CAP:
                break
    stripped = []
    for _src, tgt in pseudo:
        tgt2 = " ".join(t.split("/", 1)[0] for t in tgt.split())
        src2 = DIACRITICS_RE.sub("", tgt2).strip()
        if len(src2) >= 5:
            stripped.append((src2, tgt2))
    rng = random.Random(42)
    rng.shuffle(stripped)
    assert D.build_rows() == stripped[:D.PSEUDO_CAP]


def test_split_windows_budget():
    long_text = " ".join("abc" * 1 for _ in range(400))
    wins = D.split_windows(long_text)
    assert len(wins) > 1
    assert all(len(w.encode()) <= D.WINDOW for w in wins)
    assert " ".join(wins) == long_text


def test_lev():
    assert D.lev("kitten", "sitting") == 3
    assert D.lev("", "ab") == 2
    assert D.lev("ab", "ab") == 0


def test_completed_lines_missing(tmp_path):
    assert D.completed_lines(tmp_path / "nope.jsonl") == 0


def test_completed_lines_clean(tmp_path):
    p = tmp_path / "preds.jsonl"
    p.write_text('{"pred": "a"}\n{"pred": "b"}\n{"pred": "c"}\n')
    assert D.completed_lines(p) == 3


def test_completed_lines_truncates_partial(tmp_path):
    p = tmp_path / "preds.jsonl"
    p.write_text('{"pred": "a"}\n{"pred": "b"}\n{"pred": "c"}\n{"pred": "d')
    assert D.completed_lines(p) == 3
    assert p.read_text().endswith('{"pred": "c"}\n')


def test_completed_lines_empty(tmp_path):
    p = tmp_path / "preds.jsonl"
    p.write_text("")
    assert D.completed_lines(p) == 0

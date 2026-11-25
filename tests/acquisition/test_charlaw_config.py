"""Unit tests: the character-class law, normalization, decontam windows,
and the config schema (including the draft Jawi/Pashto recipes)."""

from __future__ import annotations

import pytest

from acquisition.charlaw import ScriptLaw, windows_of
from acquisition.configs.fas import FAS_GATE_LAW, FAS_PREP_LAW, gate_sources
from acquisition.decontam import build_eval_windows, hits_windows
from fasfixtures import HAMZA_ABOVE, V_EVAL_SRC, ZWNJ


# ---- the Persian dual law ----

def test_fas_strip_law():
    # vocalic marks U+064B-0652 + U+0670 are stripped...
    text = ("بِسْمِ" )  # kasra, sukun
    stripped = FAS_PREP_LAW.strip(text)
    assert stripped == "بسم"
    tanwin = "رسولًا"
    assert FAS_PREP_LAW.strip(tanwin) == "رسولا"  # fathatan dropped
    dagger = "رحیمٰ"
    assert FAS_PREP_LAW.strip(dagger) == "رحیم"  # dagger alif dropped
    # ...while hamza/maddah carriers STAY (they change the letter itself)
    carrier = "تو" + HAMZA_ABOVE + "انستم"
    assert FAS_PREP_LAW.strip(carrier) == carrier
    maddah = "آس" + "م" + "ان"
    assert FAS_PREP_LAW.strip("اس" + "ٓ" + "مان") .startswith("اسٓ")


def test_fas_dual_density_law():
    # the prep law's density set = the pinned strip set (064B-0652+0670);
    # the gate law is WIDER (064B-065F+0670) — the preserved pre-existing
    # dual law. The carrier marks 0653-0655 sit in the gap: excluded by
    # the prep set, included by the gate set.
    carrier = chr(0x0653)  # maddah above
    assert not FAS_PREP_LAW.is_density_mark(carrier)
    assert FAS_GATE_LAW.is_density_mark(carrier)
    hamza_above = chr(0x0654)
    assert not FAS_PREP_LAW.is_density_mark(hamza_above)
    assert FAS_GATE_LAW.is_density_mark(hamza_above)
    low_mark = chr(0x0656)
    assert not FAS_PREP_LAW.is_density_mark(low_mark)
    assert FAS_GATE_LAW.is_density_mark(low_mark)
    shadda = chr(0x0651)
    assert FAS_PREP_LAW.is_density_mark(shadda)
    assert FAS_GATE_LAW.is_density_mark(shadda)


def test_letter_norm_and_fraction():
    law = FAS_PREP_LAW
    norm = law.letter_norm("  کِتاب\t\n دانش  ")
    assert norm == "کتاب دانش"
    # non-letters dropped, whitespace collapsed
    frac = law.letter_fraction("سلام 123 ab")
    # non-space chars: سلام (4) + 123 (3) + ab (2) = 9; letters = 4
    assert frac == pytest.approx(4 / 9)


def test_windows_of_edge_cases():
    short = "ط" * 10
    assert windows_of(short, 40, 20) == [short]  # single whole-norm window
    long_ = "ب" * 95
    ws = windows_of(long_, 40, 20)
    assert ws[0] == long_[0:40]
    assert ws[1] == long_[20:60]
    assert ws[-1] == long_[40:80]  # range stops before len-40


def test_decontam_window_hit(tmp_path):
    # an eval row whose norm IS the held-out window catches the candidate
    eval_file = tmp_path / "eval.jsonl"
    import json
    eval_file.write_text(json.dumps({"src": V_EVAL_SRC, "tgt": "x"},
                                    ensure_ascii=False) + "\n", encoding="utf-8")
    windows = build_eval_windows([eval_file], FAS_PREP_LAW)
    assert hits_windows(FAS_PREP_LAW.letter_norm(V_EVAL_SRC), windows)
    assert not hits_windows(FAS_PREP_LAW.letter_norm("یک متن بی‌ربط"), windows)


# ---- config schema ----

def test_config_chain_order_is_counts_order():
    # the chain step order drives Counter insertion order = manifest
    # counts key order — load-bearing, pinned here
    from collections import Counter as C

    from acquisition.configs.fas import ccfa_web, ganjoor
    from acquisition.pipeline import apply_chain

    cfg = ganjoor()
    counters = C()
    seen: set[str] = set()
    # ratio out of band: tgt shorter than src -> the chain stops at the
    # ratio step (raw-row counters are assemble_from_raw's job)
    row = {"src": "طول نامعتبر", "tgt": "طول"}
    apply_chain(row, cfg.assemble, cfg.law, set(), counters, seen)
    assert list(counters.keys()) == ["filtered_ratio"]

    cfg_cc = ccfa_web()
    counters_cc = C()
    row_cc = {"text": "ط" * 60}
    kept = apply_chain(row_cc, cfg_cc.assemble, cfg_cc.law, set(),
                       counters_cc, set())
    # survival carries no counter — assemble_from_raw increments `kept`
    assert kept is True and list(counters_cc.keys()) == []


def test_fas_gate_defs_shapes():
    sources = gate_sources()
    assert [s.subdir for s in sources] == ["ganjoor", "fawikisource", "cc-fa-web"]
    g = sources[0].gate
    assert g.pair_mode and g.text_key == "src" and g.partner_key == "tgt"
    # the pre-registered check keys are literal (thresholds embedded)
    keys = [c.key for c in g.checks]
    assert keys == [
        "ratio_band_all_rows",
        "vocalized_share_min_0.30",
        "harakat_density_median_min_0.05",
        "degenerate_lt_1pct",
        "script_median_min_0.5",
        "persian_specific_all_present",
        "zwnj_present",
    ]
    # the gate law is the WIDER mark set (dual law)
    assert sources[0].law is FAS_GATE_LAW


def test_draft_configs_are_marked_and_wellformed():
    from acquisition.configs import jawi_ms, ps_af
    # both drafts carry the UNVERIFIED status banner
    assert "UNVERIFIED" in jawi_ms.STATUS and "R3" in jawi_ms.STATUS
    assert "UNVERIFIED" in ps_af.STATUS and "R4" in ps_af.STATUS
    # the recipes construct (schema-wellformed)
    jawi = jawi_ms.jawi_quran_mining()
    ms = jawi_ms.ms_wikisource_draft()
    pus = ps_af.pus_web_weak_draft()
    assert jawi.law.name == "jawi-draft"
    assert pus.adapter.crawl_id == "CC-MAIN-PROBE"  # unpinned until the probe
    assert ms.adapter.probed_bytes == -1  # probe required before launch
    # Jawi extension letters are inside the law
    assert jawi.law.is_letter(chr(0x06AD))  # ڭ nga
    assert pus.law.is_letter(chr(0x06CD))  # ۍ
    assert not pus.law.is_letter("a")


def test_placeholder_sources_are_invalid_tlds():
    # draft placeholder endpoints must be .invalid (never launchable)
    from acquisition.configs import jawi_ms, ps_af
    assert "jawi-source.invalid" in jawi_ms.jawi_quran_mining().adapter.base_url
    assert ps_af.pus_web_weak_draft().adapter.crawl_id == "CC-MAIN-PROBE"


def test_scriptlaw_expand_matches_range_semantics():
    law = ScriptLaw(name="t", letter_ranges=((0x0621, 0x0623),))
    assert law.is_letter(chr(0x0621))
    assert law.is_letter(chr(0x0623))
    assert not law.is_letter(chr(0x0624))

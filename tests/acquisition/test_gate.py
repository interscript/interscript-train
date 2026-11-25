"""Gate engine tests: equivalence vs the FROZEN pre-refactor gate script
on (a) a synthetic all-PASS tree and (b) the fixture acquisition tree
(with its pre-registered row floors it FAILS), plus targeted per-check
FAIL paths."""

from __future__ import annotations

import json

import pytest

from acquisition.configs.fas import (
    GATE_CAMPAIGN,
    GATE_LAW_LINE,
    gate_dir,
    gate_sources,
)
from acquisition.gate import check_sums, run_gate, verify_pairs, verify_text_rows
from acquisition.pipeline import sha256_file, write_shards
from fasfixtures import (
    load_reference,
    read_manifest,
    V1_SRC,
    V1_TGT,
    V104_SRC,
    V104_TGT,
    V201_SRC,
    V201_TGT,
    V9_SRC,
    V9_TGT,
    V_UNVOC,
)

WIKI_GATE_TEXT = "پرسش چگونه ژرف‌تر می‌شود؟ باب نخست کتاب گشوده شد."


def _write_source_tree(run_dir, prefix, rows):
    """Shards + SHA256SUMS + manifest + DONE — the artifact contract the
    gate verifies (built with the factory's own shard writer)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    shard_meta = write_shards(run_dir, rows, prefix, 8)
    (run_dir / "manifest.json").write_text(
        json.dumps({"counts": {"kept": len(rows)}}, ensure_ascii=False),
        encoding="utf-8")
    sums = "".join(f"{m['sha256']}  {m['file']}\n" for m in shard_meta)
    sums += f"{sha256_file(run_dir / 'manifest.json')}  manifest.json\n"
    (run_dir / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    (run_dir / "DONE").write_text("ok\n", encoding="utf-8")


def _pass_tree(root):
    acq = root / "run-098-fas-acquisition"
    pairs = [
        {"src": V1_SRC, "tgt": V1_TGT},
        {"src": V_UNVOC, "tgt": V_UNVOC},
        {"src": V104_SRC, "tgt": V104_TGT},
        {"src": V9_SRC, "tgt": V9_TGT},
        {"src": V201_SRC, "tgt": V201_TGT},
    ]
    _write_source_tree(acq / "ganjoor", "ganjoor", pairs)
    _write_source_tree(acq / "fawikisource", "fawikisource",
                       [{"text": WIKI_GATE_TEXT}] * 500)
    _write_source_tree(acq / "cc-fa-web", "cc-fa-web",
                       [{"text": WIKI_GATE_TEXT}] * 10_000)
    return acq


def _run_both_gates(tmp_path, monkeypatch, expect_exit):
    root = tmp_path / "tree"
    acq = _pass_tree(root)

    # reference gate first (its verdict files get overwritten after)
    monkeypatch.setenv("FAS_CKPT", str(root))
    monkeypatch.delenv("FAS_RUN_DIR", raising=False)
    mod = load_reference("gate_fas_acquisition.py")
    mod.BUCKET = root
    mod.ACQ = acq
    if expect_exit:
        with pytest.raises(SystemExit) as exc_ref:
            mod.main()
        assert exc_ref.value.code == 1
    else:
        mod.main()
    ref_verdict = json.loads((acq / "gate-verdict.json").read_text(encoding="utf-8"))
    ref_marker = (acq / ("GATE_PASS" if not expect_exit else "GATE_FAIL")).read_bytes()

    # factory gate on the SAME tree
    if expect_exit:
        with pytest.raises(SystemExit) as exc_fac:
            from acquisition.run import run_gates
            run_gates(gate_dir(), gate_sources(), GATE_CAMPAIGN, GATE_LAW_LINE)
        assert exc_fac.value.code == 1
    else:
        from acquisition.run import run_gates
        run_gates(gate_dir(), gate_sources(), GATE_CAMPAIGN, GATE_LAW_LINE)
    fac_verdict = json.loads((acq / "gate-verdict.json").read_text(encoding="utf-8"))
    fac_marker = (acq / ("GATE_PASS" if not expect_exit else "GATE_FAIL")).read_bytes()

    # equivalence: identical modulo the utc timestamp
    ref_verdict.pop("utc")
    fac_verdict.pop("utc")
    assert ref_verdict == fac_verdict
    assert ref_marker == fac_marker
    return ref_verdict


def test_gate_all_pass_synthetic_tree(tmp_path, monkeypatch):
    verdict = _run_both_gates(tmp_path, monkeypatch, expect_exit=False)
    assert verdict["verdict"] == "PASS"
    by_name = {s["stats"]["source"]: s for s in verdict["sources"]}
    assert all(s["verdict"] == "PASS" for s in verdict["sources"])
    # the ganjoor checks all hold on the engineered tree
    assert all(by_name["ganjoor"]["checks"].values())
    assert by_name["fawikisource"]["checks"]["rows_min_500"]
    assert by_name["cc-fa-web"]["checks"]["rows_min_10000"]


def test_gate_fixture_tree_fails_row_floors(tmp_path, monkeypatch, fixture_server):
    """The fixture corpora fail the gate: the pre-registered row
    floors (2 wiki rows < 500, 2 cc rows < 10k) and the text-source
    Persian-specific presence check — both engines must agree on
    FAIL and exit 1."""
    server = fixture_server
    root = tmp_path / "tree"
    monkeypatch.setenv("FAS_CKPT", str(root))
    monkeypatch.delenv("FAS_RUN_DIR", raising=False)
    monkeypatch.setenv("FAS_EVAL_POETRY", str(server["eval_poetry"]))
    monkeypatch.setenv("FAS_EVAL_DOCS", str(server["eval_docs"]))
    monkeypatch.setenv("FAS_CRAWL_SNAPSHOT", "2026-11-25")
    monkeypatch.setenv("FAS_CC_NSEG", "2")

    from dataclasses import replace
    from acquisition.configs.fas import ganjoor, fawikisource, ccfa_web
    from acquisition.run import run_prep

    run_prep(replace(ganjoor(), adapter=replace(
        ganjoor().adapter, base_url=f"{server['base']}/api/ganjoor")))
    cfg = fawikisource()
    run_prep(replace(cfg, adapter=replace(
        cfg.adapter, dump_url=f"{server['base']}/fawikisource/latest/"
        "fawikisource-latest-pages-articles.xml.bz2")))
    run_prep(replace(ccfa_web(), adapter=replace(
        ccfa_web().adapter, base_url=server["base"])))

    # the ganjoor prep packages inside raw/ (its documented layout); the
    # gate contract expects the packaging at the run-dir level — stage it
    gan = root / "run-098-fas-acquisition" / "ganjoor"
    for name in ("SHA256SUMS.txt", "manifest.json", "DONE"):
        (gan / "raw" / name).replace(gan / name)
    for shard in sorted((gan / "raw").glob("ganjoor-*.jsonl")):
        shard.replace(gan / shard.name)

    # reference gate
    mod = load_reference("gate_fas_acquisition.py")
    mod.BUCKET = root
    mod.ACQ = root / "run-098-fas-acquisition"
    with pytest.raises(SystemExit) as exc_ref:
        mod.main()
    assert exc_ref.value.code == 1
    ref = json.loads((mod.ACQ / "gate-verdict.json").read_text(encoding="utf-8"))
    ref.pop("utc")
    ref_fail_marker = (mod.ACQ / "GATE_FAIL").read_bytes()

    with pytest.raises(SystemExit) as exc_fac:
        from acquisition.run import run_gates
        run_gates(gate_dir(), gate_sources(), GATE_CAMPAIGN, GATE_LAW_LINE)
    assert exc_fac.value.code == 1
    fac = json.loads((gate_dir() / "gate-verdict.json").read_text(encoding="utf-8"))
    fac.pop("utc")
    assert ref == fac
    assert ref_fail_marker == (gate_dir() / "GATE_FAIL").read_bytes()

    by_name = {s["stats"]["source"]: s for s in fac["sources"]}
    assert by_name["ganjoor"]["verdict"] == "PASS"
    assert by_name["fawikisource"]["verdict"] == "FAIL"
    assert by_name["cc-fa-web"]["verdict"] == "FAIL"
    assert not by_name["fawikisource"]["checks"]["rows_min_500"]
    assert not by_name["cc-fa-web"]["checks"]["rows_min_10000"]


# ---- targeted per-check FAIL paths ----

def _pairs_dir(tmp_path, rows, corrupt=None):
    d = tmp_path / "ganjoor"
    d.mkdir()
    meta = write_shards(d, rows, "ganjoor", 2)
    sums = "".join(f"{m['sha256']}  {m['file']}\n" for m in meta)
    (d / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    if corrupt is not None:
        shard = sorted(d.glob("ganjoor-*.jsonl"))[0]
        shard.write_text(corrupt, encoding="utf-8")
    return d


def test_check_sums_detects_corruption(tmp_path):
    d = _pairs_dir(tmp_path, [{"src": "ط", "tgt": "ط"}], corrupt="tampered\n")
    ok, problems = check_sums(d)
    assert not ok
    assert any("sha mismatch" in p for p in problems)


def test_check_sums_missing_file(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    ok, problems = check_sums(d)
    assert not ok
    assert problems == ["SHA256SUMS.txt missing"]


def test_ratio_band_violation(tmp_path, monkeypatch):
    from acquisition.configs.fas import FAS_GATE_LAW, gate_sources
    rows = [{"src": V1_SRC, "tgt": V1_TGT},
            {"src": "ط", "tgt": "ط" + "َ" * 10}]  # ratio > 2
    d = _pairs_dir(tmp_path, rows)
    res = verify_pairs(d, gate_sources()[0].gate, FAS_GATE_LAW)
    assert res["checks"]["ratio_band_all_rows"] is False
    assert res["counts"]["ratio_max"] > 2.0


def test_degenerate_and_script_checks(tmp_path):
    from acquisition.configs.fas import FAS_GATE_LAW, gate_sources
    rows = [{"src": "", "tgt": "x"}, {"src": "hello world", "tgt": "hello world"}]
    d = _pairs_dir(tmp_path, rows)
    res = verify_pairs(d, gate_sources()[0].gate, FAS_GATE_LAW)
    assert res["checks"]["degenerate_lt_1pct"] is False
    assert res["checks"]["script_median_min_0.5"] is False
    assert res["checks"]["persian_specific_all_present"] is False
    assert res["checks"]["zwnj_present"] is False


def test_text_mode_checks(tmp_path):
    from acquisition.configs.fas import FAS_GATE_LAW, gate_sources
    d = tmp_path / "fawikisource"
    d.mkdir()
    rows = [{"text": ""}, {"text": "english only text here"}]
    write_shards(d, rows, "fawikisource", 2)
    res = verify_text_rows(d, gate_sources()[1].gate, FAS_GATE_LAW)
    assert res["checks"]["degenerate_lt_1pct"] is False
    assert res["checks"]["script_median_min"] is False
    assert res["checks"]["persian_specific_all_present"] is False
    assert res["checks"]["rows_min_500"] is False


def test_run_gate_missing_done(tmp_path, monkeypatch):
    from acquisition.configs.fas import FAS_GATE_LAW
    from acquisition.gate import GateSource
    acq = tmp_path / "acq"
    (acq / "ganjoor").mkdir(parents=True)
    src = GateSource(subdir="ganjoor", gate=gate_sources()[0].gate, law=FAS_GATE_LAW)
    log_lines = []
    overall = run_gate(acq, [src], "test campaign", "test law",
                       write=log_lines.append)
    assert overall is False
    saved = json.loads((acq / "gate-verdict.json").read_text(encoding="utf-8"))
    assert saved["sources"][0]["problems"] == ["DONE missing (acquisition incomplete)"]

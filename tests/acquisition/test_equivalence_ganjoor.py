"""Ganjoor adapter equivalence: factory vs the FROZEN pre-refactor script.

Two levels:
1. assemble-boundary BYTE equivalence — identical raw poet files (the
   canonical stage-A output) fed to both engines must produce
   byte-identical shards/SHA256SUMS/DONE and manifests identical beyond
   built_utc. This is the exact-equivalence proof.
2. end-to-end equivalence over the API fixture — the pre-refactor stage
   A is thread-completion-ordered (nondeterministic BETWEEN RUNS, a
   preserved pre-existing property), so end-to-end equality is asserted
   at counts/invariants/per_poet/row-set level, plus the exact counter
   values of the fully-mapped fixture corpus.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from acquisition.configs.fas import ganjoor
from acquisition.run import run_prep
from fasfixtures import (
    HAMZA_ABOVE,
    V_EVAL_SRC,
    ZWNJ,
    compare_trees,
    ganjoor_canonical_raw,
    load_reference,
    read_manifest,
    shard_rows,
)

RUN_REL = "run-098-fas-acquisition/ganjoor"

EXPECTED_COUNTS = {
    # poem 103 (غزل چهار) is deliberately unrouted -> 404s with zero
    # verses, so only saadi's 3 + hafez's 8 fetched verses are counted
    "verses_raw": 11,
    "filtered_len": 2,
    "filtered_script": 1,
    "filtered_ratio": 1,
    "dup_dropped": 1,
    "decontam_dropped": 1,
    "kept": 5,
    "poet_files": 2,
    "kept_unique": 5,
    "degenerate": 0,
}


def _set_env(monkeypatch, root, server):
    monkeypatch.setenv("FAS_CKPT", str(root))
    monkeypatch.delenv("FAS_RUN_DIR", raising=False)
    monkeypatch.setenv("FAS_EVAL_POETRY", str(server["eval_poetry"]))
    monkeypatch.setenv("FAS_EVAL_DOCS", str(server["eval_docs"]))
    monkeypatch.setenv("FAS_CRAWL_SNAPSHOT", "2026-11-25")
    for var in ("FAS_WORKERS", "FAS_SHARDS"):
        monkeypatch.delenv(var, raising=False)


def _seed_raw(root, raws):
    raw_dir = root / RUN_REL / "raw"
    raw_dir.mkdir(parents=True)
    for name, content in raws.items():
        (raw_dir / name).write_text(content, encoding="utf-8")
        slug = name[: -len(".jsonl")]
        url = "/hafez" if slug == "hafez" else "/saadi"
        (raw_dir / f"{slug}.done").write_text(
            json.dumps({"poet": url, "poems": 5, "verses": 5,
                        "cats": 2, "errors": 0}) + "\n",
            encoding="utf-8")
    return raw_dir


def _fixture_cfg(server):
    cfg = ganjoor()
    base = f"{server['base']}/api/ganjoor"
    source = dict(cfg.source_meta)
    source["api"] = base  # the manifest records the base (as the frozen
    # script does after the test-harness URL substitution)
    return replace(cfg, adapter=replace(cfg.adapter, base_url=base),
                   source_meta=source)


def test_ganjoor_assemble_boundary_byte_equivalence(
        tmp_path, monkeypatch, fixture_server):
    server = fixture_server
    raws = ganjoor_canonical_raw()
    outs = []
    for tag in ("ref", "fac"):
        root = tmp_path / tag
        _seed_raw(root, raws)
        _set_env(monkeypatch, root, server)
        if tag == "ref":
            mod = load_reference(
                "prep_fas_ganjoor.py",
                {"https://api.ganjoor.net": server["base"]})
            mod.main()
        else:
            run_prep(_fixture_cfg(server))
        outs.append(root / RUN_REL)

    # the exact-equivalence proof: byte-identical everything
    assert compare_trees(outs[0], outs[1]) == []


def test_ganjoor_end_to_end_set_equivalence(
        tmp_path, monkeypatch, fixture_server):
    server = fixture_server
    outs = []
    manifests = []
    for tag in ("ref", "fac"):
        root = tmp_path / tag
        _set_env(monkeypatch, root, server)
        if tag == "ref":
            mod = load_reference(
                "prep_fas_ganjoor.py",
                {"https://api.ganjoor.net": server["base"]})
            mod.main()
        else:
            run_prep(_fixture_cfg(server))
        outs.append(root / RUN_REL / "raw")
        manifests.append(read_manifest(root / RUN_REL / "raw"))

    ma, mb = manifests
    # the exact counter values of the fully-mapped fixture corpus
    assert ma["counts"] == EXPECTED_COUNTS
    assert mb["counts"] == EXPECTED_COUNTS
    assert ma["invariants"] == mb["invariants"]
    assert ma["per_poet"] == mb["per_poet"]
    assert ma["invariants"]["gates"] == {
        "vocalized_share_min": 0.30,
        "ratio_band": [1.0, 2.0],
        "degenerate_max": 0.01,
        "harakat_density_median_min": 0.05,
    }
    # vocalized share: 4 of 5 kept rows are vocalized
    assert ma["invariants"]["vocalized_share"] == pytest.approx(0.8)

    # stage A is completion-ordered (nondeterministic) — row-SET equality
    ra = sorted((r["src"], r["tgt"]) for r in shard_rows(outs[0], "ganjoor"))
    rb = sorted((r["src"], r["tgt"]) for r in shard_rows(outs[1], "ganjoor"))
    assert ra == rb

    # the pinned strip law on the surviving rows: vocalic marks
    # U+064B-0652 + U+0670 stripped, hamza carrier U+0654 and ZWNJ kept
    assert any(HAMZA_ABOVE in s and ZWNJ in s for s, _ in ra)
    assert all(
        not any(0x064B <= ord(c) <= 0x0652 or ord(c) == 0x0670 for c in s)
        for s, _ in ra)
    # the held-out row never enters the corpus
    assert all(V_EVAL_SRC not in s for s, _ in ra)

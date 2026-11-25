"""cc-fa-web adapter equivalence: factory vs the FROZEN pre-refactor
script. The pre-refactor crawl is deterministic (sequential segments,
record order) — END-TO-END BYTE equivalence (every file byte-identical
except manifest built_utc)."""

from __future__ import annotations

from dataclasses import replace

from acquisition.configs.fas import ccfa_web
from acquisition.run import run_prep
from fasfixtures import (
    CC_DOC_KEEP1,
    CC_DOC_KEEP2,
    V_EVAL_SRC,
    compare_trees,
    load_reference,
    read_manifest,
    shard_rows,
)

RUN_REL = "run-098-fas-acquisition/cc-fa-web"
KEEP1_CLEAN = "\n".join(
    ln for ln in CC_DOC_KEEP1
    if not (ln.strip() and sum(1 for c in ln if 0x0621 <= ord(c) <= 0x06D3) < 3
            and len(ln.strip()) < 60))
KEEP2 = "\n".join(CC_DOC_KEEP2)
EVAL_TEXT = V_EVAL_SRC + " و متن ادامه می‌یابد تا پنجره بسته شود و شب فرا رسد."

EXPECTED_COUNTS = {
    "docs_raw": 4,
    "chars_raw": 2 * len(KEEP1_CLEAN) + len(EVAL_TEXT) + len(KEEP2),
    "dup_dropped": 1,
    "decontam_dropped": 1,
    "kept": 2,
    "kept_unique": 2,
}


def _set_env(monkeypatch, root, server):
    monkeypatch.setenv("FAS_CKPT", str(root))
    monkeypatch.delenv("FAS_RUN_DIR", raising=False)
    monkeypatch.setenv("FAS_EVAL_POETRY", str(server["eval_poetry"]))
    monkeypatch.setenv("FAS_EVAL_DOCS", str(server["eval_docs"]))
    monkeypatch.setenv("FAS_CC_NSEG", "2")
    monkeypatch.delenv("FAS_CC_CRAWL", raising=False)
    monkeypatch.delenv("FAS_SHARDS", raising=False)


def _fixture_cfg(server):
    cfg = ccfa_web()
    adapter = replace(cfg.adapter, base_url=server["base"])
    source = dict(cfg.source_meta)
    source["channel"] = (f"{adapter.base_url}/crawl-data/{adapter.crawl_id}/"
                         "wet.paths.gz (deterministic stride sample, "
                         f"NSEG={adapter.n_segments})")
    return replace(cfg, adapter=adapter, source_meta=source)


def test_cc_end_to_end_byte_equivalence(tmp_path, monkeypatch, fixture_server):
    server = fixture_server
    outs = []
    for tag in ("ref", "fac"):
        root = tmp_path / tag
        _set_env(monkeypatch, root, server)
        if tag == "ref":
            # URL substitution at LOAD time (the frozen file stays
            # byte-verbatim); the literal appears in both the fetch code
            # and the manifest prose, so both engines record the local base
            mod = load_reference(
                "prep_fas_ccfa_web.py",
                {"https://data.commoncrawl.org": server["base"]})
            mod.main()
        else:
            run_prep(_fixture_cfg(server))
        outs.append(root / RUN_REL)

    # full byte equality: segment jsonl + .done markers + shards + sums
    # (manifest compared beyond built_utc)
    assert compare_trees(outs[0], outs[1]) == []

    manifest = read_manifest(outs[0])
    assert manifest["counts"] == EXPECTED_COUNTS

    # content assertions: kept rows, dedupe winner, the pbu quirk
    rows = shard_rows(outs[0], "cc-fa-web")
    assert len(rows) == 2
    uris = sorted(r["uri"] for r in rows)
    assert uris == ["http://ex1/a", "http://ex3/j"]
    # the pbu-header doc (Pashto code in the Persian allowlist) is KEPT —
    # the preserved pre-refactor quirk, now behavior-locked
    pbu = [r for r in rows if r["uri"] == "http://ex3/j"]
    assert pbu and pbu[0]["lang"] == "pbu"
    # nav junk dropped from the kept doc
    keep1 = [r for r in rows if r["uri"] == "http://ex1/a"][0]
    assert "Home | About | Contact" not in keep1["text"]
    assert "ادبیات فارسی" in keep1["text"]
    # rejected docs absent: kurdish / urdu / ta-marbuta / eval-window
    assert all("ڕ" not in r["text"] for r in rows)
    assert all("ٹ" not in r["text"] for r in rows)
    assert all(V_EVAL_SRC not in r["text"] for r in rows)

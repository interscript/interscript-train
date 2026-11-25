"""fa-Wikisource dump adapter equivalence: factory vs the FROZEN
pre-refactor script. The pre-refactor wiki job is a deterministic
single pass — END-TO-END BYTE equivalence (every file byte-identical
except manifest built_utc)."""

from __future__ import annotations

from dataclasses import replace

from acquisition.configs.fas import fawikisource
from acquisition.run import run_prep
from fasfixtures import (
    V_EVAL_SRC,
    compare_trees,
    load_reference,
    read_manifest,
    shard_rows,
)

RUN_REL = "run-098-fas-acquisition/fawikisource"

EXPECTED_COUNTS = {
    "pages_seen": 8,
    "redirects": 1,
    "pages_extracted": 5,
    "filtered_short": 1,
    "filtered_script": 1,
    "decontam_dropped": 1,
    "kept": 2,
}


def _set_env(monkeypatch, root, server):
    monkeypatch.setenv("FAS_CKPT", str(root))
    monkeypatch.delenv("FAS_RUN_DIR", raising=False)
    monkeypatch.setenv("FAS_EVAL_POETRY", str(server["eval_poetry"]))
    monkeypatch.setenv("FAS_EVAL_DOCS", str(server["eval_docs"]))
    monkeypatch.delenv("FAS_SHARDS", raising=False)


def _fixture_cfg(server):
    cfg = fawikisource()
    dump_url = (f"{server['base']}/fawikisource/latest/"
                "fawikisource-latest-pages-articles.xml.bz2")
    source = dict(cfg.source_meta)
    source["dump"] = dump_url  # the manifest records the base (as the
    # frozen script does with its overridden DUMP_URL constant)
    return replace(cfg, adapter=replace(cfg.adapter, dump_url=dump_url),
                   source_meta=source)


def test_wikisource_end_to_end_byte_equivalence(
        tmp_path, monkeypatch, fixture_server):
    server = fixture_server
    outs = []
    for tag in ("ref", "fac"):
        root = tmp_path / tag
        _set_env(monkeypatch, root, server)
        if tag == "ref":
            mod = load_reference("prep_fas_fawikisource.py")
            mod.DUMP_URL = (f"{server['base']}/fawikisource/latest/"
                            "fawikisource-latest-pages-articles.xml.bz2")
            mod.main()
        else:
            run_prep(_fixture_cfg(server))
        outs.append(root / RUN_REL)

    # full byte equality (manifest compared beyond built_utc)
    assert compare_trees(outs[0], outs[1]) == []

    # the exact counter values of the fully-mapped fixture corpus
    manifest = read_manifest(outs[0])
    assert manifest["counts"] == EXPECTED_COUNTS

    # content assertions: template VALUES flattened (the verses survive),
    # link labels resolved, refs/comments/tables dropped
    rows = shard_rows(outs[0], "fawikisource")
    assert len(rows) == 2
    by_title = {r["title"]: r for r in rows}
    divan = by_title["دیوان حافظ"]
    assert "بیتِ نخست" in divan["text"] and "بیتِ دوم" in divan["text"]
    assert "شعر" not in divan["text"].split("بیتِ نخست")[0]
    assert "حافظِ شیراز" in divan["text"]
    assert "منبع" not in divan["text"]
    assert "پنهان" not in divan["text"]
    assert "جدول" not in divan["text"]
    assert "مقدمه" in divan["text"]
    assert "حافظ" in divan["text"]  # bold markers dropped, text kept
    assert "'''" not in divan["text"]
    assert divan["pageid"] == "1"
    # the held-out page never enters the corpus
    assert all(V_EVAL_SRC not in r["text"] for r in rows)

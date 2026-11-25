"""Pytest fixtures: the local HTTP fixture server for acquisition tests.

All corpus content lives in `fasfixtures.py`; this conftest only wires
the session-scoped server and eval files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fasfixtures import (  # noqa: E402
    CC_CRAWL,
    CC_PATHS,
    cc_paths_gz,
    cc_segments,
    eval_docs_rows,
    eval_poetry_rows,
    GANJOOR_CATS,
    GANJOOR_POETS,
    GANJOOR_VERSES,
    wiki_dump_bz2,
)
from fasfixtures import FixtureServer  # noqa: E402


@pytest.fixture(scope="session")
def fixture_server(tmp_path_factory):
    eval_dir = tmp_path_factory.mktemp("evals")
    poetry = eval_dir / "fas-poetry-eval.jsonl"
    docs = eval_dir / "test.jsonl"
    poetry.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                              for r in eval_poetry_rows()), encoding="utf-8")
    docs.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                            for r in eval_docs_rows()), encoding="utf-8")

    routes: list[tuple[str, bytes, str]] = []
    # ganjoor API (paths under /api/ganjoor — the API constant's path)
    routes.append(("/api/ganjoor/poets",
                   json.dumps(GANJOOR_POETS).encode(), "application/json"))
    for cid, cat in GANJOOR_CATS.items():
        routes.append((f"/api/ganjoor/cat/{cid}",
                       json.dumps({"cat": cat}).encode(), "application/json"))
    for pid, verses in GANJOOR_VERSES.items():
        if pid == 103:
            continue  # poem 103 always errors (no route -> 404)
        routes.append((f"/api/ganjoor/poem/{pid}",
                       json.dumps({"id": pid, "verses": verses}).encode(),
                       "application/json"))
    # wikisource dump
    routes.append(("/fawikisource/latest/"
                   "fawikisource-latest-pages-articles.xml.bz2",
                   wiki_dump_bz2(), "application/x-bzip2"))
    # common crawl
    routes.append((f"/crawl-data/{CC_CRAWL}/wet.paths.gz",
                   cc_paths_gz(), "application/gzip"))
    seg_bodies = cc_segments()
    for seg_path in CC_PATHS:
        slug = seg_path.rsplit("/", 1)[-1].replace(".warc.wet.gz", "")
        if slug not in seg_bodies:
            continue  # never fetched (outside the stride sample) — 404 ok
        routes.append((f"/{seg_path}", seg_bodies[slug], "application/gzip"))

    server = FixtureServer(routes)
    server.start()
    yield {"base": server.base, "eval_poetry": poetry, "eval_docs": docs}
    server.stop()

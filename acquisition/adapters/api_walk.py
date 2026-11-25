"""REST collection-walk adapter (the ganjoor.net shape).

list endpoint -> per-entity category-tree DFS -> per-item fetches with a
thread pool. Per-entity raw jsonl + `.done` resume markers land in the
bucket run dir, so the crawl is resumable across job restarts. Rows are
emitted UNFILTERED (src = law-strip of tgt); the counted filter chain
runs in the assemble stage.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from ..config import AcquisitionConfig, ApiWalkConfig
from ..pipeline import jsonl_line, log

_print_lock = Lock()


def _log(msg: str) -> None:
    with _print_lock:
        log(msg)


def api_get(api: ApiWalkConfig, path: str) -> object:
    """Retrying JSON GET — backoff min(30, 2*2^i), any transport/5xx error."""
    url = f"{api.base_url}{path}"
    last = None
    for i in range(api.http_tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": api.user_agent})
            with urllib.request.urlopen(req, timeout=api.http_timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001 — retry any transport/5xx error
            last = e
            time.sleep(min(30.0, 2.0 * (2 ** i)))
    raise RuntimeError(f"api_get failed after {api.http_tries} tries: {url}: {last}")


def entity_inventory(api: ApiWalkConfig, entity: dict) -> tuple[list[dict], dict]:
    """DFS the entity's category tree -> item inventory + walk stats."""
    items: list[dict] = []
    stats = {"cats": 0, "errors": 0}
    name = next((entity[k] for k in api.entity_name_keys
                 if entity.get(k)), "")
    stack = [(entity[api.root_cat_key], name)]
    while stack:
        cid, cpath = stack.pop()
        try:
            d = api_get(api, api.cat_endpoint.format(cid=cid))
        except Exception as e:  # noqa: BLE001
            stats["errors"] += 1
            _log(f"[walk-err] cat {cid}: {e}")
            continue
        stats["cats"] += 1
        cat = d.get(api.cat_wrapper_key, {})
        for p in cat.get(api.cat_poems_key) or []:
            items.append({"poem_id": p[api.poem_id_key],
                          "title": p.get(api.poem_title_key) or "",
                          "cat_id": cid, "cat_path": cpath})
        for c in cat.get(api.cat_children_key) or []:
            stack.append(
                (c["id"], f"{cpath}/{c.get(api.child_slug_key) or c['id']}"))
    return items, stats


def fetch_item(api: ApiWalkConfig, poem_id: object) -> dict | None:
    try:
        return api_get(api, api.item_endpoint.format(pid=poem_id))
    except Exception as e:  # noqa: BLE001
        _log(f"[poem-err] {poem_id}: {e}")
        return None


def entity_slug(api: ApiWalkConfig, entity: dict) -> str:
    return (entity.get(api.entity_url_key) or "").strip("/").replace("/", "_") \
        or str(entity[api.entity_id_key])


def process_entity(cfg: AcquisitionConfig, api: ApiWalkConfig,
                   entity: dict, out_dir: Path) -> None:
    slug = entity_slug(api, entity)
    done = out_dir / f"{slug}.done"
    if done.exists():
        return
    items, stats = entity_inventory(api, entity)
    law = cfg.law
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=api.workers) as ex:
        futs = {ex.submit(fetch_item, api, it["poem_id"]): it
                for it in items}
        for fut in as_completed(futs):
            meta = futs[fut]
            d = fut.result()
            if d is None:
                continue
            for v in d.get(api.verses_key) or []:
                tgt = (v.get(api.verse_text_key) or "").strip()
                if not tgt:
                    continue
                rows.append({
                    "src": law.strip(tgt),
                    "tgt": tgt,
                    "poet_url": entity.get(api.entity_url_key),
                    "poet_name": entity.get("name"),
                    "cat_path": meta["cat_path"],
                    "poem_id": meta["poem_id"],
                    "poem_title": meta["title"],
                    "vorder": v.get(api.verse_order_key),
                    "verse_position": v.get(api.verse_position_key),
                })
    raw_path = out_dir / f"{slug}.jsonl"
    tmp = out_dir / f"{slug}.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(jsonl_line(r))
    tmp.replace(raw_path)
    done.write_text(json.dumps({
        "poet": entity.get(api.entity_url_key), "poems": len(items),
        "verses": len(rows), "cats": stats["cats"],
        "errors": stats["errors"],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    _log(f"[poet] {entity.get(api.entity_url_key)}: poems={len(items)} "
         f"verses={len(rows)} cats={stats['cats']} errors={stats['errors']}")


def fetch_stage(cfg: AcquisitionConfig, t0: float | None = None,
                done_log: Callable[[str], None] | None = None) -> None:
    """Stage A: walk all published entities into resumable raw files."""
    api = cfg.adapter
    assert isinstance(api, ApiWalkConfig)
    raw_dir = cfg.run_dir() / (cfg.raw_subdir or "")
    raw_dir.mkdir(parents=True, exist_ok=True)
    start = t0 if t0 is not None else time.time()
    _log_done = done_log or _log

    poets = api_get(api, api.list_endpoint)
    poets = [p for p in poets if p.get(api.list_published_key)]
    _log(f"[poets] {len(poets)} published")
    with ThreadPoolExecutor(max_workers=api.workers) as ex:
        futs = {ex.submit(process_entity, cfg, api, p, raw_dir): p
                for p in poets}
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if i % 10 == 0:
                _log_done(f"[walk] {i}/{len(poets)} poets "
                          f"elapsed={time.time() - start:.0f}s")

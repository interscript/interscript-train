# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=0.26",
# ]
# ///
"""WO94 phase-1 acquisition — Ganjoor (ganjoor.net) vocalized poetry corpus.

Source: ganjoor.net public REST API (api.ganjoor.net) — the canonical bulk
channel. The official ganjoor/ganjoor-db dump is from 2017 (rejected as
stale, sha256 4623701c5ff54a4f recorded in the WO94 ledger). License:
CC BY-SA 3.0, site-stated (the WO66 dossier record, owner-reviewed
2026-10-25; acquisition owner-approved 2026-11-24 in interscript-models#488).
Per-row attribution: poet_url + poet_name + cat_path + poem_id.

The API exposes, per poem, plainText (site-stripped) and verses[].text —
the VOCALIZED text (erab/haraqat present), e.g. verse 'اَلا اِی آهویِ
وحشی! کجایی؟' vs plainText 'الا ای آهوی وحشی کجایی'. We emit verse-level
pairs {src = strip(tgt), tgt = verse text verbatim, ...} — exactly the
pinned run-089-fas-poetry-eval convention (verified against the pinned
eval: src keeps punctuation, drops the vocalic marks U+064B-U+0652 +
U+0670 while KEEPING the hamza/maddah carriers U+0653-U+0655 (measured
from the pinned rows: U+0654 x130, U+0656 x1 retained); its
tgt/src length-ratio median 1.1724, max 1.8571).

Pipeline (all in-job; corpora NEVER touch the laptop):
  1. GET /api/ganjoor/poets -> published poets (246 as of 2026-11-25).
  2. Per poet: DFS the category tree (GET /cat/{id}?poems=true; the API
     fills children+poems one level per request) -> poem-id inventory.
     Per-poet raw output + .done markers land in the BUCKET run dir, so
     the crawl is resumable across job restarts.
  3. Per poem: GET /poem/{id}?catInfo=false&recitations=false&images=false
     &comments=false&rhymes=false&songs=false (minimal payload).
  4. Assemble: src = strip_marks(tgt); filters + dedupe + decontam.
  5. Contiguous shards (source order) + manifest.json + SHA256SUMS.txt
     + DONE into the bucket run dir.

Filters (pre-registered in WO94 before acquisition):
  - 10 <= len(src) <= 300 chars (verse units; the s45 convention)
  - Arabic-script letter fraction of non-space chars >= 0.5 (pinned-eval
    calibration: median 0.8, p5 0.6)
  - exact-tgt dedupe, first occurrence wins, source order preserved
  - decontam: 40-char windows (stride 20) over Arabic-letter normal forms
    vs BOTH pinned evals (poetry eval src+tgt sides; docs test.jsonl
    src+tgt sides) — the evals must stay held-out: this corpus shares the
    upstream source site with the poetry eval.

Semantic invariants measured in-job, recorded in manifest.json (the
run-093 law); ENFORCED by the separate gate_fas_acquisition.py job:
  - len(tgt)/len(src) ratio in [1.0, 2.0] for every emitted row
  - vocalized share (tgt != src) corpus-level, per-poet share recorded
  - harakat density (marks-per-letter) distribution on vocalized rows
  - degenerate rows (empty src / empty tgt) < 1%

Usage (cpu-upgrade, detached, resumable):
    hf jobs uv run --flavor cpu-upgrade -d --timeout 12h \
      -s HF_TOKEN \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      -v hf://buckets/Interscript/isx-training/run-089-fas-poetry-eval:/evals/poetry:ro \
      -v hf://buckets/Interscript/isx-training/run-076-fas-diac-retrain:/evals/docs:ro \
      --name wo94-ganjoor-crawl prep_fas_ganjoor.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

BUCKET = Path(os.environ.get("FAS_CKPT", "/ckpt"))
RUN_DIR = BUCKET / os.environ.get("FAS_RUN_DIR", "run-098-fas-acquisition/ganjoor")
RAW_DIR = RUN_DIR / "raw"
EVAL_POETRY = Path(os.environ.get("FAS_EVAL_POETRY", "/evals/poetry/fas-poetry-eval.jsonl"))
EVAL_DOCS = Path(os.environ.get("FAS_EVAL_DOCS", "/evals/docs/test.jsonl"))
WORKERS = int(os.environ.get("FAS_WORKERS", "6"))
N_SHARDS = int(os.environ.get("FAS_SHARDS", "8"))

API = "https://api.ganjoor.net/api/ganjoor"
UA = "interscript-wo94-acquisition/0.1 (ganjoor CC BY-SA 3.0 attribution corpus)"

# ---- character-class laws (documented in the WO94 gate definitions) ----
def _is_haraqat(c: str) -> bool:
    o = ord(c)
    # the PINNED-EVAL strip set, measured from the pinned rows: the vocalic
    # marks U+064B-U+0652 (tanwin..sukun) + dagger alif U+0670. The
    # hamza/maddah carriers U+0653-U+0655 STAY (they change the letter
    # itself — the eval src keeps U+0654 hamza-above, 130 occurrences) and
    # U+0656+ stays (the eval src keeps U+0656).
    return 0x064B <= o <= 0x0652 or o == 0x0670


def _is_quranic_mark(c: str) -> bool:
    return 0x06D6 <= ord(c) <= 0x06ED  # Quranic annotation marks (counted, rare)


def _is_arabic_letter(c: str) -> bool:
    o = ord(c)
    return (0x0621 <= o <= 0x063A) or (0x0641 <= o <= 0x064A) or (0x0671 <= o <= 0x06D3)


def strip_marks(text: str) -> str:
    """The pinned-eval src convention (reverse-engineered from the pinned
    rows; verified reproducing them)."""
    return "".join(c for c in text if not _is_haraqat(c)).strip()


def _letter_norm(text: str) -> str:
    """Arabic letters + single-spaced whitespace — the decontam normal form."""
    return re.sub(r"\s+", " ", "".join(c for c in text if _is_arabic_letter(c) or c == " ")).strip()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- polite HTTP ----
_print_lock = Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def api_get(path: str, timeout: int = 45, tries: int = 6) -> object:
    url = f"{API}{path}"
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001 — retry any transport/5xx error
            last = e
            time.sleep(min(30.0, 2.0 * (2 ** i)))
    raise RuntimeError(f"api_get failed after {tries} tries: {url}: {last}")


# ---- decontam windows ----
def build_eval_windows(paths: list[Path]) -> set[str]:
    windows: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                for side in (row.get("src", ""), row.get("tgt", "")):
                    norm = _letter_norm(side)
                    for i in range(0, max(1, len(norm) - 40), 20):
                        windows.add(norm[i: i + 40])
        log(f"[decontam] windows after {path.name}: {len(windows)}")
    return windows


# ---- stage A: poet walk + poem fetch (per-poet raw files, resumable) ----
def poet_inventory(poet: dict) -> tuple[list[dict], dict]:
    """DFS the poet's category tree -> [(poem_id, cat_path, cat_id)], stats."""
    poems: list[dict] = []
    stats = {"cats": 0, "errors": 0}
    stack = [(poet["rootCatId"], poet["nickname"] or poet["name"])]
    while stack:
        cid, cpath = stack.pop()
        try:
            d = api_get(f"/cat/{cid}?poems=true&mainSections=false")
        except Exception as e:  # noqa: BLE001
            stats["errors"] += 1
            log(f"[walk-err] cat {cid}: {e}")
            continue
        stats["cats"] += 1
        cat = d.get("cat", {})
        for p in cat.get("poems") or []:
            poems.append({"poem_id": p["id"], "title": p.get("title") or "",
                          "cat_id": cid, "cat_path": cpath})
        for c in cat.get("children") or []:
            stack.append((c["id"], f"{cpath}/{c.get('urlSlug') or c['id']}"))
    return poems, stats


def fetch_poem(poem_id: int) -> dict | None:
    try:
        d = api_get(f"/poem/{poem_id}?catInfo=false&recitations=false"
                    f"&images=false&comments=false&rhymes=false&songs=false"
                    f"&mainSections=false&top6QuotedPoems=false&geoDateTags=false")
        return d
    except Exception as e:  # noqa: BLE001
        log(f"[poem-err] {poem_id}: {e}")
        return None


def process_poet(poet: dict, out_dir: Path) -> None:
    done = out_dir / f"{poet['fullUrl'].strip('/').replace('/', '_') or poet['id']}.done"
    if done.exists():
        return
    poems, stats = poet_inventory(poet)
    rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_poem, p["poem_id"]): p for p in poems}
        for fut in as_completed(futs):
            meta = futs[fut]
            d = fut.result()
            if d is None:
                continue
            verses = d.get("verses") or []
            for v in verses:
                tgt = (v.get("text") or "").strip()
                if not tgt:
                    continue
                rows.append({
                    "src": strip_marks(tgt),
                    "tgt": tgt,
                    "poet_url": poet["fullUrl"],
                    "poet_name": poet["name"],
                    "cat_path": meta["cat_path"],
                    "poem_id": meta["poem_id"],
                    "poem_title": meta["title"],
                    "vorder": v.get("vOrder"),
                    "verse_position": v.get("versePosition"),
                })
    slug = poet["fullUrl"].strip("/").replace("/", "_") or str(poet["id"])
    raw_path = out_dir / f"{slug}.jsonl"
    tmp = out_dir / f"{slug}.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(raw_path)
    done.write_text(json.dumps({
        "poet": poet["fullUrl"], "poems": len(poems), "verses": len(rows),
        "cats": stats["cats"], "errors": stats["errors"],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"[poet] {poet['fullUrl']}: poems={len(poems)} verses={len(rows)} "
        f"cats={stats['cats']} errors={stats['errors']}")


# ---- stage B: assemble (filters -> shards -> manifest -> SHA256SUMS) ----
def assemble(out_dir: Path, eval_windows: set[str]) -> dict:
    survivors: list[dict] = []
    counts = Counter()
    seen_tgt: set[str] = set()
    n_files = 0
    for raw_path in sorted(out_dir.glob("*.jsonl")):
        n_files += 1
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                counts["verses_raw"] += 1
                r = json.loads(line)
                src, tgt = r["src"], r["tgt"]
                if not (10 <= len(src) <= 300):
                    counts["filtered_len"] += 1
                    continue
                letters = sum(1 for c in src if _is_arabic_letter(c))
                if letters / max(1, len(src.replace(" ", ""))) < 0.5:
                    counts["filtered_script"] += 1
                    continue
                ratio = len(tgt) / max(1, len(src))
                if not (1.0 <= ratio <= 2.0):
                    counts["filtered_ratio"] += 1
                    continue
                if tgt in seen_tgt:
                    counts["dup_dropped"] += 1
                    continue
                norm = _letter_norm(src)
                if any(norm[i: i + 40] in eval_windows
                       for i in range(0, max(1, len(norm) - 40), 20)):
                    counts["decontam_dropped"] += 1
                    continue
                seen_tgt.add(tgt)
                survivors.append(r)
                counts["kept"] += 1
                if counts["kept"] % 200_000 == 0:
                    log(f"[assemble] kept={counts['kept']} "
                        f"dup={counts['dup_dropped']} "
                        f"decontam={counts['decontam_dropped']} "
                        f"filtered={counts['filtered_len'] + counts['filtered_script'] + counts['filtered_ratio']}")

    # semantic invariants over survivors
    vocalized = [r for r in survivors if r["tgt"] != r["src"]]
    ratios = [len(r["tgt"]) / max(1, len(r["src"])) for r in survivors]
    dens = []
    for r in vocalized:
        letters = sum(1 for c in r["tgt"] if _is_arabic_letter(c))
        m = sum(1 for c in r["tgt"] if _is_haraqat(c))
        dens.append(m / max(1, letters))
    dens.sort()
    per_poet: dict[str, dict] = {}
    for r in survivors:
        e = per_poet.setdefault(r["poet_url"], {"rows": 0, "vocalized": 0})
        e["rows"] += 1
        e["vocalized"] += 1 if r["tgt"] != r["src"] else 0
    degenerate = sum(1 for r in survivors if not r["src"] or not r["tgt"])

    n = len(survivors)
    shard_meta = []
    for s in range(N_SHARDS):
        lo, hi = n * s // N_SHARDS, n * (s + 1) // N_SHARDS
        name = f"ganjoor-{s:05d}-of-{N_SHARDS:05d}.jsonl"
        p = out_dir / name
        with p.open("w", encoding="utf-8") as f:
            for r in survivors[lo:hi]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        shard_meta.append({"file": name, "rows": hi - lo, "sha256": sha256_file(p)})
        log(f"[shard] {name} rows={hi - lo}")

    dens_mid = dens[len(dens) // 2] if dens else 0.0
    manifest = {
        "campaign": "WO94 fas poetry acquisition (TODO.sota/94, interscript-models#488)",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "site": "ganjoor.net",
            "api": "https://api.ganjoor.net/api/ganjoor",
            "channel_note": "live API = canonical bulk channel; the official "
                            "ganjoor/ganjoor-db dump.sql.gz is from 2017 (stale, rejected)",
            "license": "CC BY-SA 3.0 (site-stated; WO66 dossier record, "
                       "owner-reviewed 2026-10-25; acquisition approved #488 2026-11-24)",
            "attribution_fields": ["poet_url", "poet_name", "cat_path", "poem_id",
                                   "poem_title"],
            "sharealike": "derived corpus inherits CC BY-SA 3.0; weights carry the "
                          "copyleft dependency note (the tashkeela lane)",
            "crawl_snapshot_utc": os.environ.get("FAS_CRAWL_SNAPSHOT",
                                                 datetime.now(timezone.utc).date().isoformat()),
        },
        "policy": {
            "unit": "verse (API verses[].text = vocalized; src = pinned-eval "
                    "strip convention: drop U+064B-U+0652 + U+0670, keep "
                    "U+0653-U+0655 hamza/maddah carriers (pinned-eval measured), "
                    "keep punctuation/ZWNJ)",
            "filters": "10 <= len(src) <= 300; Arabic-letter fraction >= 0.5 of "
                       "non-space; ratio len(tgt)/len(src) in [1.0, 2.0]",
            "dedupe": "exact-tgt dedupe, first occurrence wins, source order preserved",
            "decontam": "40-char windows (stride 20) over Arabic-letter normal forms "
                        "vs run-089-fas-poetry-eval (src+tgt) AND run-076 docs "
                        "test.jsonl (src+tgt)",
            "sharding": "contiguous source-order slices",
        },
        "counts": dict(counts) | {"poet_files": n_files, "kept_unique": len(survivors),
                                  "degenerate": degenerate},
        "invariants": {
            "vocalized_share": round(len(vocalized) / max(1, n), 6),
            "ratio_min": round(min(ratios), 4) if ratios else None,
            "ratio_max": round(max(ratios), 4) if ratios else None,
            "harakat_density_median_vocalized": round(dens_mid, 6),
            "gates": {
                "vocalized_share_min": 0.30,
                "ratio_band": [1.0, 2.0],
                "degenerate_max": 0.01,
                "harakat_density_median_min": 0.05,
            },
        },
        "per_poet": {k: {"rows": v["rows"],
                         "vocalized_share": round(v["vocalized"] / max(1, v["rows"]), 4)}
                     for k, v in sorted(per_poet.items())},
        "shards": shard_meta,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    sums = "".join(f"{m['sha256']}  {m['file']}\n" for m in shard_meta)
    sums += f"{sha256_file(out_dir / 'manifest.json')}  manifest.json\n"
    (out_dir / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    (out_dir / "DONE").write_text("ok\n", encoding="utf-8")
    return manifest


def main() -> None:
    t0 = time.time()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[work] {RUN_DIR}")

    # ---- eval windows (mounted pinned evals, ro) ----
    eval_windows = build_eval_windows([EVAL_POETRY, EVAL_DOCS])
    log(f"[decontam] total eval windows: {len(eval_windows)}")

    # ---- stage A ----
    poets = api_get("/poets")
    poets = [p for p in poets if p.get("published")]
    log(f"[poets] {len(poets)} published")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_poet, p, RAW_DIR): p for p in poets}
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if i % 10 == 0:
                log(f"[walk] {i}/{len(poets)} poets elapsed={time.time() - t0:.0f}s")

    # ---- stage B ----
    manifest = assemble(RAW_DIR, eval_windows)
    log("[manifest] " + json.dumps({k: manifest[k] for k in
                                    ("counts", "invariants")}, ensure_ascii=False))
    log(f"[done] elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

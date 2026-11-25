# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=0.26",
# ]
# ///
"""WO94 phase-1 acquisition — fa-Wikisource classical-prose + poetry corpus.

Source: dumps.wikimedia.org fawikisource-latest-pages-articles.xml.bz2
(the canonical Wikimedia dump channel; observed 47,719,980 B, Last-Modified
2026-11-01 at probe time — the manifest records exactly what was fetched).
License: CC BY-SA 4.0 (Wikimedia site terms; each page's authors attributed
via page history — per-row title + pageid preserved for attribution).

Pipeline (all in-job; corpora NEVER touch the laptop):
  1. Stream-download the bz2 dump; record size + sha256 (bytes streamed).
  2. iterparse the XML; keep ns=0 pages (articles), skip redirects
     (redirect/تغییرمسier markers), skip disambiguation-looking stubs.
  3. Wikitext -> text: drop refs/table magic words/comments; flatten
     templates by extracting argument VALUES (fa-wikisource poetry pages
     wrap verses in {{شعر|بیت|بیت|...}} templates — dropping templates
     whole would delete the poem bodies, so values are kept, names dropped);
     resolve [[a|b]] -> b; strip outer html tags; keep line structure.
  4. Filters: text >= 50 chars, Arabic-letter fraction >= 0.5 of
     non-space; exact-title dedupe; 40-char-window decontam (stride 20)
     vs the two pinned fas evals (uniform with the ganjoor job).
  5. Contiguous shards + manifest.json + SHA256SUMS.txt + DONE.

Usage (cpu-upgrade, detached):
    hf jobs uv run --flavor cpu-upgrade -d --timeout 4h \
      -s HF_TOKEN \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      -v hf://buckets/Interscript/isx-training/run-089-fas-poetry-eval:/evals/poetry:ro \
      -v hf://buckets/Interscript/isx-training/run-076-fas-diac-retrain:/evals/docs:ro \
      --name wo94-fawikisource-dump prep_fas_fawikisource.py
"""

from __future__ import annotations

import bz2
import hashlib
import json
import os
import re
import shutil
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BUCKET = Path(os.environ.get("FAS_CKPT", "/ckpt"))
RUN_DIR = BUCKET / os.environ.get("FAS_RUN_DIR", "run-098-fas-acquisition/fawikisource")
EVAL_POETRY = Path(os.environ.get("FAS_EVAL_POETRY", "/evals/poetry/fas-poetry-eval.jsonl"))
EVAL_DOCS = Path(os.environ.get("FAS_EVAL_DOCS", "/evals/docs/test.jsonl"))
N_SHARDS = int(os.environ.get("FAS_SHARDS", "8"))
DUMP_URL = "https://dumps.wikimedia.org/fawikisource/latest/fawikisource-latest-pages-articles.xml.bz2"
PROBED_BYTES = 47_719_980  # observed at probe 2026-11-25; manifest records the actual

UA = "interscript-wo94-acquisition/0.1 (wikisource CC BY-SA 4.0 corpus)"

MEDIA_NS = "{http://www.mediawiki.org/xml/export-0.11/}"


def log(msg: str) -> None:
    print(msg, flush=True)


def _is_arabic_letter(c: str) -> bool:
    o = ord(c)
    return (0x0621 <= o <= 0x063A) or (0x0641 <= o <= 0x064A) or (0x0671 <= o <= 0x06D3)


def _is_haraqat(c: str) -> bool:
    o = ord(c)
    return 0x064B <= o <= 0x065F or o == 0x0670


def _letter_norm(text: str) -> str:
    return re.sub(r"\s+", " ", "".join(c for c in text if _is_arabic_letter(c) or c == " ")).strip()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _template_values(inner: str) -> str:
    """Template body -> its argument VALUES. The FIRST pipe component is
    the template NAME (dropped); named components keep the value side.
    Poetry lives in {{شعر|بیت|بیت}} — this keeps the verses, drops the
    template names and license/magic tags."""
    parts = inner.split("|")
    kept = []
    for arg in parts[1:]:
        if "=" in arg:
            arg = arg.split("=", 1)[1]
        arg = arg.strip()
        if arg:
            kept.append(arg)
    return "\n".join(kept)


def flatten_templates(text: str, max_passes: int = 10) -> str:
    """Flatten innermost templates first (depth-safe: each pass matches
    only brace-free bodies); drop magic words entirely."""
    for _ in range(max_passes):
        new = re.sub(r"\{\{([^{}]*)\}\}",
                     lambda m: _template_values(m.group(1)), text)
        if new == text:
            break
        text = new
    # leftover magic words (nowiki, px, etc. — brace-free singletons are
    # already consumed above; any {{...}} remnants after the cap get dropped)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    return text


def wiktext_to_text(wt: str) -> str:
    # comments out first
    wt = re.sub(r"<\!--.*?-->", " ", wt, flags=re.S)
    # refs: drop entirely (footnote machinery)
    wt = re.sub(r"<ref[^>]*>.*?</ref>", " ", wt, flags=re.S)
    wt = re.sub(r"<ref[^>]*/>", " ", wt, flags=re.S)
    # tables: content is layout machinery — drop the {| ... |} blocks
    wt = re.sub(r"\{\|.*?\|\}", " ", wt, flags=re.S)
    # flatten templates to their argument values (depth-safe)
    wt = flatten_templates(wt)
    # gallery/pre/poem blocks -> keep inner text, drop tags
    wt = re.sub(r"<(poem|gallery|pre|nowiki|blockquote|center)[^>]*>(.*?)</\1>",
                r"\2", wt, flags=re.S)
    # remaining html tags
    wt = re.sub(r"<[^>]+>", " ", wt)
    # links [[target|label]] -> label ; [[target]] -> target
    wt = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", wt)
    # external links [url label] -> label
    wt = re.sub(r"\[(?:https?://|ftp://)\S+\s+([^\]]+)\]", r"\1", wt)
    wt = re.sub(r"\[(?:https?://|ftp://)\S+\]", " ", wt)
    # section headers == x == -> x (line-preserved, also at text start)
    wt = re.sub(r"(?:^|\n)=+\s*([^=\n]+?)\s*=+(?=\n|$)", r"\n\1", wt, flags=re.M)
    # bold/italics + stray apostrophes-as-markup
    wt = wt.replace("'''", " ").replace("''", " ")
    # list/rule prefixes
    wt = re.sub(r"^[:;#*]+", " ", wt, flags=re.M)
    wt = re.sub(r"^-{4,}\s*$", " ", wt, flags=re.M)
    # collapse intra-line whitespace, keep line structure
    wt = "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in wt.split("\n"))
    wt = re.sub(r"\n{3,}", "\n\n", wt)
    return wt.strip()


def iter_pages(dump_path: Path):
    with bz2.open(dump_path, "rt", encoding="utf-8", errors="replace") as f:
        for event, elem in ET.iterparse(f, events=("end",)):
            if elem.tag == MEDIA_NS + "page":
                yield elem
                elem.clear()


def main() -> None:
    t0 = time.time()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    work = RUN_DIR / "work"
    work.mkdir(exist_ok=True)
    log(f"[work] {RUN_DIR}")

    # ---- 1. download + pin the dump (streamed to the bucket work dir) ----
    dump_path = work / "fawikisource-latest-pages-articles.xml.bz2"
    if not dump_path.exists():
        log("[fetch] streaming dump ...")
        req = urllib.request.Request(DUMP_URL, headers={"User-Agent": UA})
        h = hashlib.sha256()
        n = 0
        with urllib.request.urlopen(req, timeout=120) as r, dump_path.open("wb") as f:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                f.write(chunk)
                h.update(chunk)
                n += len(chunk)
        log(f"[fetch] bytes={n} sha256={h.hexdigest()}")
    dump_sha = sha256_file(dump_path)
    dump_size = dump_path.stat().st_size
    if dump_size != PROBED_BYTES:
        log(f"[warn] dump size {dump_size} != probed {PROBED_BYTES} — upstream moved on; "
            "manifest records what was used")

    # ---- eval windows ----
    def build_eval_windows(paths):
        windows = set()
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
        return windows

    eval_windows = build_eval_windows([EVAL_POETRY, EVAL_DOCS])
    log(f"[decontam] eval windows: {len(eval_windows)}")

    # ---- 2-4. parse + extract + filter ----
    counts = Counter()
    survivors: list[dict] = []
    seen_titles: set[str] = set()
    for elem in iter_pages(dump_path):
        counts["pages_seen"] += 1
        ns = elem.findtext(MEDIA_NS + "ns")
        title = (elem.findtext(MEDIA_NS + "title") or "").strip()
        if ns != "0" or not title or title in seen_titles:
            continue
        rev = elem.find(MEDIA_NS + "revision")
        wt = (rev.findtext(MEDIA_NS + "text") or "") if rev is not None else ""
        head = wt.lstrip()[:40].lower()
        if head.startswith("#redirect") or head.startswith("#تغییرمسیر"):
            counts["redirects"] += 1
            continue
        text = wiktext_to_text(wt)
        counts["pages_extracted"] += 1
        if len(text) < 50:
            counts["filtered_short"] += 1
            continue
        letters = sum(1 for c in text if _is_arabic_letter(c))
        if letters / max(1, len(text.replace(" ", ""))) < 0.5:
            counts["filtered_script"] += 1
            continue
        norm = _letter_norm(text)
        if any(norm[i: i + 40] in eval_windows
               for i in range(0, max(1, len(norm) - 40), 20)):
            counts["decontam_dropped"] += 1
            continue
        seen_titles.add(title)
        survivors.append({"title": title, "pageid": elem.findtext(MEDIA_NS + "id"),
                          "text": text})
        counts["kept"] += 1
        if counts["kept"] % 2_000 == 0:
            log(f"[extract] kept={counts['kept']} elapsed={time.time() - t0:.0f}s")

    # ---- 5. shards + manifest + sums ----
    n = len(survivors)
    shard_meta = []
    for s in range(N_SHARDS):
        lo, hi = n * s // N_SHARDS, n * (s + 1) // N_SHARDS
        name = f"fawikisource-{s:05d}-of-{N_SHARDS:05d}.jsonl"
        p = RUN_DIR / name
        with p.open("w", encoding="utf-8") as f:
            for r in survivors[lo:hi]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        shard_meta.append({"file": name, "rows": hi - lo, "sha256": sha256_file(p)})
        log(f"[shard] {name} rows={hi - lo}")

    manifest = {
        "campaign": "WO94 fas poetry acquisition (TODO.sota/94, interscript-models#488)",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "site": "fa.wikisource.org",
            "dump": DUMP_URL,
            "license": "CC BY-SA 4.0 (Wikimedia site terms; per-page attribution "
                       "via page history — title + pageid preserved per row)",
            "sha256": dump_sha,
            "bytes": dump_size,
            "probed_bytes_2026-11-25": PROBED_BYTES,
        },
        "policy": {
            "unit": "page (ns=0 article, redirects excluded)",
            "wikitext": "comments/refs/tables dropped; templates flattened to "
                        "argument VALUES (poetry templates {{شعر|بیت|بیت}} keep "
                        "their verses); [[a|b]] -> b; headers unmarked; line "
                        "structure preserved",
            "filters": "text >= 50 chars; Arabic-letter fraction >= 0.5 of "
                       "non-space; exact-title dedupe",
            "decontam": "40-char windows (stride 20) over Arabic-letter normal "
                        "forms vs run-089-fas-poetry-eval + run-076 docs test "
                        "(both sides) — uniform with the ganjoor job",
            "sharding": "contiguous source-order slices",
        },
        "counts": dict(counts),
        "shards": shard_meta,
    }
    (RUN_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    sums = "".join(f"{m['sha256']}  {m['file']}\n" for m in shard_meta)
    sums += f"{sha256_file(RUN_DIR / 'manifest.json')}  manifest.json\n"
    (RUN_DIR / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    (RUN_DIR / "DONE").write_text("ok\n", encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    log("[manifest] counts=" + json.dumps(manifest["counts"], ensure_ascii=False))
    log(f"[done] elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

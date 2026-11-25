# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=0.26",
# ]
# ///
"""WO94 phase-1 acquisition — cc-fa-web: Common Crawl Persian web-register
slice (the OSCAR-fa SUBSTITUTE, see the provenance note).

OSCAR-fa (the WO66 dossier's third source) is NOT ACQUIREABLE as of
2026-11-25: every canonical OSCAR release on HF is gated with manual
approval AND the gate text states, verbatim: "We are forced to temporarily
suspend access to OSCAR ... we will not grant any access until the
situation has been clarified." No archive snapshot, no Zenodo mirror,
mc4 removed from HF; third-party mirrors carry unverifiable lineage.
This job reconstructs the dossier row's REGISTER INTENT — its own framing:
"colloquial/web; Common-Crawl-derived; register = modern web; quality
filters needed" — directly from Common Crawl WET through OSCAR-style
filters. Provenance per row: crawl id + WET segment + source URI.

Source: https://data.commoncrawl.org crawl CC-MAIN-2026-47 (Nov 2026;
100,000 WET segments, ~66MB gz each at probe time). A deterministic,
evenly-strided segment sample (stride over the segment list) keeps the
slice reproducible from the crawl id alone.

License: Common-Crawl-derived. The crawl contents carry their own
per-document terms; Common Crawl's policy permits use and redistribution
of the crawl. Registry treatment: license "LicenseRef-CommonCrawl-Terms"
+ license_review record citing the owner's dossier framing (the same
construction the owner reviewed for OSCAR-fa). Weak-pretrain mass only —
NEVER the quality or eval signal (no-teacher-poison law; machine-mass
never gates quality).

Pipeline (all in-job; corpora NEVER touch the laptop):
  1. Fetch wet.paths.gz for the crawl; take every stride-th segment
     (deterministic from the crawl id + FAS_CC_NSEG).
  2. Per segment (resumable .done markers in the bucket): stream-download,
     parse WARC records, keep WARC-Type: conversion docs whose
     WARC-Identified-Content-Language contains fas/pes/fa (header) AND
     whose Persian-letter fraction passes the floor (belt + suspenders).
  3. Clean lines (drop nav junk: short non-letter lines), collapse blanks.
  4. Filters: doc >= 50 chars, <= 100_000; Persian-letter fraction >= 0.6;
     exact-text dedupe first-occurrence-wins; 40-char-window decontam vs
     the two pinned fas evals (uniform with the other acquisition jobs).
  5. Contiguous shards + manifest.json + SHA256SUMS.txt + DONE.

Usage (cpu-upgrade, detached, resumable):
    hf jobs uv run --flavor cpu-upgrade -d --timeout 12h \
      -s HF_TOKEN \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      -v hf://buckets/Interscript/isx-training/run-089-fas-poetry-eval:/evals/poetry:ro \
      -v hf://buckets/Interscript/isx-training/run-076-fas-diac-retrain:/evals/docs:ro \
      --name wo94-ccfa-web prep_fas_ccfa_web.py
"""

from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import json
import os
import re
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BUCKET = Path(os.environ.get("FAS_CKPT", "/ckpt"))
RUN_DIR = BUCKET / os.environ.get("FAS_RUN_DIR", "run-098-fas-acquisition/cc-fa-web")
RAW_DIR = RUN_DIR / "segments"
EVAL_POETRY = Path(os.environ.get("FAS_EVAL_POETRY", "/evals/poetry/fas-poetry-eval.jsonl"))
EVAL_DOCS = Path(os.environ.get("FAS_EVAL_DOCS", "/evals/docs/test.jsonl"))
N_SHARDS = int(os.environ.get("FAS_SHARDS", "8"))
NSEG = int(os.environ.get("FAS_CC_NSEG", "120"))
CRAWL = os.environ.get("FAS_CC_CRAWL", "CC-MAIN-2026-47")

UA = "interscript-wo94-acquisition/0.1 (common-crawl-derived fa register slice)"


def log(msg: str) -> None:
    print(msg, flush=True)


def _is_arabic_letter(c: str) -> bool:
    o = ord(c)
    return (0x0621 <= o <= 0x063A) or (0x0641 <= o <= 0x064A) or (0x0671 <= o <= 0x06D3)


PERSIAN_SPECIFIC = frozenset("پچژگ")  # Urdu shares several; the letter floor disambiguates
# Script-exclusive rejection (pre-registered): letters these NEIGHBOR
# scripts use that Persian never does. Kurdish Sorani: ڕ ڵ ۆ ێ ڤ ە (and
# ۚ-style marks); Urdu: ٹ ڈ ڑ ں ے. Threshold >= 3 occurrences — a stray
# glyph in a quote is fine, a Kurdish/Urdu document is not.
KURDISH_ONLY = frozenset("ڕڵۏێە")
URDU_ONLY = frozenset("ٹڈڑںے")
ARABIC_TA_MARBUTA = "ة"


def _letter_norm(text: str) -> str:
    return re.sub(r"\s+", " ", "".join(c for c in text if _is_arabic_letter(c) or c == " ")).strip()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- segment selection (deterministic from the crawl id) ----
def segment_list() -> list[str]:
    url = f"https://data.commoncrawl.org/crawl-data/{CRAWL}/wet.paths.gz"
    raw = urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": UA}), timeout=120).read()
    paths = gzip.decompress(raw).decode().splitlines()
    log(f"[paths] crawl={CRAWL} wet_segments={len(paths)}")
    stride = max(1, len(paths) // NSEG)
    chosen = paths[::stride][:NSEG]
    return chosen


# ---- WET parsing ----
def parse_wet_records(data: bytes):
    """Yield (uri, lang_header, text) for WARC-Type: conversion records."""
    with gzip.open(io.BytesIO(data), "rt", encoding="utf-8", errors="replace") as f:
        headers: dict[str, str] = {}
        lines: list[str] = []
        in_content = False
        for line in f:
            line = line.rstrip("\n")
            if in_content:
                if line == "" and lines and lines[-1] == "":
                    yield headers.get("WARC-Target-URI", ""), \
                        headers.get("WARC-Identified-Content-Language", ""), \
                        "\n".join(lines[:-1])
                    lines = []
                    in_content = False
                    continue
                lines.append(line)
                continue
            if line == "":
                if headers.get("WARC-Type") == "conversion":
                    in_content = True
                    lines = []
                else:
                    headers = {}
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip()] = v.strip()
            elif line == "WARC/1.0":
                headers = {}
        # tail record
        if in_content and lines:
            yield headers.get("WARC-Target-URI", ""), \
                headers.get("WARC-Identified-Content-Language", ""), \
                "\n".join(lines)


def clean_doc(text: str) -> str:
    keep = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln:
            keep.append("")
            continue
        letters = sum(1 for c in ln if _is_arabic_letter(c))
        if letters < 3 and len(ln) < 60:
            continue  # nav/menu junk
        keep.append(ln)
    out = "\n".join(keep)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def lang_ok(lang_header: str, text: str) -> bool:
    """Header must be Persian-family WHEN present (fas/pes/fa/pbu); the
    Persian-letter floor always applies (headers are sometimes absent);
    script-exclusive letters reject Kurdish/Urdu neighbors."""
    codes = {c.strip().lower() for c in lang_header.split(",") if c.strip()}
    if codes and not (codes & {"fas", "pes", "fa", "pbu"}):
        return False
    letters = sum(1 for c in text if _is_arabic_letter(c))
    if letters / max(1, len(text.replace(" ", ""))) < 0.6:
        return False
    if sum(1 for c in text if c in KURDISH_ONLY) >= 3:
        return False
    if sum(1 for c in text if c in URDU_ONLY) >= 3:
        return False
    ta = text.count(ARABIC_TA_MARBUTA)
    return ta <= 0.01 * max(1, letters)


# ---- per-segment stage (resumable) ----
def process_segment(seg: str, out_dir: Path) -> None:
    slug = seg.rsplit("/", 1)[-1].replace(".warc.wet.gz", "")
    done = out_dir / f"{slug}.done"
    if done.exists():
        return
    url = f"https://data.commoncrawl.org/{seg}"
    n_docs = 0
    out_path = out_dir / f"{slug}.jsonl"
    tmp = out_dir / f"{slug}.jsonl.tmp"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    with tmp.open("w", encoding="utf-8") as f:
        for uri, lang, text in parse_wet_records(data):
            if not text or len(text) < 50 or len(text) > 100_000:
                continue
            text = clean_doc(text)
            if len(text) < 50:
                continue
            if not lang_ok(lang, text):
                continue
            f.write(json.dumps({"text": text, "uri": uri, "segment": slug,
                                "lang": lang}, ensure_ascii=False) + "\n")
            n_docs += 1
    tmp.replace(out_path)
    done.write_text(json.dumps({"segment": seg, "docs": n_docs}) + "\n")
    log(f"[seg] {slug}: docs={n_docs}")


# ---- assemble ----
def assemble(out_dir: Path, eval_windows: set[str]) -> dict:
    counts = Counter()
    survivors: list[dict] = []
    seen: set[str] = set()
    for seg_path in sorted(out_dir.glob("*.jsonl")):
        with seg_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                counts["docs_raw"] += 1
                r = json.loads(line)
                text = r["text"]
                counts["chars_raw"] += len(text)
                if text in seen:
                    counts["dup_dropped"] += 1
                    continue
                norm = _letter_norm(text)
                if any(norm[i: i + 40] in eval_windows
                       for i in range(0, max(1, len(norm) - 40), 20)):
                    counts["decontam_dropped"] += 1
                    continue
                seen.add(text)
                survivors.append(r)
                counts["kept"] += 1
                if counts["kept"] % 5_000 == 0:
                    log(f"[assemble] kept={counts['kept']} dup={counts['dup_dropped']} "
                        f"decontam={counts['decontam_dropped']}")

    n = len(survivors)
    shard_meta = []
    for s in range(N_SHARDS):
        lo, hi = n * s // N_SHARDS, n * (s + 1) // N_SHARDS
        name = f"cc-fa-web-{s:05d}-of-{N_SHARDS:05d}.jsonl"
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
            "origin": "Common Crawl WET, crawl " + CRAWL,
            "channel": "https://data.commoncrawl.org/crawl-data/"
                       f"{CRAWL}/wet.paths.gz (deterministic stride sample, "
                       f"NSEG={NSEG})",
            "oscar_note": "OSCAR-F SUBSTITUTE — canonical OSCAR releases are "
                          "upstream-suspended (gate text recorded verbatim in the "
                          "WO94 ledger); this slice reconstructs the dossier row's "
                          "register intent from the same underlying source "
                          "(Common-Crawl-derived) through OSCAR-style filters",
            "license": "LicenseRef-CommonCrawl-Terms (per-document terms; CC use/"
                       "redistribution policy) + license_review record citing the "
                       "owner's OSCAR-fa dossier framing — weak-pretrain mass ONLY, "
                       "never the quality/eval signal",
        },
        "policy": {
            "unit": "web document (WARC-Type: conversion)",
            "langid": "WARC-Identified-Content-Language header (fas/pes/fa/pbu) "
                      "AND Persian-letter fraction >= 0.6 floor; script-exclusive "
                      "rejection of Kurdish (ڕڵۏێە) / Urdu (ٹڈڑںے) letters and "
                      "Arabic ta-marbuta-heavy docs",
            "clean": "lines with <3 Arabic letters and <60 chars dropped (nav junk); "
                     "blank runs collapsed",
            "filters": "50 <= len <= 100_000 chars",
            "dedupe": "exact-text dedupe, first occurrence wins (OSCAR-style "
                      "shingle dedupe deferred to training prep)",
            "decontam": "40-char windows (stride 20) vs both pinned fas evals "
                        "(uniform with the other acquisition jobs)",
            "sharding": "contiguous source-order slices",
        },
        "counts": dict(counts) | {"kept_unique": len(survivors)},
        "shards": shard_meta,
    }
    (RUN_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    sums = "".join(f"{m['sha256']}  {m['file']}\n" for m in shard_meta)
    sums += f"{sha256_file(RUN_DIR / 'manifest.json')}  manifest.json\n"
    (RUN_DIR / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    (RUN_DIR / "DONE").write_text("ok\n", encoding="utf-8")
    return manifest


def main() -> None:
    t0 = time.time()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[work] {RUN_DIR}")

    # eval windows
    windows: set[str] = set()
    for path in (EVAL_POETRY, EVAL_DOCS):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                for side in (row.get("src", ""), row.get("tgt", "")):
                    norm = _letter_norm(side)
                    for i in range(0, max(1, len(norm) - 40), 20):
                        windows.add(norm[i: i + 40])
    log(f"[decontam] eval windows: {len(windows)}")

    segs = segment_list()
    log(f"[plan] {len(segs)} segments")
    for i, seg in enumerate(segs, 1):
        try:
            process_segment(seg, RAW_DIR)
        except Exception as e:  # noqa: BLE001 — a bad segment must not kill the crawl
            log(f"[seg-err] {seg}: {e}")
        if i % 10 == 0:
            log(f"[crawl] {i}/{len(segs)} elapsed={time.time() - t0:.0f}s")

    manifest = assemble(RAW_DIR, windows)
    log("[manifest] counts=" + json.dumps(manifest["counts"], ensure_ascii=False))
    log(f"[done] elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

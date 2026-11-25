"""Common Crawl WET slice adapter.

Deterministic, evenly-strided segment sample over the crawl's
wet.paths.gz (reproducible from the crawl id + segment count alone),
per-segment `.done` resume markers, WARC `conversion` records, and the
header+script language law: the WARC-Identified-Content-Language header
must be in the allowlist WHEN present, the letter-fraction floor always
applies, script-exclusive neighbor-script letters reject the document,
and per-character share ceilings (Arabic ta-marbuta for Persian) cap
cross-dialect bleed.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from ..charlaw import ScriptLaw
from ..config import AcquisitionConfig, CcWetConfig
from ..pipeline import jsonl_line, log


def segment_list(cc: CcWetConfig) -> list[str]:
    url = f"{cc.base_url}/crawl-data/{cc.crawl_id}/wet.paths.gz"
    raw = urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": cc.user_agent}),
        timeout=120).read()
    paths = gzip.decompress(raw).decode().splitlines()
    log(f"[paths] crawl={cc.crawl_id} wet_segments={len(paths)}")
    stride = max(1, len(paths) // cc.n_segments)
    chosen = paths[::stride][:cc.n_segments]
    return chosen


def parse_wet_records(data: bytes) -> Iterator[tuple[str, str, str]]:
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


def clean_doc(text: str, cc: CcWetConfig, law: ScriptLaw) -> str:
    """Drop nav junk (short non-letter lines), collapse blank runs."""
    keep = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln:
            keep.append("")
            continue
        letters = sum(1 for c in ln if law.is_letter(c))
        if letters < cc.nav_junk_min_letters and len(ln) < cc.nav_junk_max_len:
            continue
        keep.append(ln)
    out = "\n".join(keep)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def lang_ok(lang_header: str, text: str, cc: CcWetConfig,
            law: ScriptLaw) -> bool:
    """Header must be in the allowlist WHEN present; the letter floor
    always applies; script-exclusive letters and char ceilings reject."""
    codes = {c.strip().lower() for c in lang_header.split(",") if c.strip()}
    if codes and not (codes & cc.lang_allowlist):
        return False
    letters = sum(1 for c in text if law.is_letter(c))
    if letters / max(1, len(text.replace(" ", ""))) < cc.letter_floor:
        return False
    for rej in cc.exclusive_rejects:
        if sum(1 for c in text if c in rej.letters) >= rej.threshold:
            return False
    for ceil in cc.char_ceilings:
        n = text.count(ceil.char)
        if not (n <= ceil.max_ratio * max(1, letters)):
            return False
    return True


def process_segment(seg: str, out_dir: Path, cc: CcWetConfig,
                    law: ScriptLaw) -> None:
    slug = seg.rsplit("/", 1)[-1].replace(".warc.wet.gz", "")
    done = out_dir / f"{slug}.done"
    if done.exists():
        return
    url = f"{cc.base_url}/{seg}"
    n_docs = 0
    out_path = out_dir / f"{slug}.jsonl"
    tmp = out_dir / f"{slug}.jsonl.tmp"
    req = urllib.request.Request(url, headers={"User-Agent": cc.user_agent})
    with urllib.request.urlopen(req, timeout=cc.http_timeout) as r:
        data = r.read()
    with tmp.open("w", encoding="utf-8") as f:
        for uri, lang, text in parse_wet_records(data):
            if not text or len(text) < cc.doc_min or len(text) > cc.doc_max:
                continue
            text = clean_doc(text, cc, law)
            if len(text) < cc.clean_doc_min:
                continue
            if not lang_ok(lang, text, cc, law):
                continue
            f.write(json_line_row(text, uri, slug, lang))
            n_docs += 1
    tmp.replace(out_path)
    done.write_text(json.dumps({"segment": seg, "docs": n_docs}) + "\n")
    log(f"[seg] {slug}: docs={n_docs}")


def json_line_row(text: str, uri: str, slug: str, lang: str) -> str:
    return jsonl_line({"text": text, "uri": uri, "segment": slug, "lang": lang})


def fetch_stage(cfg: AcquisitionConfig, t0: float) -> None:
    """Stage A: download the sampled segments into resumable raw files.
    A bad segment must not kill the crawl."""
    cc = cfg.adapter
    assert isinstance(cc, CcWetConfig)
    raw_dir = cfg.run_dir() / (cfg.raw_subdir or "")
    raw_dir.mkdir(parents=True, exist_ok=True)
    segs = segment_list(cc)
    log(f"[plan] {len(segs)} segments")
    for i, seg in enumerate(segs, 1):
        try:
            process_segment(seg, raw_dir, cc, cfg.law)
        except Exception as e:  # noqa: BLE001 — a bad segment must not kill the crawl
            log(f"[seg-err] {seg}: {e}")
        if i % 10 == 0:
            log(f"[crawl] {i}/{len(segs)} elapsed={time.time() - t0:.0f}s")

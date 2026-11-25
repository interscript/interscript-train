"""Wikimedia XML dump adapter (bz2) — single-pass extraction.

Stream-download the dump (size + sha256 recorded), iterparse ns=0
article pages, skip redirects, dedupe by title (uncounted, BEFORE
conversion — the pre-refactor position), flatten wikitext to visible
text, then run the counted inline chain (len / script / decontam).

The wikitext flattening is markup-level and language-independent:
templates are flattened to their argument VALUES (a poetry template like
{{شعر|بیت|بیت}} keeps its verses — dropping templates whole would delete
the poem bodies), refs/comments/tables dropped, links resolved to
labels, line structure preserved.
"""

from __future__ import annotations

import bz2
import hashlib
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from ..config import AcquisitionConfig, WikiDumpConfig
from ..decontam import hits_windows
from ..pipeline import apply_chain, log

MEDIA_NS = "{http://www.mediawiki.org/xml/export-0.11/}"


def fetch_dump(dump: WikiDumpConfig, dump_path: Path) -> None:
    """Stream-download the dump; log bytes + sha256 as they stream."""
    if dump_path.exists():
        return
    log("[fetch] streaming dump ...")
    req = urllib.request.Request(dump.dump_url, headers={"User-Agent": dump.user_agent})
    h = hashlib.sha256()
    n = 0
    with urllib.request.urlopen(req, timeout=120) as r, dump_path.open("wb") as f:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
            h.update(chunk)
            n += len(chunk)
    log(f"[fetch] bytes={n} sha256={h.hexdigest()}")


def _template_values(inner: str) -> str:
    """Template body -> its argument VALUES. The FIRST pipe component is
    the template NAME (dropped); named components keep the value side."""
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
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    return text


def wiktext_to_text(wt: str) -> str:
    wt = re.sub(r"<\!--.*?-->", " ", wt, flags=re.S)
    wt = re.sub(r"<ref[^>]*>.*?</ref>", " ", wt, flags=re.S)
    wt = re.sub(r"<ref[^>]*/>", " ", wt, flags=re.S)
    wt = re.sub(r"\{\|.*?\|\}", " ", wt, flags=re.S)
    wt = flatten_templates(wt)
    wt = re.sub(r"<(poem|gallery|pre|nowiki|blockquote|center)[^>]*>(.*?)</\1>",
                r"\2", wt, flags=re.S)
    wt = re.sub(r"<[^>]+>", " ", wt)
    wt = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", wt)
    wt = re.sub(r"\[(?:https?://|ftp://)\S+\s+([^\]]+)\]", r"\1", wt)
    wt = re.sub(r"\[(?:https?://|ftp://)\S+\]", " ", wt)
    wt = re.sub(r"(?:^|\n)=+\s*([^=\n]+?)\s*=+(?=\n|$)", r"\n\1", wt, flags=re.M)
    wt = wt.replace("'''", " ").replace("''", " ")
    wt = re.sub(r"^[:;#*]+", " ", wt, flags=re.M)
    wt = re.sub(r"^-{4,}\s*$", " ", wt, flags=re.M)
    wt = "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in wt.split("\n"))
    wt = re.sub(r"\n{3,}", "\n\n", wt)
    return wt.strip()


def iter_pages(dump_path: Path, media_ns: str = MEDIA_NS) -> Iterator[ET.Element]:
    with bz2.open(dump_path, "rt", encoding="utf-8", errors="replace") as f:
        for event, elem in ET.iterparse(f, events=("end",)):
            if elem.tag == media_ns + "page":
                yield elem
                elem.clear()


def run_single_pass(cfg: AcquisitionConfig, windows: set[str],
                    t0: float) -> tuple[list[dict], Counter]:
    """The whole wiki pipeline in one pass over the dump: convert, filter
    (counted), decontaminate, dedupe-by-title (uncounted, pre-conversion)."""
    dump: WikiDumpConfig = cfg.adapter  # type: ignore[assignment]
    assert isinstance(dump, WikiDumpConfig)
    asm = cfg.assemble
    law = cfg.law
    ns = dump.media_ns
    counts: Counter = Counter()
    survivors: list[dict] = []
    seen_titles: set[str] = set()
    chain_seen: set[str] = set()
    dump_path = cfg.run_dir() / "work" / dump.dump_url.rsplit("/", 1)[-1]
    for elem in iter_pages(dump_path, ns):
        counts["pages_seen"] += 1
        ns_text = elem.findtext(ns + "ns")
        title = (elem.findtext(ns + "title") or "").strip()
        if ns_text != dump.article_ns or not title or title in seen_titles:
            continue
        rev = elem.find(ns + "revision")
        wt = (rev.findtext(ns + "text") or "") if rev is not None else ""
        head = wt.lstrip()[:40].lower()
        if any(head.startswith(p) for p in dump.redirect_prefixes):
            counts["redirects"] += 1
            continue
        text = wiktext_to_text(wt)
        counts["pages_extracted"] += 1
        if not apply_chain({"text": text}, asm, law, windows, counts,
                           chain_seen):
            continue
        seen_titles.add(title)
        survivors.append({"title": title,
                          "pageid": elem.findtext(ns + "id"),
                          "text": text})
        counts["kept"] += 1
        if asm.kept_log_every and counts["kept"] % asm.kept_log_every == 0:
            log(f"[extract] kept={counts['kept']} elapsed={time.time() - t0:.0f}s")
    return survivors, counts

"""WO48b — mine fully-vocalized Hebrew Wikisource (CC BY-SA).

he.wikisource.org's poetry/religious namespaces are densely
vocalized. Crawls via the MediaWiki API (generator=allpages /
categorymembers), applies the same strict detector as the he-wiki
leg, emits {src: unpointed, tgt: pointed} pairs. Attribution goes in
the dataset card on publication (CC BY-SA compliance).

Env: M_RUN, M_CAP (100000), VOC_FRAC (0.55), MIN_LETTERS (40).

Usage (cpu-basic): python /ckpt/run-074-heb-wikisource-mine/mine_heb_wikisource.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import re
import requests

OUT = Path("/ckpt") / os.environ.get("M_RUN", "run-074-heb-wikisource-mine")
CAP = int(os.environ.get("M_CAP", "100000"))
VOC_FRAC = float(os.environ.get("VOC_FRAC", "0.55"))
MIN_LETTERS = int(os.environ.get("MIN_LETTERS", "40"))
API = "https://he.wikisource.org/w/api.php"
UA = {"User-Agent": "Interscript-miner/1.0 (https://github.com/interscript; research gold-nikud corpus, CC BY-SA compliance)"}

HEBREW_LETTER = re.compile("[א-ת]")
NIQQUD = re.compile("[֑-ׇ]")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
VAL_FRAC = 0.02

# import the calibrated detector from the he-wiki miner
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from mine_heb_vocalized import paragraph_vocalized  # noqa: E402


def api_pages(limit: int = 50):
    """Category-members crawl (dense vocalized namespaces)."""
    params = {
        "action": "query", "format": "json",
        "generator": "categorymembers",
        "gcmtitle": os.environ.get(
            "WS_CATEGORY", 'קטגוריה:פרקי התנ"ך'),
        "gcmlimit": limit, "gcmsort": "name",
        "prop": "extracts", "explaintext": 1,
    }
    while True:
        for attempt in range(6):
            r = requests.get(API, params=params, headers=UA, timeout=60)
            if r.status_code == 429:
                wait = min(600, 30 * (2 ** attempt))
                print(f"[429] backing off {wait}s", flush=True)
                __import__("time").sleep(wait)
                continue
            r.raise_for_status()
            break
        else:
            raise SystemExit("rate-limited past backoff")
        __import__("time").sleep(1.0)
        data = r.json()
        for page in (data.get("query", {}).get("pages", {}) or {}).values():
            if "extract" in page:
                yield page["title"], page["extract"]
        cont = data.get("continue")
        if not cont:
            break
        params.update(cont)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    f_train = (OUT / "heb-vocalized-train.jsonl").open("w", encoding="utf-8")
    f_val = (OUT / "heb-vocalized-val.jsonl").open("w", encoding="utf-8")
    n = 0
    n_paras = 0
    for title, text in api_pages():
        if n >= CAP:
            break
        for para in text.split("\n"):
            para = para.strip()
            if not paragraph_vocalized(para):
                continue
            n_paras += 1
            for sent in SENT_SPLIT.split(para):
                sent = sent.strip()
                if len(HEBREW_LETTER.findall(sent)) < 8:
                    continue
                tgt, src = sent, NIQQUD.sub("", sent)
                if src == tgt:
                    continue
                n += 1
                row = json.dumps(
                    {"src": src, "tgt": tgt, "title": title},
                    ensure_ascii=False)
                if n % int(1 / VAL_FRAC) == 0:
                    f_val.write(row + "\n")
                else:
                    f_train.write(row + "\n")
                if n >= CAP:
                    break
        if n and n % 5000 < 30:
            print(f"[mine] pairs={n} paras={n_paras} last={title}",
                  flush=True)
    f_train.close()
    f_val.close()
    stats = {"pairs": n, "paras": n_paras, "cap": CAP, "voc_frac": VOC_FRAC}
    print(json.dumps(stats), flush=True)
    (OUT / "mine_stats.json").write_text(json.dumps(stats, indent=1))
    (OUT / "MINE_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

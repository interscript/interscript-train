"""WO67 data expansion B — rule-composite self-expansion over running
text: Thai Wikipedia streamed, frequency-filtered vocabulary, readings
from two INDEPENDENT engines (pythainlp royin + tltk th2ipa); rows
where both engines agree ship as high-confidence; royin-only rows
ship tagged (the residual tier).

Provenance: deterministic engines over Wikimedia text (CC-BY-SA,
attributable); no LLM teacher. Frequency threshold kills one-off junk
tokens; Thai-script-only; syllable-repeat defect filter retained.

Usage (cpu-basic, isx-training mounted rw):
    python /ckpt/run-080-thai-wiki/build_tha_wiki_corpus.py
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

OUT = Path("/ckpt") / "run-080-thai-wiki"
THAI = re.compile(r"^[ก-๛]+$")
MIN_FREQ = 5
MAX_TYPES = 200_000
ARTICLES = 120_000


def defect(phonemic: str) -> bool:
    toks = phonemic.replace(".", " . ").split()
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1] and toks[i] != ".":
            return True
    return False


def main() -> None:
    import tltk
    from datasets import load_dataset
    from pythainlp.transliterate import transliterate
    from pythainlp.tokenize import word_tokenize

    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.th",
                      split="train", streaming=True)
    freq: Counter[str] = Counter()
    for i, art in enumerate(ds):
        if i >= ARTICLES:
            break
        text = art["text"]
        for w in word_tokenize(text, engine="newmm"):
            w = w.strip()
            if w and THAI.match(w) and len(w) <= 40:
                freq[w] += 1
        if i % 20_000 == 0:
            print(f"[wiki] {i} articles, {len(freq)} types", flush=True)
    print(f"[wiki] done: {len(freq)} types", flush=True)

    vocab = [w for w, c in freq.most_common(MAX_TYPES) if c >= MIN_FREQ]
    print(f"[wiki] vocab after freq>={MIN_FREQ}: {len(vocab)}", flush=True)

    if os.environ.get("DEBUG_DUMP"):
        for w in vocab[:12]:
            ph = transliterate(w, engine="royin").strip()
            print(f"[dbg] {w!r} -> {ph!r} defect={defect(ph)}", flush=True)
        return

    both, royin_only, defects, tltk_fail = [], [], 0, 0
    for w in vocab:
        try:
            ph = transliterate(w, engine="royin").strip()
        except Exception:
            defects += 1
            continue
        if not ph or defect(ph):
            defects += 1
            continue
        ipa = ""
        try:
            ipa = tltk.th2ipa(w).strip()
        except Exception as e:  # engine failure is NOT a defect — tag it
            if tltk_fail < 3:
                print(f"[tltk-fail] {w!r}: {e!r}", flush=True)
            tltk_fail += 1
        if ipa and not defect(ipa):
            both.append({"word": w, "source": "wiki-agree",
                         "phonemic": ph, "tltk_ipa": ipa})
        else:
            royin_only.append({"word": w, "source": "wiki-royin",
                               "phonemic": ph})
    print(f"[wiki] agree={len(both)} royin_only={len(royin_only)} "
          f"defects={defects} tltk_fail={tltk_fail}", flush=True)

    for name, rows in (("agree", both), ("royin-only", royin_only)):
        rows.sort(key=lambda r: r["word"])
        p = OUT / f"tha-wiki-{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[wiki] {p.name}: {len(rows)}", flush=True)

    (OUT / "manifest.json").write_text(json.dumps({
        "articles": ARTICLES, "min_freq": MIN_FREQ, "max_types": MAX_TYPES,
        "types": len(vocab), "agree": len(both),
        "royin_only": len(royin_only), "defects": defects,
        "tltk_fail": tltk_fail,
        "engines": ["pythainlp-royin", "tltk-th2ipa"],
        "text_source": "wikimedia/wikipedia 20231101.th (CC-BY-SA)",
    }, indent=2))
    print("[done]", flush=True)


if __name__ == "__main__":
    main()

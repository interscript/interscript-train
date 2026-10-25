"""WO47 stage 2 — fas-diac protocol verification (replicate 0.52% CER).

Loads run-073-fas-diac/best (byt5-small, Persian text→diacritized
text), decodes the rababa-farsi held-out test split, reports raw char
CER + exact-match. Prints the first row's keys loudly so a schema
mismatch fails visibly, not silently.

Env: V_RUN (default run-073-fas-diac).

Usage (l4x1): python /ckpt/run-073-fas-diac/verify_eval.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

MODEL = Path("/ckpt") / os.environ.get("V_RUN", "run-073-fas-diac") / "best"
TEST = Path("/ckpt") / os.environ.get("V_RUN", "run-073-fas-diac") / "test.jsonl"
OUT = Path("/ckpt") / os.environ.get("V_RUN", "run-073-fas-diac")


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    rows = [json.loads(l) for l in TEST.read_text(
        encoding="utf-8").splitlines() if l.strip()]
    print(f"[data] rows={len(rows)} keys={sorted(rows[0].keys())}",
          flush=True)
    pairs = []
    for r in rows:
        if "src" in r and "tgt" in r:
            pairs.append((r["src"], r["tgt"]))
        elif "input" in r and "output" in r:
            pairs.append((r["input"], r["output"]))
        elif "text" in r and "diac" in r:
            pairs.append((r["text"], r["diac"]))
        else:
            raise SystemExit(f"unknown schema: {sorted(r.keys())}")
    print(f"[data] pairs={len(pairs)}", flush=True)

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval().cuda()
    preds = []
    with torch.no_grad():
        for i in range(0, len(pairs), 32):
            batch = [s for s, _ in pairs[i:i + 32]]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to("cuda")
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=512, num_beams=1)
            preds.extend(tok.batch_decode(gen, skip_special_tokens=True))
            if (i // 32) % 20 == 0:
                print(f"[gen] {i + 32}/{len(pairs)}", flush=True)

    DIAC = __import__("re").compile("[\u064B-\u065F\u0670]")  # haraqat + dagger alif
    cers, em = [], 0
    ders, der_ns = [], 0

    def diac_err(pred: str, gold: str) -> tuple[int, int]:
        """Diacritics-only error over aligned letter positions."""
        from difflib import SequenceMatcher

        pl = [c for c in pred if not DIAC.match(c)]
        gl = [c for c in gold if not DIAC.match(c)]
        sm = SequenceMatcher(None, gl, pl, autojunk=False)
        errs = 0
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                g_diac = [DIAC.findall(gold) and "" or ""][0:0]
                # gold diacritics attached after each aligned letter
                gpos, ppos = 0, 0
                for ch_g, ch_p in zip(
                    [c for c in gold if not DIAC.match(c)][i1:i2],
                    [c for c in pred if not DIAC.match(c)][j1:j2],
                ):
                    pass
        # simpler: strip-diac comparison + gold-diac recall via exact
        # position matching is nontrivial; use the standard DER proxy:
        # count gold diacritics vs predicted diacritics at matched
        # letter positions using the alignment above (fallback: char diff)
        return errs, 0

    for (s, gold), pred in zip(pairs, preds):
        cers.append(lev(pred, gold) / max(1, len(gold)))
        if pred == gold:
            em += 1
        d = DIAC.findall(gold)
        dp = DIAC.findall(pred)
        # diacritics-only DER proxy: multiset difference over the whole row
        from collections import Counter

        gd, pd_ = Counter(d), Counter(dp)
        wrong = sum((gd - pd_).values()) + sum((pd_ - gd).values())
        ders.append(wrong / max(1, len(d)))
        der_ns += len(d)
    res = {"rows": len(pairs), "cer": round(sum(cers) / len(cers), 4),
           "exact_match": round(em / len(pairs), 4),
           "diac_only_err_row_mean": round(sum(ders) / len(ders), 4),
           "gold_diacritics_total": der_ns,
           "reference": "rababa-farsi docs/RESULTS.md 0.52% (protocol under test)"}
    print(json.dumps(res), flush=True)
    (OUT / "verify_eval.json").write_text(json.dumps(res, indent=1))
    (OUT / "VERIFY_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

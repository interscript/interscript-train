"""Regenerate run-029 (ara-diac-news) WikiNews-2024 predictions.

The r8 trainer saved only the score json; the WO34 complement probe
needs per-sentence preds. Exact protocol: windowed (600-byte), greedy
generate, project_haraqat onto the input skeleton. Imports the helpers
from the original trainer to avoid transcription drift.

Usage:
    python /ckpt/run-029-news-specialist/eval_r8_wikinews_preds.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/ckpt/run-029-news-specialist")
sys.path.insert(0, "/code")

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from train_arabic_r8_hf import (DIACRITICS_RE, WINDOW, project_haraqat,
                                split_windows)

DATA = Path("/train_data")
OUT = Path("/ckpt/run-029-news-specialist")


def main() -> None:
    best = OUT / "best"
    tok = AutoTokenizer.from_pretrained(best)
    model = AutoModelForSeq2SeqLM.from_pretrained(best).cuda().eval()

    bench = [l for l in (DATA / "wikinews/WikiNews_2024_Multi_Ref.txt.diac")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    winputs = [DIACRITICS_RE.sub("", l) for l in bench]
    wwindows, wcounts = [], []
    for text in winputs:
        ws = split_windows(text)
        wcounts.append(len(ws))
        wwindows.extend(ws)
    print(f"[eval] {len(bench)} lines, {len(wwindows)} windows", flush=True)

    wpreds = []
    with torch.no_grad():
        for i in range(0, len(wwindows), 32):
            batch = wwindows[i:i + 32]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=WINDOW).to("cuda")
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=WINDOW * 2, num_beams=1)
            wpreds.extend(tok.batch_decode(gen, skip_special_tokens=True))
            if (i // 32) % 10 == 0:
                print(f"[gen] {i + len(batch)}/{len(wwindows)}", flush=True)

    k = 0
    wparas = []
    for text, c in zip(winputs, wcounts):
        wparas.append(project_haraqat(" ".join(wpreds[k:k + c]), text))
        k += c
    (OUT / "wikinews_preds.txt").write_text("\n".join(wparas) + "\n", encoding="utf-8")

    import eval_wikinews_multiref as WN

    score = WN.score(wparas, bench, skip_last=False)
    print(json.dumps({"run": "run-029-news-specialist",
                      "surface": "wikinews-2024-multiref", **score}), flush=True)
    (OUT / "wikinews_multiref.json").write_text(
        json.dumps({"run": "run-029-news-specialist", **score}, indent=1))
    (OUT / "PREDS_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

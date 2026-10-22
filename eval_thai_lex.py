"""WO67 — the proper Thai lexicon-arm dev metrics: phoneme-token PER,
tone accuracy, and raw string CER, on the final oepoch checkpoint.

The trainer gates on raw string CER (structural-char-dominated — 41%
of target chars are separators); the task convention is PER over
phoneme tokens with tone scored separately (the WO82 layer split).

Usage (a100-large or gpu, isx-training mounted):
    python /ckpt/run-080-thai-lex/eval_thai_lex.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi

DATA = Path("/ckpt") / os.environ.get("E_DATA", "run-080-thai-lexicon")
CKPT = Path("/ckpt") / os.environ.get("E_CKPT", "run-080-thai-lex")


def load(name: str) -> list[tuple[str, str]]:
    rows = [json.loads(l) for l in (DATA / name).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    return [(r["word"].strip(), r["phonemic"].strip()) for r in rows
            if r.get("word", "").strip() and r.get("phonemic", "").strip()]


def token_err(pred: list[str], tgt: list[str]) -> int:
    prev = list(range(len(tgt) + 1))
    for i, a in enumerate(pred, 1):
        cur = [i]
        for j, b in enumerate(tgt, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1]


def main() -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    eps = sorted(CKPT.glob("oepoch-*"), key=lambda p: int(p.name.split("-")[1]))
    latest = eps[-1]
    model = AutoModelForSeq2SeqLM.from_pretrained(latest).cuda().eval()
    tok = AutoTokenizer.from_pretrained(latest)
    dev = load("tha-phonemic-test.jsonl")
    print(f"[eval] {latest.name}, {len(dev)} rows", flush=True)

    n_tok = n_err = n_str = n_str_chars = 0
    with torch.no_grad():
        for src, tgt in dev:
            ids = tok(src, return_tensors="pt", truncation=True,
                      max_length=256).input_ids.cuda()
            out = model.generate(ids, max_new_tokens=256, num_beams=1)
            pred = tok.decode(out[0], skip_special_tokens=True)
            pt, gt = pred.split(), tgt.split()
            n_tok += len(gt)
            n_err += token_err(pt, gt)
            n_str_chars += max(len(tgt), 1)
            prev = list(range(len(tgt) + 1))
            for i, a in enumerate(pred, 1):
                cur = [i]
                for j, b in enumerate(tgt, 1):
                    cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                                   prev[j - 1] + (a != b)))
                prev = cur
            n_str += prev[-1]
    per = n_err / max(n_tok, 1)
    cer = n_str / max(n_str_chars, 1)
    verdict = {
        "checkpoint": latest.name,
        "rows": len(dev),
        "phoneme_tokens": n_tok,
        "token_errors": n_err,
        "per": round(per, 4),
        "string_cer": round(cer, 4),
        "convention": ("PER = phoneme-token Levenshtein / gold tokens "
                       "(tone marks are tokens). Tone accuracy needs "
                       "aligned-skeleton pairing — tracked separately "
                       "before any tone claim is published"),
    }
    print(json.dumps(verdict), flush=True)
    (CKPT / "eval-verdict.json").write_text(json.dumps(verdict, indent=2))
    HfApi(token=True)  # mounted-bucket writeback flushes on shutdown


if __name__ == "__main__":
    main()

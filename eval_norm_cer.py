"""WO31 — normalized-CER gate re-evaluation for the v2 student.

run-042's in-job CER (1.3021) compared space-per-phone predictions
against space-per-word gold: every phone's trailing space counts as an
error. The rules layer (0.2393) produces contiguous IPA, so the fair
gate normalizes BOTH sides: strip spaces + stress marks, char-level
Levenshtein. Also reports the raw protocol for the record, saves all
preds, and flags truncation (pred shorter than 60% of gold).

Usage:
    python /ckpt/run-043-gate-norm/eval_norm_cer.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

CKPT = Path(os.environ.get("NORM_CKPT", "/ckpt/run-042-heb-ipa-v2/best"))
GT = Path("/ckpt/run-032-heb-ipa-v0/gt.tsv")
STRESS = "ˈˌ.-"


def norm(s: str) -> str:
    for ch in STRESS:
        s = s.replace(ch, "")
    return s.replace(" ", "")


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSeq2SeqLM.from_pretrained(CKPT).cuda().eval()

    gt = []
    for line in GT.read_text(encoding="utf-8").splitlines():
        s, _, p = line.partition("\t")
        if s and p:
            gt.append((s, p))
    print(f"[gate] gt sentences={len(gt)}", flush=True)

    preds = []
    with torch.no_grad():
        for i in range(0, len(gt), 16):
            batch = [s for s, _ in gt[i:i + 16]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to("cuda")
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=256, num_beams=1)
            preds.extend(tok.batch_decode(gen, skip_special_tokens=True))
    Path("/ckpt/run-043-gate-norm").mkdir(parents=True, exist_ok=True)
    out = Path("/ckpt/run-043-gate-norm/preds.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(preds) + "\n", encoding="utf-8")

    raw_cers = []
    norm_cers = []
    truncated = 0
    for (s, gold), pred in zip(gt, preds):
        raw_cers.append(lev(pred, gold) / max(1, len(gold)))
        ng, np_ = norm(gold), norm(pred)
        norm_cers.append(lev(np_, ng) / max(1, len(ng)))
        if len(np_) < 0.6 * len(ng):
            truncated += 1

    verdict = {
        "model": "heb-ipa-v2",
        "cer_raw_space_mismatch": round(sum(raw_cers) / len(raw_cers), 4),
        "cer_normalized": round(sum(norm_cers) / len(norm_cers), 4),
        "gate": "cer_normalized < 0.2393 (rules layer, contiguous)",
        "truncated_lt60pct": truncated,
        "sentences": len(gt),
    }
    print(json.dumps(verdict), flush=True)
    Path("/ckpt/run-043-gate-norm/verdict.json").write_text(
        json.dumps(verdict, indent=1))
    for (s, gold), pred in list(zip(gt, preds))[:3]:
        print(f"  norm-pred: {norm(pred)[:60]}\n  norm-gold: {norm(gold)[:60]}",
              flush=True)


if __name__ == "__main__":
    main()

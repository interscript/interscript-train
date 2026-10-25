"""WO47 closer — run-079/best on the docs split, BOTH CER conventions.

run-001's 0.52% is CORPUS-level CER (modal_app_diacrit.py evaluate():
total_ed / total_gold_chars); our gates report the row-mean. This job
decodes the docs split once and reports both, so the gate compares
like with like.

Usage (l4x1): python /ckpt/run-079-fas-diac-docslr/eval_fas_metrics.py
"""

import json
import os
from pathlib import Path

import torch

CKPT = Path("/ckpt") / os.environ.get("E_RUN", "run-079-fas-diac-docslr") / "best"
TEST = Path("/ckpt") / os.environ.get("E_TEST", "run-076-fas-diac-retrain") / "test.jsonl"
OUT = Path("/ckpt") / os.environ.get("E_RUN", "run-079-fas-diac-docslr") / "metrics-verdict.json"


def ed(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


rows = [json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()]
examples = [((r.get("src") or "").strip(), (r.get("tgt") or "").strip())
            for r in rows if (r.get("src") or "").strip() and (r.get("tgt") or "").strip()]

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForSeq2SeqLM.from_pretrained(CKPT).eval().cuda()

total_ed = 0
total_gold = 0
row_sum = 0.0
exact = 0
preds_path = CKPT.parent / "docs-split-preds.jsonl"
with preds_path.open("w", encoding="utf-8") as pf, torch.no_grad():
    for i in range(0, len(examples), 8):
        batch = examples[i:i + 8]
        enc = tok([s for s, _ in batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to("cuda")
        out = model.generate(**enc, max_new_tokens=512, num_beams=1)
        preds = tok.batch_decode(out, skip_special_tokens=True)
        for (src, gold), p in zip(batch, preds):
            p = p.strip()
            d = ed(list(p), list(gold))
            total_ed += d
            total_gold += len(gold.strip())
            row_sum += d / max(1, len(gold.strip()))
            exact += int(p == gold.strip())
            pf.write(json.dumps({"src": src, "gold": gold.strip(), "pred": p},
                                ensure_ascii=False) + "\n")

n = len(examples)
res = {
    "rows": n,
    "model": os.environ.get("E_RUN", "run-079-fas-diac-docslr"),
    "corpus_cer": round(total_ed / max(1, total_gold), 5),  # the run-001 convention
    "row_mean_cer": round(row_sum / n, 5),                  # our gate convention
    "word_acc": round(exact / n, 4),
    "reference_corpus_cer": 0.0052,
    "verdict_corpus": "PASS" if total_ed / max(1, total_gold) <= 0.0052 else "MISS",
    "verdict_row_mean": "PASS" if row_sum / n <= 0.0052 else "MISS",
}
print(json.dumps(res), flush=True)
OUT.write_text(json.dumps(res, indent=1))
try:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    for f in (OUT, preds_path):
        dst = f"buckets/Interscript/isx-training/{f.relative_to(Path('/ckpt'))}"
        with open(f, "rb") as fh, fs.open(dst, "wb") as g:
            g.write(fh.read())
        print("synced", fs.exists(dst), flush=True)
except Exception as e:  # noqa: BLE001
    print(f"sync failed: {e}", flush=True)

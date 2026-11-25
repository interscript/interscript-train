# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers<5",
#   "accelerate>=0.26",
#   "sentencepiece",
#   "Levenshtein",
#   "huggingface_hub>=0.26",
# ]
# ///
"""WO94 docs-instrument re-baseline — the clean-instrument eval.

Runs the IDENTICAL greedy decode protocol (the lineage crown protocol:
num_beams=1, MAXLEN 256, bs 16, row-mean RAW char CER + word accuracy —
the run-099 evaluator's exact metric code) on the contamination-controlled
docs test (isx-training:/wo94-docs-instrument/docs-test-clean/, 7,621
rows) for ONE model per job:

  job A (crown):  /ckpt/run-078-fas-diac-base/best  (the shipped
                  fas-vocalized-base-1.0 tier's torch source; its
                  historical contaminated-instrument row-mean = 0.603%)
  job B (r2):     /ckpt/run-099-fas-vocalized-sft/best (run-099 attempt-2)

Set FAS_MODEL to the checkpoint dir. The clean-test pin is verified
in-job (SHA256SUMS re-hash). Verdict written to
<model-dir>-clean-eval/eval-verdict.json in the bucket (next to the
model dir), plus DONE.

Usage (l4x1, detached — eval-only, no training):
    hf jobs uv run --flavor l4x1 -d --timeout 8h -s HF_TOKEN \
      -e FAS_MODEL=/ckpt/run-078-fas-diac-base/best \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      --name wo94-clean-eval-crown eval_docs_clean_hf.py
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch

MODEL = Path(os.environ["FAS_MODEL"])
CLEAN = Path("/ckpt/wo94-docs-instrument/docs-test-clean")
MAXLEN = 256
BS = 16


def log(m):
    print(m, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lev(a: str, b: str) -> int:
    try:
        from Levenshtein import distance

        return distance(a, b)
    except ImportError:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def main() -> None:
    t0 = time.time()

    # ---- instrument pin ----
    for line in (CLEAN / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sha, name = line.split("  ", 1)
        got = sha256_file(CLEAN / name)
        if got != sha:
            raise SystemExit(f"PIN FAIL: {name} sha {got} != {sha}")
    log("[pins] clean instrument SHA256SUMS clean")

    rows = []
    with (CLEAN / "docs-test-clean.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                s, t = (r.get("src") or "").strip(), (r.get("tgt") or "").strip()
                if s and t:
                    rows.append((s, t))
    log(f"[data] clean docs test rows={len(rows)}")

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).cuda().eval()
    log(f"[model] {MODEL} loaded")

    cers, correct = [], 0
    with torch.no_grad():
        for i in range(0, len(rows), BS):
            batch = [s for s, _ in rows[i:i + BS]]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=MAXLEN).to("cuda")
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=MAXLEN, num_beams=1)
            preds = tok.batch_decode(gen, skip_special_tokens=True)
            for (_, gold), pred in zip(rows[i:i + BS], preds):
                cers.append(lev(pred, gold) / max(1, len(gold)))
                if pred.strip() == gold.strip():
                    correct += 1
            if (i // BS) % 50 == 0:
                log(f"[eval] {i + len(batch)}/{len(rows)}")

    res = {
        "model": str(MODEL),
        "instrument": "wo94-docs-instrument/docs-test-clean (7,621 rows, "
                      "contamination-controlled)",
        "protocol": "greedy num_beams=1 MAXLEN 256 bs 16 row-mean raw CER",
        "rows": len(rows),
        "cer_row_mean": round(sum(cers) / max(1, len(cers)), 6),
        "word_acc": round(correct / max(1, len(rows)), 6),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
    }
    parts = str(MODEL).strip("/").split("/")
    model_tag = f"{parts[-2]}-{parts[-1]}"  # e.g. run-078-fas-diac-base-best
    out_dir = Path("/ckpt") / f"{model_tag}-clean-eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval-verdict.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "DONE").write_text("ok\n", encoding="utf-8")
    log("[verdict] " + json.dumps(res, ensure_ascii=False))
    log(f"[done] elapsed={res['elapsed_s']}s")


if __name__ == "__main__":
    main()

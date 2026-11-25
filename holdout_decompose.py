# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=0.26",
# ]
# ///
"""WO94 — decompose the run-099 docs-side holdout drops (a cpu probe).

The r2 holdout dropped 9,270 docs rows (dropped-rows.jsonl, sha256
2b1c29fe9992ad52cc062d985f72e0be9f623fabfa52439970180c2000beb00c). This
probe classifies each drop by WHICH key matched — docs-test src, docs-test
tgt, poetry-eval src, poetry-eval tgt — over the bucket-mounted corpora
(no corpus data leaves the bucket), and reports:
  - src-side vs tgt-side counts (identical inputs = severe train/test
    contamination; shared targets only = benign-ish collisions)
  - the src-side overlap fraction of the 9,288-row docs test (how much of
    the gate set is visible in the register train split)

Writes the breakdown to
isx-training:/run-099-fas-vocalized-sft/holdout-decomposition.json.

Usage (cpu-upgrade, detached):
    hf jobs uv run --flavor cpu-upgrade -d --timeout 2h -s HF_TOKEN \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      --name wo099-holdout-decompose holdout_decompose.py
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

CKPT = Path("/ckpt")
RUN = CKPT / "run-099-fas-vocalized-sft"
DOCS = CKPT / "run-076-fas-diac-retrain"
POETRY = CKPT / "run-089-fas-poetry-eval/fas-poetry-eval.jsonl"


def log(m):
    print(m, flush=True)


def load(path, keys=("src", "tgt")):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append((r.get(keys[0], "").strip(), r.get(keys[1], "").strip()))
    return rows


def main():
    log("[probe] loading corpora from the bucket ...")
    docs_train = load(DOCS / "train.jsonl")
    docs_test = load(DOCS / "test.jsonl")
    poetry = load(POETRY)
    test_srcs = {s for s, _ in docs_test}
    test_tgts = {t for _, t in docs_test}
    ev_srcs = {s for s, _ in poetry}
    ev_tgts = {t for _, t in poetry}
    log(f"[probe] docs_train={len(docs_train)} docs_test={len(docs_test)} poetry={len(poetry)}")

    reasons = Counter()
    src_side = tgt_side = 0
    matched_test_srcs = set()
    train_src_set = {s for s, _ in docs_train}
    test_src_in_train = sum(1 for s in test_srcs if s in train_src_set)
    for s, t in docs_train:
        hit = False
        if s in test_srcs:
            reasons["docs_test_src"] += 1
            matched_test_srcs.add(s)
            hit = True
        if t in test_tgts:
            reasons["docs_test_tgt"] += 1
            hit = True
        if s in ev_srcs:
            reasons["poetry_eval_src"] += 1
            hit = True
        if t in ev_tgts:
            reasons["poetry_eval_tgt"] += 1
            hit = True
        if hit:
            if s in test_srcs:
                src_side += 1
            else:
                tgt_side += 1
    breakdown = {
        "total_dropped": len(open(RUN / "dropped-rows.jsonl", encoding="utf-8").read().splitlines()),
        "reason_counts": dict(reasons),
        "src_side_drops": src_side,
        "tgt_side_drops": tgt_side,
        "docs_test_srcs_total": len(test_srcs),
        "docs_test_srcs_also_in_train": test_src_in_train,
        "docs_test_srcs_also_in_train_fraction": round(test_src_in_train / max(1, len(test_srcs)), 6),
    }
    log("[breakdown] " + json.dumps(breakdown, ensure_ascii=False))
    (RUN / "holdout-decomposition.json").write_text(
        json.dumps(breakdown, ensure_ascii=False, indent=1), encoding="utf-8")
    (RUN / "DECOMPOSE_DONE").write_text("ok\n", encoding="utf-8")
    log("[done]")


if __name__ == "__main__":
    main()

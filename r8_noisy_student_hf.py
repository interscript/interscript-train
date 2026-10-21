"""WO64/B3 stage 1 — r8 noisy-student synthesis (single canonical teacher).

Decodes run-047's exact silver corpus recipe (build_rows: seed-42
shuffle, D_SUBSET rows) with the canonical teacher (p047 =
run-047-large-strip — the ara-diac-2.0 lineage, DER 2.29) and writes
noisy-targets.jsonl {src, tgt} to the bucket. Teacher output refines
the ~3% WER QCRI silver labels; stage 2 retrains from these targets.
Env: D_RUN (default run-080-r8-noisy), D_SUBSET (default 300000).
Usage (a100-large):
    python /ckpt/run-080-r8-noisy/r8_noisy_student_hf.py
"""
from __future__ import annotations

import json

from distill_oracle_hf import CKPT, build_rows, decode_arm, sync_to_bucket

TEACHER = ("p047", "/ckpt/run-047-large-strip/best", 32)


def main() -> None:
    rows = build_rows()
    srcs = [s for s, _ in rows]
    # decode_arm resumes internally (preds-p047.jsonl, durable syncs)
    # and returns the full prediction list
    dec = decode_arm(*TEACHER, srcs)
    assert len(dec) == len(srcs), f"{len(dec)} preds for {len(srcs)} srcs"
    out = CKPT / "noisy-targets.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for s, t in zip(srcs, dec):
            f.write(json.dumps({"src": s, "tgt": t},
                               ensure_ascii=False) + "\n")
        f.flush()
    sync_to_bucket(out)
    print(f"[r8] wrote {len(dec)} noisy rows -> {out}", flush=True)


if __name__ == "__main__":
    main()

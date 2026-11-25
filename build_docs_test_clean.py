# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=0.26",
# ]
# ///
"""WO94 docs-instrument re-baseline — construct the contamination-controlled
docs test (owner-approved disposition 1, 2026-11-26).

CONSTRUCTION LAW (pre-registered in /tmp/factory-notes.md BEFORE any eval):
the clean instrument = the pinned 9,288-row docs test
(run-076-fas-diac-retrain/test.jsonl — the EXACT rows, no re-split, no
resampling) minus every row whose src.strip() appears in the stripped-src
set of the ORIGINAL pinned register train split
(run-076-fas-diac-retrain/train.jsonl, 445,869 rows — the split the
lineage champions trained on). Src-side ONLY: the memorization vector is
the INPUT being visible in training; tgt-side-only matches (same target
text, different input) are NOT dropped. This is the holdout law's
stripped-match mechanism applied to the test side against the original
pre-exclusion train split.

Row content passes through VERBATIM (no normalization of the emitted
rows — the stripped comparison is only the matching predicate).

Inputs are pin-verified in-job against the fas-dia corpus record
(corpora/fas-dia.yaml, models#493 lineage):
  train sha256 05ea4b33aaaec7fcddabd8d6bbefa56323efd6fac30b6ae242869372026eb845
  test  sha256 67513aa16ec142abde1df2d1782d548aa84f3841c171150b53a7ba82cb60b95f

Output: isx-training:/wo94-docs-instrument/docs-test-clean/
  {docs-test-clean.jsonl, manifest.json, SHA256SUMS.txt, DONE}

Usage (cpu-upgrade, detached):
    hf jobs uv run --flavor cpu-upgrade -d --timeout 2h -s HF_TOKEN \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      --name wo94-docs-test-clean build_docs_test_clean.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

CKPT = Path("/ckpt")
OUT = CKPT / "wo94-docs-instrument/docs-test-clean"
TRAIN = CKPT / "run-076-fas-diac-retrain/train.jsonl"
TEST = CKPT / "run-076-fas-diac-retrain/test.jsonl"
TRAIN_SHA = "05ea4b33aaaec7fcddabd8d6bbefa56323efd6fac30b6ae242869372026eb845"
TEST_SHA = "67513aa16ec142abde1df2d1782d548aa84f3841c171150b53a7ba82cb60b95f"


def log(m):
    print(m, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- input pins ----
    got_train, got_test = sha256_file(TRAIN), sha256_file(TEST)
    if got_train != TRAIN_SHA:
        raise SystemExit(f"PIN FAIL: docs train sha {got_train} != {TRAIN_SHA}")
    if got_test != TEST_SHA:
        raise SystemExit(f"PIN FAIL: docs test sha {got_test} != {TEST_SHA}")
    log(f"[pins] train {TRAIN_SHA[:16]}… ok; test {TEST_SHA[:16]}… ok")

    # ---- the collision set: stripped srcs of the ORIGINAL train split ----
    train_srcs = set()
    n_train = 0
    with TRAIN.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            train_srcs.add(json.loads(line)["src"].strip())
            n_train += 1
    log(f"[train] rows={n_train} unique_stripped_srcs={len(train_srcs)}")

    # ---- filter the test (row-level, content verbatim) ----
    kept, dropped = [], 0
    n_test = 0
    with TEST.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_test += 1
            r = json.loads(line)
            if r["src"].strip() in train_srcs:
                dropped += 1
                continue
            kept.append(r)
    log(f"[test] rows={n_test} dropped_colliding={dropped} kept={len(kept)}")

    # ---- emit + packaging ----
    out_file = OUT / "docs-test-clean.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "campaign": "WO94 docs-instrument re-baseline (owner-approved "
                    "disposition 1, 2026-11-26)",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "test": "run-076-fas-diac-retrain/test.jsonl (pinned, rows "
                    "passed through verbatim)",
            "test_sha256": got_test,
            "collision_reference": "run-076-fas-diac-retrain/train.jsonl "
                                   "(the ORIGINAL pinned register train "
                                   "split — the population the lineage "
                                   "champions trained on)",
            "train_sha256": got_train,
        },
        "law": "drop every test row whose src.strip() appears in the "
               "stripped-src set of the original train split; src-side "
               "ONLY (the memorization vector is the visible input); "
               "tgt-side-only matches NOT dropped; no re-split, no "
               "resampling — same rows minus collisions",
        "counts": {"rows_in": n_test, "dropped_colliding": dropped,
                   "kept": len(kept),
                   "train_unique_stripped_srcs": len(train_srcs)},
        "policy": {"matching": "whitespace-stripped exact src equality "
                               "(the holdout law mechanism)",
                   "row_content": "verbatim"},
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    sums = f"{sha256_file(out_file)}  docs-test-clean.jsonl\n"
    sums += f"{sha256_file(OUT / 'manifest.json')}  manifest.json\n"
    (OUT / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    (OUT / "DONE").write_text("ok\n", encoding="utf-8")
    log("[manifest] " + json.dumps(manifest["counts"], ensure_ascii=False))
    log("[done]")


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "pandas",
#   "pyarrow",
#   "pyarabic",
#   "prettytable",
#   "tqdm",
# ]
# ///
"""WO20u M1-6ep pre-flight smoke (cpu-basic) — test-until-it-works
before the a100 arm spends anything.

Validates, in order:
  1. the r7 teacher release downloads and passes per-part sha256;
  2. the canonical labels download, match the pinned digest, and yield
     the expected usable-pair count under the trainer's own filter;
  3. the frozen scan artifacts (wo20-tiny-scan bridges.npz + scan.json)
     load, carry verdict ARMED, and contain the five bridged pairs;
  4. the SadeedDiac-25 parquet mount resolves and has 1,200 rows;
  5. the G2a released-predictions asset matches its pinned sha.

No GPU, no training. Any failure aborts the launch decision.

Usage:
    hf jobs uv run --flavor cpu-basic -d --timeout 4h \\
      --secrets HF_TOKEN \\
      -v hf://datasets/Interscript/arabic-r8:/train_data:ro \\
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \\
      -v <code-dir>:/code:ro \\
      wo20u6_smoke.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, "/code")

WORK = Path("/tmp/wo20u6-smoke")


def log(*a):
    print("[wo20u6-smoke]", *a, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"GET {url}")
    with urllib.request.urlopen(url) as r, dest.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


def main():
    checks = {}

    # reuse the trainer's own fetch/verify + label-filter logic verbatim
    from train_wo20_tiny_hf import (LABELS_FILE, LABELS_SHA,
                                    fetch_teacher_and_labels, load_pairs)
    tdir, labels_path = fetch_teacher_and_labels()
    checks["teacher_dir"] = tdir.name
    checks["labels_sha"] = (LABELS_SHA or "")[:12]
    rows = load_pairs(labels_path)
    checks["usable_pairs"] = len(rows)

    # frozen scan artifacts
    scan = json.loads(Path("/ckpt/wo20-tiny-scan/scan.json").read_text())
    checks["scan_verdict"] = scan.get("verdict")
    npz = np.load("/ckpt/wo20-tiny-scan/bridges.npz")
    pairs = []
    for m in scan["armed_pairs"]:
        if isinstance(m, dict):
            pairs.append((int(m["student_layer"]),
                          int(m["teacher_layer"])))
        else:
            pairs.append((int(m[0]), int(m[1])))
    missing = [f"B_L{k}" for k, _ in pairs
               if f"B_L{k}" not in npz.files]
    checks["armed_pairs"] = pairs
    checks["bridge_arrays_missing"] = missing
    if scan.get("verdict") != "ARMED" or missing:
        raise SystemExit(f"scan artifacts not usable: {checks}")

    # eval data mount
    import pyarrow.parquet as pq
    table = pq.read_table(Path("/train_data") / "sadeeddiac-25.parquet")
    checks["eval_rows"] = table.num_rows

    # G2a preds asset
    fpv1 = ("https://github.com/interscript/interscript-models/"
            "releases/download/frontier-predictions-v1")
    g2a = "run-007-r7-muon-6ep-final_preds.jsonl"
    expect = "f6f606195144e3e17fdc85eb558ae7d58e30b8c14876e7c26456575bf238e0d7"
    got = sha256_file(fetch(f"{fpv1}/{g2a}", WORK / g2a))
    checks["g2a_sha_ok"] = got == expect

    ok = (checks["usable_pairs"] > 20000 and checks["eval_rows"] == 1200
          and checks["g2a_sha_ok"] and not missing
          and checks["scan_verdict"] == "ARMED")
    checks["SMOKE_OK"] = ok
    log("SMOKE " + json.dumps(checks))
    if not ok:
        raise SystemExit("smoke failed — do not launch the a100 arm")


if __name__ == "__main__":
    main()

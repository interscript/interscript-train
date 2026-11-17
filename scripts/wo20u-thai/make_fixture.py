"""WO20u fixture builder — slices a fixture from the mounted v5 corpus.

Runs INSIDE the job with the isx-training bucket mounted at /ckpt.
Writes fixture slices into the fixture run dir and prints counts +
sha256 of the real corpus files (provenance only — no corpus text).

The fixture exists to validate the full chain (mount rw, data load,
train step, epoch checkpoint save, eval harness, verdict writeback)
at trivial cost. It is NOT a quality signal.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SRC = Path("/ckpt") / "run-080-thai-lexicon-v5"
DST = Path("/ckpt") / "run-wo20u-thai-fixture"
TRAIN_ROWS = 400
DEV_ROWS = 80
FILES = ["tha-phonemic-v5-train.jsonl", "tha-phonemic-v5-dev.jsonl",
         "tha-phonemic-v5-test.jsonl"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slice_rows(name: str, n: int, out_name: str) -> int:
    rows = []
    with (SRC / name).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("word") or "").strip() and (r.get("phonemic") or "").strip():
                rows.append({"word": r["word"], "phonemic": r["phonemic"]})
            if len(rows) >= n:
                break
    with (DST / out_name).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = SRC / name
        print(f"[corpus] {name} rows-bytes={path.stat().st_size} "
              f"sha256={sha256(path)}", flush=True)
    n_train = slice_rows("tha-phonemic-v5-train.jsonl", TRAIN_ROWS,
                         "fixture-train.jsonl")
    n_dev = slice_rows("tha-phonemic-v5-dev.jsonl", DEV_ROWS,
                       "fixture-dev.jsonl")
    print(f"[fixture] train={n_train} dev={n_dev}", flush=True)


if __name__ == "__main__":
    main()

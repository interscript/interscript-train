"""WO36 run-050 — ivrit webdataset builder (driver).

Streams crowd-transcribe-v5 + audio-labeled parquets in deterministic
order, decodes embedded audio, writes ~500MB tar shards to
/ckpt/run-050-ivrit-wds/ until CAP_HOURS (default 200) is reached.
Extending the slice later = rerun with a higher cap (same shard order
prefix). Everything in-job.

Usage:
    python /ckpt/run-050-ivrit-wds/build_wds.py
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem

sys.path.insert(0, "/code/wds-core")

from ivrit_wds_core import build_sample, shard_stats, write_shard  # noqa: E402

OUT = Path("/ckpt") / os.environ.get("S_OUT", "run-050-ivrit-wds")
OUT.mkdir(parents=True, exist_ok=True)
TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=TOKEN)
fs = HfFileSystem(token=TOKEN)
CAP_HOURS = float(os.environ.get("CAP_HOURS", "200"))
SHARD_BYTES = 500 * 1024 * 1024

# pointed map from run-049 (uuid-keyed: text lives in the pair itself)
POINTED_JSONL = Path("/ckpt/run-049-ivrit-point/pairs-pointed.jsonl")


def load_pointed_by_uuid() -> dict:
    m = {}
    if POINTED_JSONL.exists():
        for line in POINTED_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("pointed"):
                    m[f"{r['repo']}-{r['uuid']}"] = r["pointed"]
    print(f"[pointed] by-uuid map={len(m)}", flush=True)
    return m


def main() -> None:
    import pyarrow.parquet as pq

    pointed = load_pointed_by_uuid()
    jobs = []
    for repo in ("crowd-transcribe-v5", "audio-labeled"):
        info = api.dataset_info(f"ivrit-ai/{repo}", files_metadata=True)
        for s in sorted(info.siblings, key=lambda x: x.rfilename):
            if s.rfilename.endswith(".parquet"):
                jobs.append((repo, f"datasets/ivrit-ai/{repo}/{s.rfilename}"))
    print(f"[plan] shards={len(jobs)} cap_hours={CAP_HOURS}", flush=True)

    hours = 0.0
    shard_idx = 0
    buf: list[dict] = []
    buf_bytes = 0
    done_marker = OUT / "BUILD_DONE"

    def flush():
        nonlocal shard_idx, buf, buf_bytes
        if not buf:
            return
        st = shard_stats(buf)
        p = OUT / f"shard-{shard_idx:04d}.tar"
        write_shard(p, buf)
        print(f"[shard] {p.name} {st} (total {hours:.2f}h)", flush=True)
        shard_idx += 1
        buf = []
        buf_bytes = 0

    for repo, path in jobs:
        if hours >= CAP_HOURS:
            break
        with fs.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    row["_repo"] = repo
                    s = build_sample(row, pointed)
                    if s is None:
                        continue
                    buf.append(s)
                    buf_bytes += len(s["wav"]) + 512
                    hours += s["dur"] / 3600
                if buf_bytes >= SHARD_BYTES or hours >= CAP_HOURS:
                    flush()
        print(f"[file] {path.rsplit('/', 1)[-1]} total={hours:.2f}h", flush=True)
    flush()

    st = {"shards": shard_idx, "hours": round(hours, 2), "cap": CAP_HOURS}
    (OUT / "build_report.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st), flush=True)
    done_marker.write_text("ok\n")


if __name__ == "__main__":
    main()

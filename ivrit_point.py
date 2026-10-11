"""WO36 stage 2 — extract + nikud-point the ivrit-ai paired corpora.

crowd-transcribe-v5 (~165K rows) + audio-labeled (~87K rows): both
embed audio bytes with verified transcripts — ready pairs, no VAD or
cutting. This job extracts the TEXT side only (audio stays in the
parquets for the teacher stage), points every transcript through our
crown nikud model (heb-diac-plane-2.0), and writes:

- /ckpt/run-049-ivrit-point/pairs-pointed.jsonl
    {repo, uuid, text, orig, pointed, split-ish dedupe}
- duration stats from a 200-audio sample (embedded bytes decoded via
  soundfile — cheap, no full pass)

Multiprocessed across parquet shards; everything in-job, zero local.
"""

from __future__ import annotations

import io
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem

OUT = Path("/ckpt/run-049-ivrit-point")
OUT.mkdir(parents=True, exist_ok=True)
TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=TOKEN)
# fs is built INSIDE each worker: a module-level HfFileSystem forked
# into the pool deadlocks on its parent's connection state (4h stall,
# zero shards; the standalone probe read the same shard in 1.4s)

_ZIP = Path("/ckpt/run-049-ivrit-point/heb-diac-plane-2.0.zip")


def _model():
    import sys

    sys.path.insert(0, "/code/ml-py")
    from interscript.ml.plane import PlaneModel

    return PlaneModel.from_zip(_ZIP.read_bytes())


def shard_paths(repo: str) -> list[str]:
    info = api.dataset_info(f"ivrit-ai/{repo}", files_metadata=True)
    return [f"datasets/ivrit-ai/{repo}/{s.rfilename}"
            for s in info.siblings if s.rfilename.endswith(".parquet")]


def process_shard(args) -> dict:
    idx, repo, path = args
    fs = HfFileSystem(token=TOKEN)  # fresh per worker — no forked session
    import pyarrow.parquet as pq

    model = _model()
    n = kept = 0
    tmp = OUT / f"part-{idx:03d}.jsonl"
    with fs.open(path, "rb") as f, tmp.open("w", encoding="utf-8") as out:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=64, columns=["uuid", "sentence", "orig_sentence", "text", "orig_text", "audio"]):
            cols = batch.to_pylist()
            for r in cols:
                n += 1
                text = (r.get("sentence") or r.get("text") or "").strip()
                orig = (r.get("orig_sentence") or r.get("orig_text") or "").strip()
                if not text:
                    continue
                try:
                    pointed = model.translate(text)
                except Exception:  # noqa: BLE001 — keep the row, flag it
                    pointed = ""
                kept += 1
                audio_len = None
                a = r.get("audio")
                if isinstance(a, dict) and a.get("bytes"):
                    audio_len = len(a["bytes"])
                elif isinstance(a, (bytes, bytearray)):
                    audio_len = len(a)
                out.write(json.dumps(
                    {"repo": repo, "uuid": r.get("uuid"), "text": text,
                     "orig": orig, "pointed": pointed, "audio_bytes": audio_len},
                    ensure_ascii=False) + "\n")
    return {"repo": repo, "path": path, "rows": n, "kept": kept,
            "part": tmp.name}


def main() -> None:
    import concurrent.futures as cf

    if not _ZIP.exists():
        import urllib.request

        urllib.request.urlretrieve(
            "https://github.com/interscript/interscript-models/releases/"
            "download/heb-diac-plane-2.0/heb-diac-plane-2.0.zip", _ZIP)
        import hashlib

        h = hashlib.sha256(_ZIP.read_bytes()).hexdigest()
        assert h == ("9c8f0432294ebae6abc847ddb79572c1ab979f4851f03d44a"
                     "9bb6495ba767a9e"), h
        print("[artifact] sha256 verified", flush=True)

    jobs = [(i, repo, p)
            for i, (repo, p) in enumerate(
                (repo, p) for repo in ("crowd-transcribe-v5", "audio-labeled")
                for p in shard_paths(repo))]
    print(f"[plan] shards={len(jobs)}", flush=True)

    stats = {"rows": 0, "kept": 0}
    # SPAWN, not fork: the parent touches HTTP (shard listing) and
    # forked children inherit poisoned module-global connection state
    # (diag: every component passes in a fresh process; pool hangs)
    import multiprocessing

    stats_ctx = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=8, mp_context=stats_ctx) as ex:
        for res in ex.map(process_shard, jobs):
            stats["rows"] += res["rows"]
            stats["kept"] += res["kept"]
            print(f"[shard] {res['path'].split('/')[-1]} rows={res['rows']} "
                  f"kept={res['kept']} (total {stats['kept']})", flush=True)

    # merge parts
    import glob

    merged = OUT / "pairs-pointed.jsonl"
    with merged.open("w", encoding="utf-8") as out:
        for p in sorted(glob.glob(str(OUT / "part-*.jsonl"))):
            for line in open(p, encoding="utf-8"):
                out.write(line)
            os.remove(p)

    # duration stats from audio_bytes (compressed proxy — report only)
    lens = []
    for i, line in enumerate(merged.open(encoding="utf-8")):
        if i >= 5000:
            break
        r = json.loads(line)
        if r.get("audio_bytes"):
            lens.append(r["audio_bytes"])
    if lens:
        avg_b = sum(lens) / len(lens)
        report = {"rows": stats["rows"], "kept": stats["kept"],
                  "audio_bytes_sample_avg": round(avg_b),
                  "note": "bytes proxy only; real hours measured at teacher stage"}
    else:
        report = stats
    (OUT / "point_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)
    (OUT / "POINT_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

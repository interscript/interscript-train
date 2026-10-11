"""WO36 stage 1 — ivrit-ai corpus audit. Everything in-job, zero local.

Deliverables to /ckpt/run-048-ivrit-audit/:
- audit.json: file/byte inventory, ffprobe-calibrated hour estimate,
  companion schemas + row counts, pairing-path recommendation
- transcripts_sample.json: 50 verified-transcript samples for QC
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

OUT = Path("/ckpt/run-048-ivrit-audit")
OUT.mkdir(parents=True, exist_ok=True)
TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=TOKEN)
fs = HfFileSystem(token=TOKEN)


def main() -> None:
    report: dict = {}

    # ---- audio-v2 inventory ----
    info = api.dataset_info("ivrit-ai/audio-v2", files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings if s.size]
    ext_bytes: Counter = Counter()
    ext_count: Counter = Counter()
    for name, size in files:
        ext = name.rsplit(".", 1)[-1].lower()
        ext_bytes[ext] += size
        ext_count[ext] += 1
    report["audio_v2"] = {
        "files": len(files),
        "bytes_total": sum(b for _, b in files),
        "ext_bytes": dict(ext_bytes.most_common(8)),
        "ext_count": dict(ext_count.most_common(8)),
    }
    print(f"[v2] files={len(files)} bytes={report['audio_v2']['bytes_total']/1e9:.1f}GB",
          flush=True)

    # ---- ffprobe calibration on small audio files (job scratch only) ----
    audio = sorted(((n, b) for n, b in files
                    if n.endswith((".m4a", ".mp3")) and b), key=lambda x: x[1])
    small = audio[:20] + audio[len(audio)//2:len(audio)//2 + 20] + audio[-20:]
    ratio_samples = []
    for name, size in small:
        try:
            p = hf_hub_download("ivrit-ai/audio-v2", name, token=TOKEN)
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", p],
                capture_output=True, text=True, check=True, timeout=120)
            dur = float(out.stdout.strip())
            if dur > 0 and size > 0:
                ratio_samples.append(dur / size)
            os.remove(p)
        except Exception as e:  # noqa: BLE001 — audit must survive bad files
            print(f"[probe-skip] {name}: {e}", flush=True)
    if ratio_samples:
        ratio = sum(ratio_samples) / len(ratio_samples)
        hours = report["audio_v2"]["bytes_total"] * ratio / 3600
        report["hours_estimate"] = round(hours, 1)
        report["probe_files"] = len(ratio_samples)
        print(f"[v2] hours≈{hours:.0f} (calibrated on {len(ratio_samples)} probes)",
              flush=True)

    # ---- companion schemas via parquet footers ----
    for repo in ("crowd-transcribe-v5", "audio-labeled", "audio-vad",
                 "audio-transcripts"):
        try:
            info = api.dataset_info(f"ivrit-ai/{repo}", files_metadata=True)
            pq_files = [s.rfilename for s in info.siblings
                        if s.rfilename.endswith((".parquet", ".arrow"))]
            entry = {"files": len(pq_files)}
            total_rows = 0
            schema = None
            checked = 0
            for name in pq_files[:6]:
                try:
                    with fs.open(f"datasets/ivrit-ai/{repo}/{name}", "rb") as f:
                        import pyarrow.parquet as pq

                        pf = pq.ParquetFile(f)
                        total_rows += pf.metadata.num_rows
                        if schema is None:
                            schema = [f.name for f in pf.schema_arrow]
                            batch = next(pf.iter_batches(batch_size=3))
                            entry["sample_rows"] = json.dumps(
                                batch.to_pylist(), ensure_ascii=False,
                                default=str)[:2000]
                    checked += 1
                except Exception as e:  # noqa: BLE001
                    entry.setdefault("errors", []).append(f"{name}: {e}")
            entry["rows_sampled_from"] = checked
            entry["schema"] = schema
            entry["rows_est_full"] = (total_rows * len(pq_files)) // max(1, checked)
            report[repo] = entry
            print(f"[{repo}] files={len(pq_files)} schema={schema}", flush=True)
        except Exception as e:  # noqa: BLE001
            report[repo] = {"error": str(e)}
            print(f"[{repo}] error {e}", flush=True)

    # ---- 50 transcript samples from crowd-transcribe-v5 ----
    try:
        import pyarrow.parquet as pq

        info = api.dataset_info("ivrit-ai/crowd-transcribe-v5",
                                files_metadata=True)
        pq_files = [s.rfilename for s in info.siblings
                    if s.rfilename.endswith(".parquet")]
        samples = []
        for name in pq_files:
            if len(samples) >= 50:
                break
            with fs.open(f"datasets/ivrit-ai/crowd-transcribe-v5/{name}", "rb") as f:
                pf = pq.ParquetFile(f)
                for batch in pf.iter_batches(batch_size=25):
                    samples.extend(batch.to_pylist())
                    if len(samples) >= 50:
                        break
        (OUT / "transcripts_sample.json").write_text(
            json.dumps(samples[:50], ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"[qc] {min(50, len(samples))} transcript samples written", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[qc] failed: {e}", flush=True)

    (OUT / "audit.json").write_text(json.dumps(report, indent=1),
                                    encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k != "audio_labeled"}, ensure_ascii=False)[:1500],
          flush=True)
    (OUT / "AUDIT_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

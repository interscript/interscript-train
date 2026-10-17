"""WO36 run-050 — ivrit webdataset builder (core, importable for tests).

Streams the self-contained pair repos (crowd-transcribe-v5,
audio-labeled), decodes embedded audio to 16-bit PCM WAV at native sr,
pairs by uuid with pointed text, writes tar shards of
{uuid}.wav + {uuid}.json to the bucket. Deterministic shard order;
CAP_HOURS bounds the slice; 1s <= dur <= 30s filter.
"""

from __future__ import annotations

import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

MIN_S = 1.0
MAX_S = 30.0


def decode_audio(raw_bytes: bytes):
    """bytes -> (pcm16 bytes, sr, duration_s). Raises on undecodable."""
    audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="int16",
                        always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1, keepdims=True).astype("int16")
    pcm = audio.tobytes()
    dur = audio.shape[0] / sr
    return pcm, sr, dur


def build_sample(row: dict, pointed_map: dict) -> dict | None:
    """One parquet row -> {uuid, wav, sr, dur, text, pointed} or None."""
    text = (row.get("sentence") or row.get("text") or "").strip()
    if not text:
        return None
    a = row.get("audio")
    raw = a.get("bytes") if isinstance(a, dict) else a
    if not raw:
        return None
    pcm, sr, dur = decode_audio(raw)
    if not (MIN_S <= dur <= MAX_S):
        return None
    uuid = f"{row.get('_repo', 'x')}-{row.get('uuid')}"
    return {
        "uuid": uuid,
        "wav": pcm,
        "sr": sr,
        "dur": round(dur, 3),
        "text": text,
        "orig": (row.get("orig_sentence") or row.get("orig_text") or "").strip(),
        "pointed": pointed_map.get(text, ""),
    }


def write_shard(tar_path: Path, samples: list[dict]) -> None:
    with tarfile.open(tar_path, "w") as tf:
        for s in samples:
            u = s["uuid"]
            meta = {k: s[k] for k in ("sr", "dur", "text", "orig", "pointed")}
            j = json.dumps(meta, ensure_ascii=False).encode("utf-8")
            _add(tf, f"{u}.json", j)
            _add(tf, f"{u}.wav", s["wav"])


def _add(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def shard_stats(samples: list[dict]) -> dict:
    hours = sum(s["dur"] for s in samples) / 3600
    srcs = Counter(s["uuid"].rsplit("-", 1)[0].split("-")[0] for s in samples)
    return {"samples": len(samples), "hours": round(hours, 4),
            "sources": dict(srcs)}

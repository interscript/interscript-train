"""WO40 — v3 teacher TRUE in-domain PER (fixed-eval re-measurement).

run-061's gate reported ivrit_dev_per 0.8722 through a broken eval
path (asr_phones ignored the records' 32000 Hz sample rate -> 2x
time-warp -> garbage decode). Training-path resampling was correct,
so the CHECKPOINTS are fine — only the number was garbage. This job
re-measures run-061/best with the fixed eval: ivrit dev 400
(last len//20 shards, same convention) + FLEURS dev 300.

Decision this feeds: if true ivrit PER is low (labels decent), the
e12 student is the bottleneck and label-filter/scale arms wait; if
high, teacher v4 on the 500h slice (WO36 stage 3) is mandatory.

Env: E_RUN (default run-061-ivrit-teacher-v3).

Usage (l4x1): python /ckpt/run-040-teacher-reeval/eval_heb_teacher_hf.py
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import torch

TEACHER = Path("/ckpt") / os.environ.get("E_RUN", "run-061-ivrit-teacher-v3") / "best"
WDS = Path("/ckpt/run-050-ivrit-wds")
OUT = Path("/ckpt") / os.environ.get("E_RUN", "run-061-ivrit-teacher-v3")
STRESS = "ˈˌ.-"


def norm_phones(ph: str) -> list[str]:
    for ch in STRESS:
        ph = ph.replace(ch, "")
    return ph.split()


def phone_lev(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    from transformers import AutoModelForCTC, AutoProcessor

    proc = AutoProcessor.from_pretrained(TEACHER)
    tok = proc.tokenizer
    tok.init_backend("he")
    model = AutoModelForCTC.from_pretrained(TEACHER).eval().cuda()
    unk = tok.unk_token_id

    @torch.no_grad()
    def asr_phones(r: dict) -> list[str]:
        sr = int(r.get("sr", 16000))
        wav = (np.frombuffer(r["wav"], dtype=np.int16)
               .astype(np.float32) / 32768.0)
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != 16000:
            import torchaudio  # noqa: PLC0415

            wt = torch.from_numpy(wav)
            wav = torchaudio.functional.resample(wt, sr, 16000).numpy()
        feats = proc.feature_extractor(wav, sampling_rate=16000,
                                       return_tensors="pt")
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(feats.input_values.cuda()).logits
        ids = logits.argmax(-1)[0].tolist()
        kept, seen = [], None
        for i in ids:
            if i != unk and i != seen:
                kept.append(i)
            seen = i
        return norm_phones(tok.decode(kept) if kept else "")

    def gold_of(text: str) -> list[str]:
        return norm_phones(" ".join(tok.convert_ids_to_tokens(
            tok(text, add_special_tokens=False).input_ids)))

    def per_of(pairs):
        pers = [phone_lev(p, g) / len(g) for p, g in pairs if g]
        return float(np.mean(pers)) if pers else 1.0

    # ivrit dev = last len//20 shards (the v3 convention)
    shards = sorted(WDS.glob("shard-*.tar"))
    dev_shards = shards[-max(1, len(shards) // 20):]
    dev = []
    for name in dev_shards:
        with tarfile.open(name) as tf:
            members = {m.name: m for m in tf.getmembers()}
            for m in members.values():
                if not m.name.endswith(".json"):
                    continue
                meta = json.load(tf.extractfile(m))
                wav_m = members.get(m.name[:-5] + ".wav")
                if wav_m is None:
                    continue
                dev.append({**meta, "wav": tf.extractfile(wav_m).read()})
    print(f"[dev] rows={len(dev)}", flush=True)

    pairs = [(asr_phones(r), gold_of(r["pointed"] if r.get("pointed")
                                    else r["text"])) for r in dev[:400]]
    ivrit_per = per_of(pairs)
    print(f"[per] ivrit_dev_true={ivrit_per:.4f}", flush=True)

    import csv

    from huggingface_hub import hf_hub_download

    tsv = hf_hub_download("google/fleurs", "data/he_il/dev.tsv",
                          repo_type="dataset")
    wanted = {}
    with open(tsv, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 3 and row[2].strip():
                name = row[1] if row[1].endswith(".wav") else row[1] + ".wav"
                wanted[name] = row[2].strip()
    tar = hf_hub_download("google/fleurs", "data/he_il/audio/dev.tar.gz",
                          repo_type="dataset")
    fleurs_pairs = []
    n = 0
    import tarfile as tf2

    with tf2.open(tar, "r:gz") as t:
        for m in t.getmembers():
            base = m.name.split("/")[-1]
            if base not in wanted or n >= 300:
                continue
            wav, sr = __import__("soundfile").read(
                io.BytesIO(t.extractfile(m).read()), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(1)
            feats = proc.feature_extractor(wav, sampling_rate=sr,
                                           return_tensors="pt")
            with torch.autocast("cuda", torch.bfloat16):
                logits = model(feats.input_values.cuda()).logits
            ids = logits.argmax(-1)[0].tolist()
            kept, seen = [], None
            for i in ids:
                if i != unk and i != seen:
                    kept.append(i)
                seen = i
            fleurs_pairs.append((norm_phones(tok.decode(kept) if kept else ""),
                                 gold_of(wanted[base])))
            n += 1
    fleurs_per = per_of(fleurs_pairs)

    verdict = {"run": TEACHER.parent.name,
               "ivrit_dev_per_true": round(ivrit_per, 4),
               "fleurs_dev_per": round(fleurs_per, 4),
               "note": "fixed-eval re-measurement (WO40); the 0.8722 "
                       "previously reported was the sr-ignored eval bug",
               "prior_buggy_report": 0.8722,
               "decision_rule": "ivrit<0.35: student-bottleneck; "
                                ">=0.35: teacher v4 @500h mandatory"}
    print(json.dumps(verdict), flush=True)
    (OUT / "teacher_reeval.json").write_text(json.dumps(verdict, indent=1))
    (OUT / "REEVAL_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

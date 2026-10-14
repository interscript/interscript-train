"""WO36 run-051 — Hebrew phoneme-ASR teacher v2 at ivrit scale.

Fine-tunes facebook/wav2vec2-lv-60-espeak-cv-ft on the run-050
webdataset slice (pointed-espeak targets via the tokenizer's he
backend — the WO31-conventions fix at scale). Gates:
- FLEURS he_il dev per-phone PER < 0.25 (cross-domain; run-038 scored
  0.319 from 10h, run-041 0.319 pointed — this is the comparability
  anchor)
- ivrit held-out shard PER reported (in-domain)

Usage (apt espeak-ng):
    python /ckpt/run-051-ivrit-teacher/train_ivrit_teacher_hf.py
"""

from __future__ import annotations

import io
import json
import os
import random
import tarfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download

CKPT = Path("/ckpt") / os.environ.get("T_RUN", "run-051-ivrit-teacher")
WDS = Path("/ckpt/run-050-ivrit-wds")
EPOCHS = int(os.environ.get("T_EPOCHS", "3"))
BS = int(os.environ.get("T_BS", "12"))
LR = float(os.environ.get("T_LR", "5e-5"))
MAX_HOURS = float(os.environ.get("T_MAX_HOURS", "500"))
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


def load_slice() -> tuple[list[dict], list[dict]]:
    shards = sorted(WDS.glob("shard-*.tar"))
    if not shards:
        raise SystemExit("no webdataset shards — run run-050 first")
    dev_shards = shards[-max(1, len(shards) // 20):]
    train_shards = shards[:-len(dev_shards)]
    train, dev = [], []
    for group, names in (("train", train_shards), ("dev", dev_shards)):
        for name in names:
            with tarfile.open(name) as tf:
                members = {m.name: m for m in tf.getmembers()}
                for m in list(tf.getmembers()):
                    if not m.name.endswith(".json"):
                        continue
                    meta = json.load(tf.extractfile(m))
                    wav_m = members.get(m.name[:-5] + ".wav")
                    if wav_m is None:
                        continue
                    rec = {"uuid": m.name[:-5], **meta,
                           "wav": tf.extractfile(wav_m).read()}
                    # targets = crown-nikud vocalization (WO31 convention:
                    # pointed text -> espeak -> vowel-full phonemes)
                    if rec.get("pointed"):
                        rec["text"] = rec["pointed"]
                    (train if group == "train" else dev).append(rec)
        if group == "train":
            hours = sum(r["dur"] for r in train) / 3600
            if hours > MAX_HOURS:
                rng = random.Random(42)
                rng.shuffle(train)
                keep, acc = [], 0.0
                for r in train:
                    if acc >= MAX_HOURS:
                        break
                    keep.append(r)
                    acc += r["dur"] / 3600
                train = keep
    random.Random(42).shuffle(train)
    th = sum(r["dur"] for r in train) / 3600
    dh = sum(r["dur"] for r in dev) / 3600
    print(f"[slice] train={len(train)} ({th:.1f}h) dev={len(dev)} ({dh:.1f}h)",
          flush=True)
    return train, dev


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCTC, AutoProcessor

    proc = AutoProcessor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    tok = proc.tokenizer
    tok.init_backend("he")
    model = AutoModelForCTC.from_pretrained(
        "facebook/wav2vec2-lv-60-espeak-cv-ft").cuda().train()
    model.freeze_feature_encoder()

    # resume from the latest step ckpt if present (silent infra kills);
    # step dirs are save_pretrained outputs -> load via from_pretrained
    steps = sorted(CKPT.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    if steps:
        model = AutoModelForCTC.from_pretrained(steps[-1]).cuda().train()
        model.freeze_feature_encoder()
        proc = AutoProcessor.from_pretrained(steps[-1])
        tok = proc.tokenizer
        print(f"[resume] {steps[-1].name}", flush=True)

    train, dev = load_slice()

    unk_total = tok_total = 0
    for r in train[:2000]:
        ids = tok(r["text"], add_special_tokens=False).input_ids
        unk_total += sum(1 for i in ids if i == tok.unk_token_id)
        tok_total += len(ids)
    unk_rate = unk_total / max(1, tok_total)
    print(f"[targets] unk_rate={unk_rate:.4f}", flush=True)
    if unk_rate > 0.05:
        raise SystemExit("pointed text phonemization off-vocab — aborting")

    def pcm_to_16k(r: dict) -> np.ndarray:
        # shards carry RAW int16 PCM (no RIFF header) + sr in json
        import torchaudio  # noqa: PLC0415

        t = torch.from_numpy(
            np.frombuffer(r["wav"], dtype=np.int16).astype(np.float32) / 32768.0)
        sr = int(r["sr"])
        if sr != 16000:
            t = torchaudio.functional.resample(t, sr, 16000)
        return t.numpy()

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(train)

        def __getitem__(self, i):
            r = train[i]
            return pcm_to_16k(r).astype(np.float32), r["text"]

    def collate(b):
        feats = proc.feature_extractor([w for w, _ in b], sampling_rate=16000,
                                       return_tensors="pt", padding=True)
        texts = [t for _, t in b]
        lb = tok(texts, add_special_tokens=False, padding=True,
                 return_tensors="pt", return_length=True)
        labels = torch.full_like(lb.input_ids, -100)
        for i, ids in enumerate(lb.input_ids):
            n = int(lb.length[i])
            labels[i, :n] = ids[:n]
        return feats.input_values, getattr(feats, "attention_mask", None), labels

    loader = torch.utils.data.DataLoader(
        DS(), batch_size=BS, shuffle=True, collate_fn=collate,
        num_workers=4, drop_last=True)
    total_steps = len(loader) * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps)

    step = 0
    t0 = time.time()
    for _ep in range(EPOCHS):
        for iv, am, labels in loader:
            iv, labels = iv.cuda(), labels.cuda()
            kwargs = {"attention_mask": am.cuda()} if am is not None else {}
            with torch.autocast("cuda", torch.bfloat16):
                out = model(input_values=iv, labels=labels, **kwargs)
            if not torch.isfinite(out.loss):
                print(f"[nan] step {step}", flush=True)
                opt.zero_grad()
                continue
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f"[train] {step}/{total_steps} ctc={float(out.loss):.1f} "
                      f"({step/(time.time()-t0):.2f} it/s)", flush=True)
            if step % 2000 == 0:
                d = CKPT / f"step-{step}"
                d.mkdir(exist_ok=True)
                model.save_pretrained(d)
                proc.save_pretrained(d)

    best = CKPT / "best"
    best.mkdir(exist_ok=True)
    model.save_pretrained(best)
    proc.save_pretrained(best)

    # ---- gates ----
    model.eval()
    unk = tok.unk_token_id

    @torch.no_grad()
    def asr_phones(wav, sr: int = 16000) -> list[str]:
        if isinstance(wav, dict):  # {wav: raw pcm bytes, sr}
            sr = int(wav.get("sr", 16000))
            wav = (np.frombuffer(wav["wav"], dtype=np.int16)
                   .astype(np.float32) / 32768.0)
        elif isinstance(wav, (bytes, bytearray)):
            wav = np.frombuffer(wav, dtype=np.int16).astype(np.float32) / 32768.0
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

    def per_of(pairs):
        pers = []
        for pred, gold in pairs:
            if gold:
                pers.append(phone_lev(pred, gold) / len(gold))
        return float(np.mean(pers)) if pers else 1.0

    ivrit_pairs = [(asr_phones(r),
                    norm_phones(" ".join(tok.convert_ids_to_tokens(
                        tok(r["text"], add_special_tokens=False).input_ids))))
                   for r in dev[:400]]
    ivrit_per = per_of(ivrit_pairs)

    fleurs_pairs = []
    for split in ("dev",):
        import csv

        tsv = hf_hub_download("google/fleurs", f"data/he_il/{split}.tsv",
                              repo_type="dataset")
        import tarfile

        with open(tsv, encoding="utf-8") as f:
            wanted = {}
            for row in csv.reader(f, delimiter="\t"):
                if len(row) >= 3 and row[2].strip():
                    name = row[1] if row[1].endswith(".wav") else row[1] + ".wav"
                    wanted[name] = row[2].strip()
        tar = hf_hub_download("google/fleurs", f"data/he_il/audio/{split}.tar.gz",
                              repo_type="dataset")
        n = 0
        with tarfile.open(tar, "r:gz") as tf:
            for m in tf.getmembers():
                base = m.name.split("/")[-1]
                if base not in wanted or n >= 300:
                    continue
                wav, sr = sf.read(io.BytesIO(tf.extractfile(m).read()),
                                  dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(1)
                pred = asr_phones(wav)
                gold = norm_phones(" ".join(tok.convert_ids_to_tokens(
                    tok(wanted[base], add_special_tokens=False).input_ids)))
                fleurs_pairs.append((pred, gold))
                n += 1
    fleurs_per = per_of(fleurs_pairs)

    verdict = {"run": CKPT.name, "fleurs_dev_per": round(fleurs_per, 4),
               "ivrit_dev_per": round(ivrit_per, 4),
               "note": ("pointed-convention model: FLEURS PER is "
                        "convention-shifted (skeleton golds); primary "
                        "gate = ivrit dev, same convention as training"),
               "gate": "ivrit_dev_per < 0.30 (primary); fleurs reported",
               "baseline_run038": 0.319, "utts_ivrit": len(ivrit_pairs),
               "utts_fleurs": len(fleurs_pairs), "epochs": EPOCHS,
               "slice_hours": round(sum(r["dur"] for r in train) / 3600, 1)}
    (CKPT / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict), flush=True)
    (CKPT / "TEACHER_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

"""WO36 run-052 — ivrit labeler + student v3 + gate, one resumable job.

Phase 1 (resumable via LABEL_DONE): decode the webdataset slice with
the tuned teacher (run-051 best) -> audio-derived vowel-full labels,
written to labels.jsonl {uuid, text, ipa}.
Phase 2: byt5-BASE student text->IPA on the labels (byte-correct
collate — the WO31 lesson), gate: normalized CER < 0.2393 (rules
layer) on the phonikud heb-g2p benchmark (gt.tsv from run-032).
Ladder: Nakdimon 0.1102 · phonikud 0.0495 · ReNikud 0.0244.

Usage (apt espeak-ng):
    python /ckpt/run-052-ivrit-student/ivrit_student_hf.py
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

CKPT = Path("/ckpt") / os.environ.get("S_RUN", "run-052-ivrit-student")
WDS = Path("/ckpt/run-050-ivrit-wds")
TEACHER = Path(os.environ.get("S_TEACHER", "/ckpt/run-051-ivrit-teacher/best"))
GT = Path("/ckpt/run-032-heb-ipa-v0/gt.tsv")
EPOCHS = int(os.environ.get("S_EPOCHS", "4"))
BS = int(os.environ.get("S_BS", "16"))
BACKBONE = os.environ.get("S_BACKBONE", "google/byt5-base")
STRESS = "ˈˌ.-"


def norm_target(ph: str) -> str:
    for ch in STRESS:
        ph = ph.replace(ch, "")
    return " ".join(ph.split())


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def build_labels() -> list[dict]:
    labels_path = CKPT / "labels.jsonl"
    if labels_path.exists():
        rows = [json.loads(l) for l in labels_path.read_text(
            encoding="utf-8").splitlines() if l.strip()]
        print(f"[labels] resumed {len(rows)}", flush=True)
        return rows

    from transformers import AutoModelForCTC, AutoProcessor

    proc = AutoProcessor.from_pretrained(TEACHER)
    tok = proc.tokenizer
    asr = AutoModelForCTC.from_pretrained(TEACHER).eval().cuda()
    unk = tok.unk_token_id

    shards = sorted(WDS.glob("shard-*.tar"))
    rows = []
    t0 = time.time()
    with labels_path.open("w", encoding="utf-8") as out:
        for si, name in enumerate(shards):
            with tarfile.open(name) as tf:
                members = {m.name: m for m in tf.getmembers()}
                for m in members:
                    if not m.name.endswith(".json"):
                        continue
                    meta = json.load(tf.extractfile(m))
                    wav_m = members.get(m.name[:-5] + ".wav")
                    if wav_m is None:
                        continue
                    wav, sr = sf.read(io.BytesIO(tf.extractfile(wav_m).read()),
                                      dtype="float32")
                    if wav.ndim > 1:
                        wav = wav.mean(1)
                    feats = proc.feature_extractor(
                        wav, sampling_rate=sr, return_tensors="pt")
                    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                        logits = asr(feats.input_values.cuda()).logits
                    ids = logits.argmax(-1)[0].tolist()
                    kept, seen = [], None
                    for i in ids:
                        if i != unk and i != seen:
                            kept.append(i)
                        seen = i
                    ipa = norm_target(tok.decode(kept) if kept else "")
                    if len(ipa) < 4:
                        continue
                    rows.append({"uuid": m.name[:-5], "text": meta["text"],
                                 "ipa": ipa})
                    out.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
            print(f"[labels] shard {si+1}/{len(shards)} total={len(rows)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    (CKPT / "LABEL_DONE").write_text("ok\n")
    return rows


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    src = os.environ.get("LABELS_SRC")
    if src:
        rows = [json.loads(l) for l in Path(src).read_text(
            encoding="utf-8").splitlines() if l.strip()]
        print(f"[labels] external source {src}: {len(rows)} rows", flush=True)
    else:
        rows = build_labels()
    random.Random(42).shuffle(rows)
    train = rows[: int(len(rows) * 0.98)]
    print(f"[data] pairs={len(rows)} train={len(train)}", flush=True)

    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)

    tok = AutoTokenizer.from_pretrained(BACKBONE)
    model = AutoModelForSeq2SeqLM.from_pretrained(BACKBONE).cuda().train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    class DS(Dataset):
        def __len__(self):
            return len(train)

        def __getitem__(self, i):
            return (train[i]["text"], train[i]["ipa"])

    def collate(b):
        # ByT5 tokens are BYTES: Hebrew chars are multi-byte — pad to
        # tokenized lengths, never char lengths
        es = [tok(t).input_ids[:-1] for t, _ in b]
        ds_ = [tok(p).input_ids for _, p in b]
        mx = max(len(e) for e in es)
        md = max(len(d) for d in ds_)
        ids = torch.zeros((len(b), mx), dtype=torch.long)
        am = torch.zeros((len(b), mx), dtype=torch.long)
        lb = torch.full((len(b), md), -100, dtype=torch.long)
        for i, (e, d) in enumerate(zip(es, ds_)):
            ids[i, :len(e)] = torch.tensor(e)
            am[i, :len(e)] = 1
            lb[i, :len(d)] = torch.tensor(d)
        return ids, am, lb

    loader = DataLoader(DS(), batch_size=BS, shuffle=True, collate_fn=collate,
                        num_workers=2, drop_last=True)
    total_steps = len(loader) * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, 200, total_steps)

    step = 0
    t0 = time.time()
    for _ep in range(EPOCHS):
        for ids, am, lb in loader:
            ids, am, lb = ids.cuda(), am.cuda(), lb.cuda()
            with torch.autocast("cuda", torch.bfloat16):
                out = model(input_ids=ids, attention_mask=am, labels=lb)
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
            if step % 200 == 0:
                print(f"[train] {step}/{total_steps} loss={float(out.loss):.3f} "
                      f"({step/(time.time()-t0):.1f} it/s)", flush=True)

    model.config.use_cache = True
    model.eval()
    best = CKPT / "best"
    best.mkdir(exist_ok=True)
    model.save_pretrained(best)
    tok.save_pretrained(best)

    # ---- gate: phonikud heb-g2p-benchmark, normalized CER ----
    gt = []
    for line in GT.read_text(encoding="utf-8").splitlines():
        s, _, p = line.partition("\t")
        if s and p:
            gt.append((s, p))
    print(f"[gate] gt sentences={len(gt)}", flush=True)

    def _norm_ipa(s: str) -> str:
        for ch in STRESS:
            s = s.replace(ch, "")
        return s.replace(" ", "")

    preds = []
    with torch.no_grad():
        for i in range(0, len(gt), 16):
            batch = [s for s, _ in gt[i:i+16]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to("cuda")
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=256, num_beams=1)
            preds.extend(tok.batch_decode(gen, skip_special_tokens=True))

    cers, cers_norm = [], []
    for (s, gold), pred in zip(gt, preds):
        ng, np_ = _norm_ipa(gold), _norm_ipa(pred)
        cers.append(lev(pred, gold) / max(1, len(gold)))
        cers_norm.append(lev(np_, ng) / max(1, len(ng)))
    verdict = {"model": "heb-ipa-ivrit-v3", "teacher": str(TEACHER),
               "backbone": BACKBONE, "pairs": len(rows),
               "cer_raw": round(sum(cers) / len(cers), 4),
               "cer_normalized": round(sum(cers_norm) / len(cers_norm), 4),
               "gate": "cer_normalized < 0.2393 (rules layer)",
               "rules_baseline": 0.2393,
               "board": {"renikud": 0.0244, "phonikud": 0.0495,
                         "nakdimon": 0.1102}}
    (CKPT / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict), flush=True)
    for (s, gold), pred in list(zip(gt, preds))[:3]:
        print(f"  src: {s[:40]}\n  pred: {pred[:60]}\n  gold: {gold[:60]}",
              flush=True)
    (CKPT / "V3_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

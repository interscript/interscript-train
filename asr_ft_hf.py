"""WO31 stage 1 — Hebrew-tuned phoneme ASR (CTC fine-tune).

WO30 v0 verdict: the universal espeak-cv-ft phonemizer hears Hebrew at
PER 0.395 — the teacher, not the student, is the bottleneck. Fix: the
ReNikud first stage — fine-tune facebook/wav2vec2-lv-60-espeak-cv-ft
on FLEURS he_il audio with espeak-ng phonemizations of the TRANSCRIPTS
as CTC targets (espeak reads the text; the CTC learns to hear it).

Gate: dev PER < 0.395 (expect well under). Per-PHONE scoring (the v0
per-word lesson). Run on an apt-enabled image:

    apt-get update && apt-get install -y espeak-ng
    python /ckpt/run-038-heb-asr-ft/asr_ft_hf.py
"""

from __future__ import annotations

import csv
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

CKPT = Path("/ckpt") / os.environ.get("ASR_RUN", "run-038-heb-asr-ft")
POINTED = os.environ.get("ASR_POINTED", "")  # optional text->pointed map (WO31b)
EPOCHS = int(os.environ.get("ASR_EPOCHS", "30"))
BS = int(os.environ.get("ASR_BS", "8"))
LR = float(os.environ.get("ASR_LR", "5e-5"))
STRESS = "ˈˌ.-"


def norm_phones(ph: str) -> list[str]:
    for ch in STRESS:
        ph = ph.replace(ch, "")
    return ph.split()


def build_corpus() -> list[dict]:
    rows = []
    for split in ("train", "dev", "test"):
        tsv = hf_hub_download("google/fleurs", f"data/he_il/{split}.tsv",
                              repo_type="dataset")
        with open(tsv, encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) >= 3:
                    rows.append((split, row))
    wanted: dict[str, dict[str, str]] = {}
    for split, row in rows:
        name = row[1] if row[1].endswith(".wav") else row[1] + ".wav"
        wanted.setdefault(split, {})[name] = row[2].strip()

    corpus = []
    for split in ("train", "dev", "test"):
        tar = hf_hub_download("google/fleurs", f"data/he_il/audio/{split}.tar.gz",
                              repo_type="dataset")
        with tarfile.open(tar, "r:gz") as tf:
            for m in tf.getmembers():
                base = m.name.split("/")[-1]
                if base not in wanted[split]:
                    continue
                f = tf.extractfile(m)
                if not f:
                    continue
                text = wanted[split][base]
                if not text:
                    continue
                speech, sr = sf.read(io.BytesIO(f.read()))
                if speech.ndim > 1:
                    speech = speech.mean(axis=1)
                if len(speech) < 16000 or sr != 16000:
                    continue
                corpus.append({"text": text, "wav": speech.astype(np.float32),
                               "split": split})
    random.Random(42).shuffle(corpus)
    return corpus


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCTC, AutoProcessor

    proc = AutoProcessor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    tok = proc.tokenizer
    # phonemize Hebrew through the tokenizer's own espeak backend —
    # its separator conventions are the ones the vocab was built with
    # (verified locally: unk rate 0.0000; manual espeak --ipa emits
    # contiguous phone strings that tokenize as one unk per word)
    tok.init_backend("he")
    model = AutoModelForCTC.from_pretrained(
        "facebook/wav2vec2-lv-60-espeak-cv-ft").cuda().train()
    model.freeze_feature_encoder()

    corpus = build_corpus()
    print(f"[data] utterances={len(corpus)}", flush=True)

    pointed = {}
    if POINTED:
        for line in Path(POINTED).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                pointed[r["text"]] = r["pointed"]
        print(f"[pointed] map={len(pointed)}", flush=True)

    labeled = []
    unk_total = tok_total = 0
    for r in corpus:
        text = pointed.get(r["text"], r["text"])
        if not text.strip():
            continue
        ids = tok(text, add_special_tokens=False).input_ids
        unk_total += sum(1 for i in ids if i == tok.unk_token_id)
        tok_total += len(ids)
        if not ids:
            continue
        r["labels"] = ids
        labeled.append(r)
    unk_rate = unk_total / max(1, tok_total)
    print(f"[targets] utts={len(labeled)} unk_rate={unk_rate:.4f}", flush=True)
    if unk_rate > 0.05:
        raise SystemExit("espeak alphabet not covered by tokenizer — aborting")

    train = [r for r in labeled if r["split"] == "train"]
    dev = [r for r in labeled if r["split"] == "dev"]
    print(f"[split] train={len(train)} dev={len(dev)}", flush=True)

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(train)

        def __getitem__(self, i):
            return train[i]["wav"], train[i]["labels"]

    def collate(b):
        feats = proc.feature_extractor(
            [w for w, _ in b], sampling_rate=16000, return_tensors="pt",
            padding=True)
        lb_max = max(len(y) for _, y in b)
        labels = torch.full((len(b), lb_max), -100, dtype=torch.long)
        for i, (_, y) in enumerate(b):
            labels[i, :len(y)] = torch.tensor(y)
        return feats.input_values, getattr(feats, "attention_mask", None), labels

    loader = torch.utils.data.DataLoader(
        DS(), batch_size=BS, shuffle=True, collate_fn=collate,
        num_workers=0, drop_last=True)
    total_steps = len(loader) * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps)

    step = 0
    t0 = time.time()
    for _ep in range(EPOCHS):
        for iv, am, labels in loader:
            iv, labels = iv.cuda(), labels.cuda()
            kwargs = {}
            if am is not None:
                kwargs["attention_mask"] = am.cuda()
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
                print(f"[train] {step}/{total_steps} ctc={float(out.loss):.3f} "
                      f"({step/(time.time()-t0):.1f} it/s)", flush=True)
            if step % 2000 == 0:
                d = CKPT / f"step-{step}"
                d.mkdir(exist_ok=True)
                model.save_pretrained(d)
                proc.save_pretrained(d)

    best = CKPT / "best"
    best.mkdir(exist_ok=True)
    model.save_pretrained(best)
    proc.save_pretrained(best)

    # ---- gate: dev PER, per-phone (the v0 lesson) ----
    model.eval()
    unk = tok.unk_token_id

    def phone_lev(a: list[str], b: list[str]) -> int:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                               prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    @torch.no_grad()
    def asr_phones(wav: np.ndarray) -> list[str]:
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

    pers = []
    for r in dev:
        pred = asr_phones(r["wav"])
        gold = norm_phones(" ".join(
            tok.convert_ids_to_tokens([i for i in r["labels"] if i != -100])))
        if not gold:
            continue
        pers.append(phone_lev(pred, gold) / len(gold))
    per = float(np.mean(pers))
    verdict = {"run": CKPT.name, "dev_per": round(per, 4),
               "baseline_universal_per": 0.395, "gate": "per < 0.395",
               "utts": len(dev), "epochs": EPOCHS}
    (CKPT / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict), flush=True)
    (CKPT / "ASR_FT_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

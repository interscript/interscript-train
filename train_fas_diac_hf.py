"""WO47 retrain — fas-diac: byt5-small on the recovered 528K-pair Persian corpus.

The WO38 pattern (run-059: pretrained byt5-base, 3 epochs, full-split
gate) applied to the diacritization task — urd-diac-1.0 is small-tier
(1.42GB fp32 zip, 0.0377 full-split CER) and the base tier is untried.

Gate: full 11,940-row test split RAW char CER < 0.0377 (urd-diac-1.0,
the WO37 protocol). Epoch checkpoints + resume (the e12 lesson).

Env: U_RUN (default run-066-urd-diac-base), U_BACKBONE
(google/byt5-base), U_EPOCHS (3), U_BS (16), U_ACCUM (2).

Usage (a100-large):
    -v hf://buckets/rhtse/jobs-artifacts/rababa-urdu-cc174b96:/data-dia:ro
    python /ckpt/run-066-urd-diac-base/train_urd_diac_base_hf.py
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

DATA = Path("/ckpt") / os.environ.get("U_RUN", "run-076-fas-diac-retrain")
CKPT = Path("/ckpt") / os.environ.get("U_RUN", "run-066-urd-diac-base")
BACKBONE = os.environ.get("U_BACKBONE", "google/byt5-small")
EPOCHS = int(os.environ.get("U_EPOCHS", "3"))
BS = int(os.environ.get("U_BS", "16"))
ACCUM = int(os.environ.get("U_ACCUM", "2"))
MAXLEN = 256
GATE = float(os.environ.get("U_GATE", "0.0052"))  # docs reference


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)

    def load(name):
        rows = [json.loads(l) for l in (DATA / name).read_text(
            encoding="utf-8").splitlines() if l.strip()]
        return [(r["src"].strip(), r["tgt"].strip()) for r in rows
                if r.get("src", "").strip() and r.get("tgt", "").strip()]

    train = load("train.jsonl")
    test = load("test.jsonl")
    random.Random(42).shuffle(train)
    print(f"[data] train={len(train)} test={len(test)}", flush=True)

    tok = AutoTokenizer.from_pretrained(BACKBONE)
    model = AutoModelForSeq2SeqLM.from_pretrained(BACKBONE).cuda().train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    eps = sorted(CKPT.glob("epoch-*"), key=lambda p: int(p.name.split("-")[1]))
    start_ep = 0
    if eps:
        model = AutoModelForSeq2SeqLM.from_pretrained(eps[-1]).cuda().train()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        start_ep = int(eps[-1].name.split("-")[1]) + 1
        print(f"[resume] {eps[-1].name} (epoch {start_ep})", flush=True)

    class DS(Dataset):
        def __len__(self):
            return len(train)

        def __getitem__(self, i):
            return train[i]

    def collate(b):
        es = [tok(s, truncation=True, max_length=MAXLEN).input_ids
              for s, _ in b]
        ds_ = [tok(t, truncation=True, max_length=MAXLEN).input_ids
               for _, t in b]
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
    total_steps = (len(loader) // ACCUM) * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, 200, total_steps)

    step = start_ep * (len(loader) // ACCUM)
    t0 = time.time()
    for _ep in range(start_ep, EPOCHS):
        for ids, am, lb in loader:
            ids, am, lb = ids.cuda(), am.cuda(), lb.cuda()
            with torch.autocast("cuda", torch.bfloat16):
                out = model(input_ids=ids, attention_mask=am, labels=lb)
            if not torch.isfinite(out.loss):
                print(f"[nan] step {step}", flush=True)
                opt.zero_grad()
                continue
            (out.loss / ACCUM).backward()
            if step % ACCUM == ACCUM - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"[train] {step}/{total_steps} loss={float(out.loss):.3f} "
                      f"({step/(time.time()-t0):.1f} it/s)", flush=True)
        ep_dir = CKPT / f"epoch-{_ep}"
        ep_dir.mkdir(exist_ok=True)
        model.save_pretrained(ep_dir)
        tok.save_pretrained(ep_dir)
        print(f"[ckpt] epoch-{_ep} saved", flush=True)

    model.config.use_cache = True
    model.eval()
    best = CKPT / "best"
    best.mkdir(exist_ok=True)
    model.save_pretrained(best)
    tok.save_pretrained(best)

    # gate: full test split, raw char CER (the WO37 protocol)
    preds = []
    with torch.no_grad():
        for i in range(0, len(test), 16):
            batch = [s for s, _ in test[i:i + 16]]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=MAXLEN).to("cuda")
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=MAXLEN, num_beams=1)
            preds.extend(tok.batch_decode(gen, skip_special_tokens=True))
    cers, correct = [], 0
    for (s, gold), pred in zip(test, preds):
        cers.append(lev(pred, gold) / max(1, len(gold)))
        if pred == gold:
            correct += 1
    cer = sum(cers) / len(cers)
    res = {"model": BACKBONE, "pairs": len(train),
           "cer": round(cer, 4), "word_acc": round(correct / len(test), 4),
           "gate": f"cer < {GATE} (urd-diac-1.0 full split)",
           "baseline": GATE,
           "verdict": "PASS" if cer < GATE else "MISS"}
    print(json.dumps(res), flush=True)
    (CKPT / "verdict.json").write_text(json.dumps(res, indent=1))
    (CKPT / "GATE_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

"""WO88 — khm-latn retrain under the framework (the secryst-lineage corpus).

ByT5-small SFT on the Royal-Institute-derived Thai→phonemic lexicon
(run-080-thai-lexicon: 28,644 train / 3,225 dev, pinned in
interscript-models corpora/thai-lexicon.yaml). This is the model
fallback for novel tokens behind the deterministic RTGS composite
(maps#204) — the lexicon hit-rate itself is measured upstream.

Resume + epoch checkpoints into the mounted bucket path (the
run-080-r8-stage2 pattern: oepoch-N dirs land remotely while the job
runs). Dev gate: raw char CER on tha-phonemic-test.jsonl.

Env: T_RUN (default run-080-thai-lex), T_EPOCHS (8), T_BS (32),
T_ACCUM (1), T_BACKBONE (google/byt5-small).

Usage (a100-large):
    -v hf://buckets/Interscript/isx-training:/ckpt:rw
    python /ckpt/run-080-thai-lex/train_thai_lexicon_hf.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

DATA = Path("/ckpt") / os.environ.get("T_DATA", "run-088-khm-latn")
TRAIN_FILE = os.environ.get("T_TRAIN", "khm-latn-train.jsonl")
DEV_FILE = os.environ.get("T_DEV", "khm-latn-test.jsonl")
CKPT = Path("/ckpt") / os.environ.get("T_RUN", "run-088-khm-latn")
BACKBONE = os.environ.get("T_BACKBONE", "google/byt5-small")
EPOCHS = int(os.environ.get("T_EPOCHS", "8"))
BS = int(os.environ.get("T_BS", "32"))
ACCUM = int(os.environ.get("T_ACCUM", "1"))
MAXLEN = 256


def load(name: str) -> list[tuple[str, str]]:
    rows = [json.loads(l) for l in (DATA / name).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    return [(r["src"].strip(), r["tgt"].strip()) for r in rows
            if (r.get("src") or "").strip() and (r.get("tgt") or "").strip()]


class Pairs(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], tok):
        self.pairs = pairs
        self.tok = tok

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        src, tgt = self.pairs[i]
        enc = self.tok(src, truncation=True, max_length=MAXLEN,
                       return_tensors="pt")
        with self.tok.as_target_tokenizer():
            dec = self.tok(tgt, truncation=True, max_length=MAXLEN,
                           return_tensors="pt")
        return (enc.input_ids[0], enc.attention_mask[0], dec.input_ids[0])


def collate(batch, pad):
    ids = torch.nn.utils.rnn.pad_sequence([b[0] for b in batch],
                                          batch_first=True, padding_value=pad)
    att = torch.nn.utils.rnn.pad_sequence([b[1] for b in batch],
                                          batch_first=True, padding_value=0)
    lab = torch.nn.utils.rnn.pad_sequence([b[2] for b in batch],
                                          batch_first=True, padding_value=-100)
    return ids, att, lab


@torch.no_grad()
def dev_cer(model, tok, pairs) -> float:
    model.eval()
    total_c, total_n = 0, 0
    for src, tgt in pairs:
        ids = tok(src, return_tensors="pt", truncation=True,
                  max_length=MAXLEN).input_ids.to(model.device)
        out = model.generate(ids, max_new_tokens=MAXLEN, num_beams=1)
        pred = tok.decode(out[0], skip_special_tokens=True)
        prev = list(range(len(tgt) + 1))
        for i, ca in enumerate(pred, 1):
            cur = [i]
            for j, cb in enumerate(tgt, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                               prev[j - 1] + (ca != cb)))
            prev = cur
        total_c += prev[-1]
        total_n += max(len(tgt), 1)
    model.train()
    return total_c / max(total_n, 1)


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)

    tok = AutoTokenizer.from_pretrained(BACKBONE)
    train = load(TRAIN_FILE)
    dev = load(DEV_FILE)
    print(f"[data] train={len(train)} dev={len(dev)}", flush=True)

    eps = sorted(CKPT.glob("oepoch-*"),
                 key=lambda p: int(p.name.split("-")[1]))
    if eps:
        model = AutoModelForSeq2SeqLM.from_pretrained(eps[-1])
        start = int(eps[-1].name.split("-")[1]) + 1
        print(f"[resume] from {eps[-1].name}", flush=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(BACKBONE)
        start = 0
    model.cuda()

    ds = Pairs(train, tok)
    dl = DataLoader(ds, batch_size=BS, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    steps = max(1, (len(dl) + ACCUM - 1) // ACCUM) * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    sched = get_cosine_schedule_with_warmup(opt, int(steps * 0.05), steps)

    model.train()
    gstep, t0 = 0, time.time()
    for ep in range(start, EPOCHS):
        for i, (ids, att, lab) in enumerate(dl):
            out = model(input_ids=ids.cuda(), attention_mask=att.cuda(),
                        labels=lab.cuda())
            (out.loss / ACCUM).backward()
            if (i + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                gstep += 1
                if gstep % 100 == 0:
                    print(f"[train] ep{ep} {gstep}/{steps} "
                          f"loss={out.loss.item():.4f} "
                          f"({(time.time()-t0)/60:.0f}m)", flush=True)
        ep_dir = CKPT / f"oepoch-{ep}"
        model.save_pretrained(ep_dir)
        tok.save_pretrained(ep_dir)
        cer = dev_cer(model, tok, dev)
        print(f"[ckpt] oepoch-{ep} saved dev_cer={cer:.4f}", flush=True)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()

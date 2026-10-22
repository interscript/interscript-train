"""WO54 stage-2 — continue-train run-077/best on fas oracle targets.

The multi-arm oracle (run-080-fas-oracle) showed oracle CER 0.375% vs
the RC arm's 0.446% on the same 120K rows — this job distills that
into the weights: +2 epochs on oracle-targets.jsonl from the RC
checkpoint, then the docs-split verdict under BOTH CER conventions.
Gate: row_mean < 0.00603 (beat run-077/best).

Env: F_S2_RUN (default run-081-fas-stage2).

Usage (a100-large, mounts isx-training:/ckpt:rw):
    python /ckpt/run-081-fas-stage2/fas_stage2_hf.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

CKPT = Path("/ckpt") / os.environ.get("F_S2_RUN", "run-081-fas-stage2")
INIT = Path("/ckpt/run-077-fas-diac-epochs/best")
ORACLE = Path("/ckpt/run-080-fas-oracle/oracle-targets.jsonl")
TEST = Path("/ckpt/run-076-fas-diac-retrain/test.jsonl")
EPOCHS = int(os.environ.get("F_EPOCHS", "2"))
BS = int(os.environ.get("F_BS", "16"))
ACCUM = int(os.environ.get("F_ACCUM", "2"))
MAXLEN = 256
GATE_ROW_MEAN = 0.00603  # run-077/best, the number to beat


def lev(a: str, b: str) -> int:
    try:
        from Levenshtein import distance

        return distance(a, b)
    except ImportError:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                               prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


class Rows(Dataset):
    def __init__(self, path: Path, tok, max_len: int = MAXLEN):
        self.examples = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            src, tgt = (r.get("src") or "").strip(), (r.get("tgt") or "").strip()
            if src and tgt:
                self.examples.append((src, tgt))
        self.tok, self.max_len = tok, max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        src, tgt = self.examples[idx]
        mi = self.tok(src, truncation=True, max_length=self.max_len)
        lab = self.tok(tgt, truncation=True, max_length=self.max_len)
        return {"input_ids": mi.input_ids, "labels": lab.input_ids}


def collate(batch, pad: int):
    width = max(len(b["input_ids"]) for b in batch)
    lw = max(len(b["labels"]) for b in batch)
    ids = torch.full((len(batch), width), pad, dtype=torch.long)
    lab = torch.full((len(batch), lw), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, :len(b["input_ids"])] = torch.tensor(b["input_ids"])
        lab[i, :len(b["labels"])] = torch.tensor(b["labels"])
    return {"input_ids": ids, "labels": lab}


def docs_eval(model, tok) -> dict:
    rows = [json.loads(l) for l in TEST.read_text(
        encoding="utf-8").splitlines() if l.strip()]
    ex = [((r.get("src") or "").strip(), (r.get("tgt") or "").strip())
          for r in rows if (r.get("src") or "").strip() and (r.get("tgt") or "").strip()]
    model.eval()
    total_ed = total_gold = 0
    row_sum = 0.0
    exact = 0
    with torch.no_grad():
        for i in range(0, len(ex), 16):
            batch = ex[i:i + 16]
            enc = tok([s for s, _ in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to("cuda")
            out = model.generate(**enc, max_new_tokens=256, num_beams=1)
            preds = tok.batch_decode(out, skip_special_tokens=True)
            for (_, gold), p in zip(batch, preds):
                p = p.strip()
                d = lev(p, gold.strip())
                total_ed += d
                total_gold += len(gold.strip())
                row_sum += d / max(1, len(gold.strip()))
                exact += int(p == gold.strip())
    n = len(ex)
    model.train()
    return {"rows": n, "corpus_cer": round(total_ed / max(1, total_gold), 5),
            "row_mean_cer": round(row_sum / n, 5),
            "word_acc": round(exact / n, 4)}


def sync(path: Path) -> None:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    dst = f"buckets/Interscript/isx-training/{path.relative_to(Path('/ckpt'))}"
    with open(path, "rb") as f, fs.open(dst, "wb") as g:
        g.write(f.read())


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)

    tok = AutoTokenizer.from_pretrained(INIT)
    model = AutoModelForSeq2SeqLM.from_pretrained(INIT).cuda()
    train = Rows(ORACLE, tok)
    print(f"[data] oracle rows={len(train)}", flush=True)
    dl = DataLoader(train, batch_size=BS, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    steps = len(dl) * EPOCHS // ACCUM
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, 200, steps)

    best = None
    t0 = time.time()
    step = 0
    model.train()
    for ep in range(EPOCHS):
        for i, batch in enumerate(dl):
            batch = {k: v.cuda() for k, v in batch.items()}
            loss = model(**batch).loss / ACCUM
            loss.backward()
            if (i + 1) % ACCUM == 0:
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
            if (i + 1) % 500 == 0:
                print(f"[train] ep{ep} {i + 1}/{len(dl)} "
                      f"loss={loss.item():.4f} ({time.time() - t0:.0f}s)",
                      flush=True)
        m = docs_eval(model, tok)
        m["epoch"] = ep
        print("[gate] " + json.dumps(m), flush=True)
        (CKPT / f"epoch-{ep}").mkdir(parents=True, exist_ok=True)
        model.save_pretrained(CKPT / f"epoch-{ep}")
        tok.save_pretrained(CKPT / f"epoch-{ep}")
        if best is None or m["row_mean_cer"] < best["row_mean_cer"]:
            best = m
            (CKPT / "best").mkdir(parents=True, exist_ok=True)
            model.save_pretrained(CKPT / "best")
            tok.save_pretrained(CKPT / "best")

    verdict = dict(best or {})
    verdict["gate"] = f"row_mean < {GATE_ROW_MEAN} (beat run-077/best)"
    verdict["verdict"] = ("PASS" if best and best["row_mean_cer"] < GATE_ROW_MEAN
                          else "MISS")
    print(json.dumps(verdict), flush=True)
    (CKPT / "verdict.json").write_text(json.dumps(verdict, indent=1))
    try:
        sync(CKPT / "verdict.json")
    except Exception as e:  # noqa: BLE001
        print(f"sync failed: {e}", flush=True)


if __name__ == "__main__":
    main()

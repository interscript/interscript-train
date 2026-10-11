"""WO33 — the word-identity channel: variant-disambiguation tagger.

The lexical decomposition (RESULTS.md WO32): pure most-frequent-reading
lookup over the corpus scores 25.15 WER with 18.1% OOV — lexical
memorization is 3/4 of the task, context disambiguation is the rest.
This arm adds the context: word-embedding + char-CNN + BiLSTM choosing
among each word's observed readings (Fadel's design, word level).

Labels: variant index per word (0=most frequent ... cap N_VAR-1);
OOV class for unseen words. Decode emits the chosen variant string;
output preserves the input skeleton letter-exactly (variants are keyed
by their own bare form). fp32 (the LSTM bf16 lesson).

Usage:
    python /ckpt/run-035-wordlex/train_wordlex_hf.py
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/code")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA = Path("/train_data")
RUN = os.environ.get("WL_RUN", "run-035-wordlex")
CKPT = Path("/ckpt") / RUN
EPOCHS = int(os.environ.get("WL_EPOCHS", "5"))
BS = int(os.environ.get("WL_BS", "128"))
N_VAR = 6          # variant classes + 1 OOV
EMB_W, EMB_C = 128, 64
HID, LAYERS = 512, 2
MAX_WORDS = 250    # sentence truncation

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")


def is_arabic(w: str) -> bool:
    return any("؀" <= c <= "ۿ" for c in w)


def sentences_from_articles(articles: list[str], cap: int) -> list[str]:
    out: list[str] = []
    for a in articles:
        if not a.strip():
            continue
        art = json.loads(a).get("text", "")
        for para in art.split("\n"):
            p = para.strip()
            if p and len(p.split()) >= 3:
                out.append(p)
                if len(out) >= cap:
                    return out
    return out


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    # ---- corpus (news-pure + full silver, same as run-033/034) ----
    articles = (DATA / "qcri-wiki.jsonl").read_text(encoding="utf-8").splitlines()
    rng.shuffle(articles)
    sents: list[str] = sentences_from_articles(articles, 900_000)
    gold = [l.strip() for l in (DATA / "wikinews2014_gold.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    news = [l.strip() for l in (DATA / "news.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    sents += gold * 4 + news
    print(f"[data] sentences={len(sents)}", flush=True)

    # ---- variant table ----
    table: dict[str, list[str]] = {}
    counts: dict[str, Counter] = defaultdict(Counter)
    for s in sents:
        for w in s.split():
            bare = DIACRITICS_RE.sub("", w)
            if bare and is_arabic(bare):
                counts[bare][w] += 1
    for bare, c in counts.items():
        table[bare] = [v for v, _ in c.most_common(N_VAR)]
    print(f"[table] word types={len(table)}", flush=True)

    # word vocab (bounded)
    wv = {"<pad>": 0, "<unk>": 1}
    for bare, c in counts.items():
        if c.total() >= 2 and len(wv) < 400_000:
            wv[bare] = len(wv)
    cv = {"<pad>": 0, "<unk>": 1}
    for bare in wv:
        for ch in bare:
            cv.setdefault(ch, len(cv))
    print(f"[vocab] words={len(wv)} chars={len(cv)}", flush=True)

    def encode(s: str):
        words = s.split()[:MAX_WORDS]
        wids, cids, labels = [], [], []
        for w in words:
            bare = DIACRITICS_RE.sub("", w)
            wid = wv.get(bare, 1)
            cid = [cv.get(ch, 1) for ch in bare][:20] or [1]
            if bare in table:
                variants = table[bare]
                lab = variants.index(w) if w in variants else 0
            else:
                lab = N_VAR
            wids.append(wid)
            cids.append(cid)
            labels.append(lab)
        return wids, cids, labels

    class DS(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return encode(self.rows[i])

    def collate(b):
        mw = max(len(x[0]) for x in b)
        mc = max(max((len(c) for c in x[1]), default=1) for x in b)
        wids = torch.zeros((len(b), mw), dtype=torch.long)
        am = torch.zeros((len(b), mw), dtype=torch.long)
        cids = torch.zeros((len(b), mw, mc), dtype=torch.long)
        cam = torch.zeros((len(b), mw, mc), dtype=torch.long)
        lab = torch.full((len(b), mw), -100, dtype=torch.long)
        for i, (w, cs, ys) in enumerate(b):
            wids[i, :len(w)] = torch.tensor(w)
            am[i, :len(w)] = 1
            lab[i, :len(ys)] = torch.tensor(ys)
            for j, c in enumerate(cs):
                cids[i, j, :len(c)] = torch.tensor(c)
                cam[i, j, :len(c)] = 1
        return wids, am, cids, cam, lab

    loader = DataLoader(DS(sents), batch_size=BS, shuffle=True, collate_fn=collate,
                        num_workers=2, drop_last=True)
    total_steps = len(loader) * EPOCHS

    class WordLex(nn.Module):
        def __init__(self):
            super().__init__()
            self.wemb = nn.Embedding(len(wv), EMB_W, padding_idx=0)
            self.cemb = nn.Embedding(len(cv), EMB_C, padding_idx=0)
            self.convs = nn.ModuleList(
                [nn.Conv1d(EMB_C, 50, k, padding=k // 2) for k in (2, 3, 4, 5)])
            self.lstm = nn.LSTM(EMB_W + 200, HID, num_layers=LAYERS,
                                bidirectional=True, batch_first=True, dropout=0.3)
            self.head = nn.Linear(HID * 2, N_VAR + 1)

        def forward(self, wids, cids, cam):
            ce = self.cemb(cids)                       # [B,W,C,E]
            ce = ce * cam.unsqueeze(-1)
            ce = ce.permute(0, 2, 1, 3).reshape(cids.shape[0] * cids.shape[1], EMB_C, -1)
            conv = torch.cat([c(ce).relu().max(-1).values for c in self.convs], -1)
            conv = conv.view(cids.shape[0], cids.shape[1], 200)
            x = torch.cat([self.wemb(wids), conv], -1)
            h, _ = self.lstm(x)
            return self.head(h)

    model = WordLex().cuda().train()
    print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"steps={total_steps}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    step = 0
    t0 = time.time()
    for _ep in range(EPOCHS):
        for wids, am, cids, cam, lab in loader:
            wids, cids, cam, lab = wids.cuda(), cids.cuda(), cam.cuda(), lab.cuda()
            logits = model(wids, cids, cam)  # fp32 (LSTM bf16 = NaN)
            loss = ce(logits.reshape(-1, N_VAR + 1), lab.reshape(-1))
            if not torch.isfinite(loss):
                print(f"[nan] step {step}", flush=True)
                opt.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"[train] {step}/{total_steps} loss={float(loss):.3f} "
                      f"({step/(time.time()-t0):.1f} it/s)", flush=True)

    model.eval()
    best = CKPT / "best"
    best.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "table": table, "wv": wv, "cv": cv},
               best / "model.pt")

    @torch.no_grad()
    def tag(text: str) -> str:
        words = text.split()[:MAX_WORDS]
        wids, cids, labels = encode(text)
        if not wids:
            return text
        tw = torch.tensor([wids], device="cuda")
        tc = torch.zeros((1, len(cids), max(len(c) for c in cids)), dtype=torch.long, device="cuda")
        tcam = torch.zeros_like(tc)
        for j, c in enumerate(cids):
            tc[0, j, :len(c)] = torch.tensor(c, device="cuda")
            tcam[0, j, :len(c)] = 1
        logits = model(tw, tc, tcam)
        preds = logits[0].argmax(-1).tolist()
        out = []
        for w, bare, p in zip(words, [DIACRITICS_RE.sub("", w) for w in words], preds):
            if bare in table:
                vi = p if p < len(table[bare]) else 0
                out.append(table[bare][vi])
            else:
                out.append(w)
        return " ".join(out)

    # ---- gates ----
    import pandas as pd
    import pyarrow.parquet as pq

    t = pq.read_table(DATA / "sadeeddiac-25.parquet")
    s_in = [DIACRITICS_RE.sub("", x) for x in t.column("input").to_pylist()]
    s_out = t.column("output").to_pylist()
    paras = [tag(x) for x in s_in]
    csv_p = Path("/tmp/wl_sadeed.csv")
    pd.DataFrame({"gt": s_out, "pred": paras}).to_csv(csv_p, index=False, header=False)
    (CKPT / "sadeed_preds.csv").write_text(csv_p.read_text(), encoding="utf-8")
    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    print(f"===== {RUN} SadeedDiac-25 =====", flush=True)
    E.report_errors_on_csv_file(str(csv_p), ground_truth_column_index=0,
                                predicted_column_index=1, has_header=False,
                                gt_missing_diacritic_is_error=False)

    import eval_wikinews_multiref as WN

    bench = [l for l in (DATA / "wikinews/WikiNews_2024_Multi_Ref.txt.diac")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    wparas = [tag(DIACRITICS_RE.sub("", l)) for l in bench]
    wscore = WN.score(wparas, bench, skip_last=False)
    print(json.dumps({"run": RUN, "surface": "wikinews-2024-multiref", **wscore},
                     ensure_ascii=False), flush=True)
    (CKPT / "wikinews_multiref.json").write_text(
        json.dumps({"run": RUN, **wscore}, ensure_ascii=False, indent=1))
    (CKPT / "EVAL_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

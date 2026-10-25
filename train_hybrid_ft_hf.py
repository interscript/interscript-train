"""WO34 run-037 — hybrid char-BiLSTM + PRETRAINED fastText word vectors.

run-036 injects word identity via scratch embeddings trained on ~3M
words of corpus; Fadel-class systems ride on word vectors trained on
billions. This arm swaps the scratch channel for cc.ar.300 (fastText,
Common Crawl) — the lexical-density hypothesis: our word channel is
undertrained, not wrong. Same corpus, same gates, fp32.

Usage:
    python /ckpt/run-037-hybrid-ft/train_hybrid_ft_hf.py
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/code")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import haraqat_planes as HP

DATA = Path("/train_data")
RUN = os.environ.get("HYB_RUN", "run-037-hybrid-ft")
CKPT = Path("/ckpt") / RUN
SILVER_CAP = int(os.environ.get("HYB_SILVER_CAP", "900000"))
EPOCHS = int(os.environ.get("HYB_EPOCHS", "6"))
BS = int(os.environ.get("HYB_BS", "64"))
EMB = 128
WEMB = 300
FT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ar.300.vec.gz"
HID = 512
LAYERS = 3
WINDOW = 1400
WV_CAP = 300_000

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")


def split_windows(text: str, budget: int = WINDOW) -> list[str]:
    if len(text.encode("utf-8")) <= budget:
        return [text]
    words, cur, n, wins = text.split(), [], 0, []
    for w in words:
        c = len(w.encode("utf-8")) + 1
        if cur and n + c > budget:
            wins.append(" ".join(cur)); cur, n = [], 0
        cur.append(w); n += c
    if cur:
        wins.append(" ".join(cur))
    return wins


def word_ids_per_char(skel: str, wv: dict) -> list[int]:
    """wid of the word each char belongs to; non-word chars get 0.
    Exact per-char alignment: len(out) == len(skel)."""
    out: list[int] = []
    cur: list[str] = []

    def flush() -> None:
        if cur:
            tok = "".join(cur)
            core = "".join(c for c in tok if c.isalpha())
            wid = wv.get(core, 1) if core else 0
            out.extend([wid] * len(tok))
            cur.clear()

    for ch in skel:
        if ch.isspace():
            flush()
            out.append(0)
        else:
            cur.append(ch)
    flush()
    return out


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    lines: list[str] = []
    news = [l for l in (DATA / "news.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = [l for l in (DATA / "wikinews2014_gold.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    units = news + gold * 4
    silver = 0
    articles = (DATA / "qcri-wiki.jsonl").read_text(encoding="utf-8").splitlines()
    rng.shuffle(articles)
    for aline in articles:
        if silver >= SILVER_CAP:
            break
        if not aline.strip():
            continue
        art = json.loads(aline).get("text", "")
        for para in art.split("\n"):
            para = para.strip()
            if not para:
                continue
            for w in split_windows(para):
                units.append(w)
                silver += 1
            if silver >= SILVER_CAP:
                break
    print(f"[data] news={len(news)} gold={len(gold)} silver={silver}", flush=True)

    units = [u for u in (u.strip() for u in units)
             if u and DIACRITICS_RE.sub("", u).encode("utf-8") != b""]
    random.Random(42).shuffle(units)
    classes, counts = HP.plane_classes(units)
    class_to_id = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    print(f"[data] units={len(units)} classes={n_classes}", flush=True)

    vocab = {"<pad>": 0, "<unk>": 1}
    for u in units:
        for ch in HP.split_planes(u)[0]:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    print(f"[vocab] chars={len(vocab)}", flush=True)

    wc: Counter = Counter()
    for u in units:
        for tok in HP.split_planes(u)[0].split(" "):
            core = "".join(c for c in tok if c.isalpha())
            if core and any("؀" <= c <= "ۿ" for c in core):
                wc[core] += 1
    wv = {"<pad>": 0, "<unk>": 1}
    for tok, n in wc.most_common():
        if len(wv) >= WV_CAP:
            break
        if n >= 2:
            wv[tok] = len(wv)
    print(f"[vocab] words={len(wv)} (types={len(wc)})", flush=True)

    # ---- pretrained fastText init for the word channel (cached) ----
    import gzip
    import subprocess

    ft_cache = CKPT / "ft_init.pt"
    if ft_cache.exists():
        emb_init = torch.load(ft_cache)
        print(f"[ft] cached init {tuple(emb_init.shape)}", flush=True)
    else:
        ft_path = CKPT / "cc.ar.300.vec.gz"
        if not ft_path.exists():
            print("[ft] downloading cc.ar.300.vec.gz ...", flush=True)
            subprocess.run(["curl", "-sL", "-o", str(ft_path), FT_URL], check=True)
            print(f"[ft] {ft_path.stat().st_size/1e9:.2f} GB", flush=True)

        needed = {w for w in wv if w not in ("<pad>", "<unk>")}
        found: dict[str, list[float]] = {}
        with gzip.open(ft_path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) != WEMB + 1:
                    continue
                w = parts[0]
                if w in needed and w not in found:
                    try:
                        found[w] = [float(x) for x in parts[1:]]
                    except ValueError:
                        continue
        emb_init = torch.randn(len(wv), WEMB) * 0.02
        emb_init[0].zero_()  # pad
        for w, vec in found.items():
            emb_init[wv[w]] = torch.tensor(vec)
        print(f"[ft] coverage {len(found)}/{len(needed)} "
              f"({100*len(found)/max(1,len(needed)):.1f}%)", flush=True)
        torch.save(emb_init, ft_cache)

    def encode(text: str):
        skel, labels = HP.split_planes(text)
        ids = [vocab.get(ch, 1) for ch in skel]
        wids = word_ids_per_char(skel, wv)
        lab = [class_to_id.get(HP.canon_combo(c), 0) for c in labels]
        return ids, wids, lab

    class DS(Dataset):
        def __len__(self):
            return len(units)

        def __getitem__(self, i):
            return encode(units[i])

    def collate(b):
        mx = max(len(x[0]) for x in b)
        ids = torch.zeros((len(b), mx), dtype=torch.long)
        wids = torch.zeros((len(b), mx), dtype=torch.long)
        lab = torch.full((len(b), mx), -100, dtype=torch.long)
        for i, (x, w, y) in enumerate(b):
            ids[i, :len(x)] = torch.tensor(x)
            wids[i, :len(w)] = torch.tensor(w)
            lab[i, :len(y)] = torch.tensor(y)
        return ids, wids, lab

    g = torch.Generator()
    g.manual_seed(42)
    loader = DataLoader(DS(), batch_size=BS, shuffle=True, collate_fn=collate,
                        num_workers=2, drop_last=True, generator=g)
    total_steps = len(loader) * EPOCHS

    class HybridTagger(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(vocab), EMB, padding_idx=0)
            self.wemb = nn.Embedding.from_pretrained(emb_init, padding_idx=0,
                                                     freeze=False)
            self.lstm = nn.LSTM(EMB + WEMB, HID, num_layers=LAYERS,
                                bidirectional=True, batch_first=True, dropout=0.3)
            self.head = nn.Linear(HID * 2, n_classes)

        def forward(self, ids, wids):
            x = torch.cat([self.emb(ids), self.wemb(wids)], -1)
            h, _ = self.lstm(x)
            return self.head(h)

    model = HybridTagger().cuda().train()
    print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"steps={total_steps}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    saves = sorted(CKPT.glob("step-*/model.pt"),
                   key=lambda p: int(p.parent.name.split("-")[1]))
    start_step = 0
    if saves:
        st = torch.load(saves[-1] / "model.pt", map_location="cuda")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_step = st["step"]
        print(f"[resume] from {saves[-1].parent.name} step {start_step}", flush=True)

    step = 0
    t0 = time.time()
    for _ep in range(EPOCHS):
        for ids, wids, lab in loader:
            step += 1
            if step <= start_step:  # deterministic order: seeded generator
                continue
            ids, wids, lab = ids.cuda(), wids.cuda(), lab.cuda()
            logits = model(ids, wids)  # fp32: LSTMs NaN-collapse under bf16
            loss = ce(logits.reshape(-1, n_classes), lab.reshape(-1))
            if not torch.isfinite(loss):
                print(f"[nan] step {step}: skipping batch", flush=True)
                opt.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            if step % 2000 == 0:
                d = CKPT / f"step-{step}"
                d.mkdir(exist_ok=True)
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step}, d / "model.pt")
            if step % 200 == 0:
                print(f"[train] {step}/{total_steps} loss={float(loss):.3f} "
                      f"({(step-start_step)/(time.time()-t0):.1f} it/s)", flush=True)

    model.eval()
    best = CKPT / "best"
    best.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": classes,
                "vocab": vocab, "wv": wv}, best / "model.pt")

    import pandas as pd
    import pyarrow.parquet as pq

    @torch.no_grad()
    def tag(text: str) -> str:
        skel, _ = HP.split_planes(text)
        ids = [vocab.get(ch, 1) for ch in skel] or [0]
        wids = word_ids_per_char(skel, wv) or [0]
        t = torch.tensor([ids], device="cuda")
        wt = torch.tensor([wids], device="cuda")
        logits = model(t, wt)
        preds = logits[0].argmax(-1).tolist()
        return "".join(ch + (classes[preds[i]] if i < len(preds) else "")
                       for i, ch in enumerate(skel))

    table = pq.read_table(DATA / "sadeeddiac-25.parquet")
    s_inputs = [DIACRITICS_RE.sub("", t) for t in table.column("input").to_pylist()]
    s_outputs = table.column("output").to_pylist()
    all_windows, cw = [], []
    for text in s_inputs:
        ws = split_windows(text)
        cw.append(len(ws))
        all_windows.extend(ws)
    preds = [tag(w) for w in all_windows]
    k = 0
    paras = []
    for _t, c in zip(s_inputs, cw):
        paras.append(" ".join(preds[k:k+c])); k += c
    csv_p = Path("/tmp/hybrid_sadeed.csv")
    pd.DataFrame({"gt": s_outputs, "pred": paras}).to_csv(csv_p, index=False, header=False)
    (CKPT / "sadeed_preds.csv").write_text(csv_p.read_text(), encoding="utf-8")

    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    print(f"===== {RUN} SadeedDiac-25 =====", flush=True)
    E.report_errors_on_csv_file(str(csv_p), ground_truth_column_index=0,
                                predicted_column_index=1, has_header=False,
                                gt_missing_diacritic_is_error=False)

    import eval_wikinews_multiref as WN

    bench = [l for l in (DATA / "wikinews/WikiNews_2024_Multi_Ref.txt.diac")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    winputs = [DIACRITICS_RE.sub("", l) for l in bench]
    ww, wc2 = [], []
    for text in winputs:
        ws = split_windows(text); wc2.append(len(ws)); ww.extend(ws)
    wpreds = [tag(w) for w in ww]
    k = 0; wparas = []
    for _t, c in zip(winputs, wc2):
        wparas.append(" ".join(wpreds[k:k+c])); k += c
    (CKPT / "wikinews_preds.txt").write_text("\n".join(wparas) + "\n", encoding="utf-8")
    wscore = WN.score(wparas, bench, skip_last=False)
    print(json.dumps({"run": RUN, "surface": "wikinews-2024-multiref", **wscore},
                     ensure_ascii=False), flush=True)
    (CKPT / "wikinews_multiref.json").write_text(
        json.dumps({"run": RUN, **wscore}, ensure_ascii=False, indent=1))
    (CKPT / "EVAL_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

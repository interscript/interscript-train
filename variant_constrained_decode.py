"""WO34 v2 — variant-constrained decode on the trained hybrid model.

Loads run-036's best checkpoint and re-decodes both gates with lexical
constraint: for each in-table word, score every observed variant's
plane-combo sequence under the model's per-position log-softmax
(plus the free-greedy candidate as a competing option) and emit the
argmax. OOV words and non-Arabic tokens fall back to free decode —
compositional generalization is preserved.

Usage:
    python /ckpt/run-036-hybrid/variant_constrained_decode.py
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/code")

import torch

import haraqat_planes as HP

DATA = Path("/train_data")
CKPT = Path("/ckpt") / os.environ.get("VCD_RUN", "run-036-hybrid")
SILVER_CAP = 900_000
N_VAR = 6

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")


def split_windows(text: str, budget: int = 1400) -> list[str]:
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


def main() -> None:
    ck = torch.load(CKPT / "best" / "model.pt", map_location="cuda")
    classes = ck["classes"]
    vocab = ck["vocab"]
    wv = ck["wv"]
    cls_to_id = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    import torch.nn as nn

    class HybridTagger(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(vocab), 128, padding_idx=0)
            self.wemb = nn.Embedding(len(wv), 128, padding_idx=0)
            self.lstm = nn.LSTM(256, 512, num_layers=3, bidirectional=True,
                                batch_first=True, dropout=0.3)
            self.head = nn.Linear(1024, n_classes)

        def forward(self, ids, wids):
            x = torch.cat([self.emb(ids), self.wemb(wids)], -1)
            h, _ = self.lstm(x)
            return self.head(h)

    model = HybridTagger().cuda().eval()
    model.load_state_dict(ck["model"])
    print(f"[model] loaded n_classes={n_classes}", flush=True)

    # ---- variant table (deterministic rebuild, same corpus + seed) ----
    rng = random.Random(42)
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
    units = [u for u in (u.strip() for u in units)
             if u and DIACRITICS_RE.sub("", u).encode("utf-8") != b""]
    print(f"[table-corpus] units={len(units)}", flush=True)

    counts: dict[str, Counter] = defaultdict(Counter)
    for u in units:
        for w in u.split():
            core = "".join(c for c in w if c.isalpha())
            if core and any("؀" <= c <= "ۿ" for c in core):
                counts[core][w] += 1
    table = {b: [v for v, _ in c.most_common(N_VAR)] for b, c in counts.items()}
    print(f"[table] types={len(table)}", flush=True)

    def word_ids_per_char(skel: str) -> list[int]:
        out: list[int] = []
        cur: list[str] = []

        def flush() -> None:
            if cur:
                tok = "".join(cur)
                core = "".join(c for c in tok if c.isalpha())
                out.extend([wv.get(core, 1) if core else 0] * len(tok))
                cur.clear()

        for ch in skel:
            if ch.isspace():
                flush()
                out.append(0)
            else:
                cur.append(ch)
        flush()
        return out

    @torch.no_grad()
    def decode(text: str) -> str:
        skel, _ = HP.split_planes(text)
        if not skel:
            return text
        ids = [vocab.get(ch, 1) for ch in skel]
        wids = word_ids_per_char(skel)
        t = torch.tensor([ids], device="cuda")
        wt = torch.tensor([wids], device="cuda")
        logits = model(t, wt)
        logp = torch.log_softmax(logits[0], -1)
        free = logp.argmax(-1).tolist()

        out: list[str] = []
        i = 0
        n = len(skel)
        while i < n:
            if skel[i].isspace():
                out.append(skel[i])
                i += 1
                continue
            j = i
            while j < n and not skel[j].isspace():
                j += 1
            tok = skel[i:j]
            core = "".join(c for c in tok if c.isalpha())
            cands: list[tuple[float, list[str]]] = []
            if core in table:
                for v in table[core]:
                    vskel, vlabels = HP.split_planes(v)
                    if vskel != tok:
                        continue
                    lab = [cls_to_id.get(HP.canon_combo(c)) for c in vlabels]
                    if any(x is None for x in lab):
                        continue
                    s = sum(logp[i + k, x] for k, x in enumerate(lab)).item()
                    cands.append((s, [classes[x] for x in lab]))
            free_marks = [classes[free[i + k]] for k in range(j - i)]
            fs = sum(logp[i + k, free[i + k]] for k in range(j - i)).item()
            cands.append((fs, free_marks))
            best = max(cands, key=lambda x: x[0])[1]
            for k in range(j - i):
                out.append(skel[i + k] + best[k])
            i = j
        return "".join(out)

    # ---- gate 1: SadeedDiac-25 ----
    import pandas as pd
    import pyarrow.parquet as pq

    table_pq = pq.read_table(DATA / "sadeeddiac-25.parquet")
    s_inputs = [DIACRITICS_RE.sub("", t) for t in table_pq.column("input").to_pylist()]
    s_outputs = table_pq.column("output").to_pylist()
    aw, cw = [], []
    for text in s_inputs:
        ws = split_windows(text)
        cw.append(len(ws))
        aw.extend(ws)
    preds = [decode(w) for w in aw]
    k = 0
    paras = []
    for _t, c in zip(s_inputs, cw):
        paras.append(" ".join(preds[k:k + c])); k += c
    csv_p = Path("/tmp/vcd_sadeed.csv")
    pd.DataFrame({"gt": s_outputs, "pred": paras}).to_csv(csv_p, index=False, header=False)
    (CKPT / "sadeed_preds_vcd.csv").write_text(csv_p.read_text(), encoding="utf-8")

    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    print("===== run-036+vcd SadeedDiac-25 =====", flush=True)
    E.report_errors_on_csv_file(str(csv_p), ground_truth_column_index=0,
                                predicted_column_index=1, has_header=False,
                                gt_missing_diacritic_is_error=False)

    # ---- gate 2: WikiNews-2024 multiref ----
    import eval_wikinews_multiref as WN

    bench = [l for l in (DATA / "wikinews/WikiNews_2024_Multi_Ref.txt.diac")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    winputs = [DIACRITICS_RE.sub("", l) for l in bench]
    ww, wc2 = [], []
    for text in winputs:
        ws = split_windows(text); wc2.append(len(ws)); ww.extend(ws)
    wpreds = [decode(w) for w in ww]
    k = 0
    wparas = []
    for _t, c in zip(winputs, wc2):
        wparas.append(" ".join(wpreds[k:k + c])); k += c
    (CKPT / "wikinews_preds_vcd.txt").write_text("\n".join(wparas) + "\n", encoding="utf-8")
    wscore = WN.score(wparas, bench, skip_last=False)
    print(json.dumps({"run": f"{CKPT.name}-vcd", "surface": "wikinews-2024-multiref",
                      **wscore}, ensure_ascii=False), flush=True)
    (CKPT / "wikinews_multiref_vcd.json").write_text(
        json.dumps({"run": f"{CKPT.name}-vcd", **wscore}, ensure_ascii=False, indent=1))
    (CKPT / "VCD_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

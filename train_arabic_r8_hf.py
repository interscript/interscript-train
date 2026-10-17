# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.5.1",
#   "transformers==4.46.3",
#   "accelerate>=1.1.0",
#   "pyarrow",
#   "pandas",
#   "pyarabic",
#   "prettytable",
# ]
# ///
"""WO06 stage 2: r8 noisy-student — gold tashkeela + r7-pseudo arwiki.

Student initializes FROM r7 (born-again variant: guarantees the r7
starting point) and trains one epoch on the interleaved mixture.
Gate (pre-registered): beat 2.2864 DER on SadeedDiac-25 under the exact
windowed protocol (WINDOW=600, greedy, project_haraqat, Misraj evaluator,
gt_missing_diacritic_is_error=False).

Usage:
    hf jobs uv run --flavor a100-large -d --timeout 20h \
      -v hf://datasets/Interscript/arabic-r8:/train_data:ro \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      -v /tmp/isx-hf-code:/code:ro train_arabic_r8_hf.py
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/code")

DATA = Path("/train_data")
TEACHER = Path("/ckpt/r7-best")
import os
MIX = os.environ.get("R8_MIX", "arwiki")  # arwiki | qcri | both
STRIP_ALT = os.environ.get("R8_STRIP_ALT") == "1"
BACKBONE = os.environ.get("R8_BACKBONE", "google/byt5-base")
CORPUS = os.environ.get("R8_CORPUS", "tashkeela")  # tashkeela | r5
RUN = os.environ.get("R8_RUN", f"run-024-arabic-r8-{MIX}")
CKPT = Path(f"/ckpt/{RUN}")
WINDOW = 600
GOLD_CAP = 400_000
PSEUDO_CAP = int(os.environ.get("R8_PSEUDO_CAP", "200_000"))
MICRO_BS = 16
GRAD_ACCUM = 2
EPOCHS = int(os.environ.get("R8_EPOCHS", "1"))
LR = float(os.environ.get("R8_LR", "5e-5"))
WARMUP = 500

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


def project_haraqat(pred: str, text: str) -> str:
    from difflib import SequenceMatcher

    pred_haraqat = [""]
    for ch in pred:
        if DIACRITICS_RE.match(ch):
            pred_haraqat[-1] += ch
        else:
            pred_haraqat.append("")
    pred_haraqat = pred_haraqat[1:]
    pred_letters = [c for c in pred if not DIACRITICS_RE.match(c)]
    text_letters = [c for c in text if not DIACRITICS_RE.match(c)]
    sm = SequenceMatcher(None, text_letters, pred_letters, autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                out.append(text_letters[i1 + k] + pred_haraqat[j1 + k])
        else:
            for k in range(i1, i2):
                out.append(text_letters[k])
    return "".join(out)


def main() -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_cosine_schedule_with_warmup

    CKPT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    gold: list[tuple[str, str]] = []
    if CORPUS == "r5":
        # the r7 lineage corpus (r5 units + news), no silver
        for name in ("domain.txt", "replay.txt"):
            p = DATA / name
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                tgt = line.strip()
                if not tgt:
                    continue
                src = DIACRITICS_RE.sub("", tgt).strip()
                if len(src) < 5 or len(src.encode()) > WINDOW - 50:
                    continue
                gold.append((src, tgt))
                if len(gold) >= GOLD_CAP * 2:
                    break
            if len(gold) >= GOLD_CAP * 2:
                break
        for line in (DATA / "news.txt").read_text(encoding="utf-8").splitlines()[:120_000]:
            tgt = line.strip()
            if not tgt:
                continue
            src = DIACRITICS_RE.sub("", tgt).strip()
            if len(src) < 5 or len(src.encode()) > WINDOW - 50:
                continue
            gold.append((src, tgt))
        print(f"[data] r5-corpus gold={len(gold)}", flush=True)
    gold_files = ("tashkeela-full/train-001.txt", "tashkeela-full/train-002.txt",
                  "tashkeela-full/train-003.txt", "tashkeela-full/val-001.txt")
    if os.environ.get("R8_GOLD", "tashkeela") == "none":
        gold_files = ()
    for name in gold_files:
        p = DATA / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            tgt = line.strip()
            if not tgt:
                continue
            src = DIACRITICS_RE.sub("", tgt).strip()
            if len(src) < 5 or len(src.encode()) > WINDOW - 50:
                continue
            gold.append((src, tgt))
            if len(gold) >= GOLD_CAP:
                break
        if len(gold) >= GOLD_CAP:
            break
    print(f"[data] gold={len(gold)}", flush=True)

    pseudo: list[tuple[str, str]] = []

    def add_silver(tgt: str) -> None:
        tgt = tgt.strip()
        src = DIACRITICS_RE.sub("", tgt).strip()
        if len(src) >= 5 and len(src.encode()) <= WINDOW - 50:
            pseudo.append((src, tgt))

    if MIX in ("arwiki", "both"):
        ppath = Path("/ckpt/r8-pseudo/arwiki-pseudo.jsonl")
        if ppath.exists():
            for line in ppath.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("keep_frac", 1.0) < 0.9:
                    continue
                add_silver(r["tgt"])
                if len(pseudo) >= PSEUDO_CAP:
                    break
    if MIX in ("qcri", "both") and len(pseudo) < PSEUDO_CAP:
        qpath = DATA / "qcri-wiki.jsonl"
        # QCRI silver (Wikipedia_20240420.diac.jsonl, their BiLSTM labels,
        # published with EMNLP 2025) - windowed to the 600-byte protocol
        rng2 = random.Random(7)
        articles = qpath.read_text(encoding="utf-8").splitlines()
        rng2.shuffle(articles)
        for aline in articles:
            if len(pseudo) >= PSEUDO_CAP:
                break
            if not aline.strip():
                continue
            art = json.loads(aline).get("text", "")
            for para in art.split("\n"):
                para = para.strip()
                if not para:
                    continue
                if len(para.encode()) <= WINDOW - 50:
                    add_silver(para)
                else:
                    for w in split_windows(para, WINDOW - 50):
                        add_silver(w)
                if len(pseudo) >= PSEUDO_CAP:
                    break
    print(f"[data] pseudo={len(pseudo)} (mix={MIX})", flush=True)

    units = gold + pseudo
    if STRIP_ALT:
        # WO35: the corpus's multi-reference format is decode noise —
        # train the primary reading as the whole job
        stripped = []
        for _src, tgt in units:
            tgt2 = " ".join(t.split("/", 1)[0] for t in tgt.split())
            src2 = DIACRITICS_RE.sub("", tgt2).strip()
            if len(src2) >= 5:
                stripped.append((src2, tgt2))
        units = stripped
        print(f"[data] strip-alt units={len(units)}", flush=True)
    rng.shuffle(units)
    rng.shuffle(gold)

    tok = AutoTokenizer.from_pretrained(BACKBONE if os.environ.get("R8_SCRATCH") == "1" else TEACHER)

    class DS(Dataset):
        def __len__(self):
            return len(units)

        def __getitem__(self, i):
            src, tgt = units[i]
            e = tok(src, truncation=True, max_length=WINDOW)
            d = tok(tgt, truncation=True, max_length=WINDOW * 2)
            return {"input_ids": e.input_ids, "attention_mask": e.attention_mask,
                    "labels": d.input_ids}

    def collate(b):
        mx = max(len(x["input_ids"]) for x in b)
        md = max(len(x["labels"]) for x in b)
        ids = torch.zeros((len(b), mx), dtype=torch.long)
        am = torch.zeros((len(b), mx), dtype=torch.long)
        lb = torch.full((len(b), md), -100, dtype=torch.long)
        for i, x in enumerate(b):
            ids[i, :len(x["input_ids"])] = torch.tensor(x["input_ids"])
            am[i, :len(x["attention_mask"])] = torch.tensor(x["attention_mask"])
            lb[i, :len(x["labels"])] = torch.tensor(x["labels"])
        return ids, am, lb

    loader = DataLoader(DS(), batch_size=MICRO_BS, shuffle=True, collate_fn=collate,
                        num_workers=2, drop_last=True)
    total_steps = (len(loader) // GRAD_ACCUM) * EPOCHS
    print(f"[data] units={len(units)} steps={total_steps}", flush=True)

    if os.environ.get("R8_SCRATCH") == "1":
        init_from = BACKBONE
    else:
        init_from = TEACHER
    print(f"[init] from {init_from}", flush=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(init_from, torch_dtype=torch.float32)
    model = model.cuda().train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP, total_steps)

    # resume contract
    ckpts = sorted(CKPT.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    step = 0
    if ckpts:
        sd = torch.load(ckpts[-1] / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        optimizer.load_state_dict(torch.load(ckpts[-1] / "optim.pt", map_location="cpu", weights_only=True))
        step = int(ckpts[-1].name.split("-")[1])
        for _ in range(step):
            scheduler.step()
        print(f"[resume] step-{step}", flush=True)

    def save(s: int) -> None:
        d = CKPT / f"step-{s}"
        d.mkdir(exist_ok=True)
        torch.save(model.state_dict(), d / "model.pt")
        torch.save(optimizer.state_dict(), d / "optim.pt")
        print(f"[ckpt] step-{s}", flush=True)

    optimizer.zero_grad()
    micro = 0
    t0 = time.time()
    for _epoch in range(EPOCHS):
        for ids, am, lb in loader:
            if step >= total_steps:
                break
            ids, am, lb = ids.cuda(), am.cuda(), lb.cuda()
            with torch.autocast("cuda", torch.bfloat16):
                out = model(input_ids=ids, attention_mask=am, labels=lb)
            (out.loss / GRAD_ACCUM).backward()
            micro += 1
            if micro % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % 100 == 0:
                    print(f"[{RUN} {step}/{total_steps}] loss={float(out.loss):.4f} "
                          f"({step / (time.time() - t0):.2f} it/s)", flush=True)
                if step % 1000 == 0:
                    save(step)

    best = CKPT / "best"
    best.mkdir(exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(best)
    tok.save_pretrained(best)
    print("[ckpt] best saved", flush=True)

    # ---- gate: SadeedDiac-25, exact windowed protocol ----
    import pandas as pd
    import pyarrow.parquet as pq

    model.eval()
    table = pq.read_table(DATA / "sadeeddiac-25.parquet")
    inputs = [DIACRITICS_RE.sub("", t) for t in table.column("input").to_pylist()]
    outputs = table.column("output").to_pylist()
    all_windows, counts = [], []
    for text in inputs:
        ws = split_windows(text)
        counts.append(len(ws))
        all_windows.extend(ws)
    print(f"[eval] {len(inputs)} paragraphs, {len(all_windows)} windows", flush=True)

    preds = []
    with torch.no_grad():
        for i in range(0, len(all_windows), 32):
            batch = all_windows[i:i + 32]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=WINDOW).to("cuda")
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=WINDOW * 2, num_beams=1)
            preds.extend(tok.batch_decode(gen, skip_special_tokens=True))
            if (i // 32) % 10 == 0:
                print(f"[gen] {i + len(batch)}/{len(all_windows)}", flush=True)

    k = 0
    paragraphs = []
    for text, c in zip(inputs, counts):
        stitched = " ".join(preds[k:k + c])
        k += c
        paragraphs.append(project_haraqat(stitched, text))

    csv_path = Path("/tmp/sadeed_r8.csv")
    pd.DataFrame({"gt": outputs, "pred": paragraphs}).to_csv(csv_path, index=False, header=False)

    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    print(f"===== {RUN} SadeedDiac-25 (their default protocol) =====", flush=True)
    E.report_errors_on_csv_file(str(csv_path), ground_truth_column_index=0,
                                predicted_column_index=1, has_header=False,
                                gt_missing_diacritic_is_error=False)
    (CKPT / "sadeed_preds_r8.csv").write_text(csv_path.read_text(), encoding="utf-8")
    (CKPT / "EVAL_DONE").write_text("ok\n")
    print("[done] EVAL_DONE", flush=True)

    # ---- surface 2: WikiNews-2024 multiref (the register gate) ----
    import eval_wikinews_multiref as WN

    bench = (DATA / "wikinews/WikiNews_2024_Multi_Ref.txt.diac").read_text(
        encoding="utf-8").splitlines()
    bench = [l for l in bench if l.strip()]
    winputs = [DIACRITICS_RE.sub("", l) for l in bench]
    wwindows, wcounts = [], []
    for text in winputs:
        ws = split_windows(text)
        wcounts.append(len(ws))
        wwindows.extend(ws)
    wpreds = []
    with torch.no_grad():
        for i in range(0, len(wwindows), 32):
            batch = wwindows[i:i + 32]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=WINDOW).to("cuda")
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=WINDOW * 2, num_beams=1)
            wpreds.extend(tok.batch_decode(gen, skip_special_tokens=True))
            if (i // 32) % 5 == 0:
                print(f"[wgen] {i + len(batch)}/{len(wwindows)}", flush=True)
    k = 0
    wparas = []
    for text, c in zip(winputs, wcounts):
        stitched = " ".join(wpreds[k:k + c])
        wparas.append(project_haraqat(stitched, text))
        k += c
    wscore = WN.score(wparas, bench, skip_last=False)
    print(json.dumps({"run": RUN, "surface": "wikinews-2024-multiref", **wscore},
                     ensure_ascii=False), flush=True)
    (CKPT / "wikinews_multiref.json").write_text(
        json.dumps({"run": RUN, **wscore}, ensure_ascii=False, indent=1))
    # WO35 corrected protocol: primary-choice decode
    fa = [" ".join(w.split("/", 1)[0] for w in p.split()) for p in wparas]
    (CKPT / "wikinews_preds.txt").write_text("\n".join(fa) + "\n", encoding="utf-8")
    fascore = WN.score(fa, bench, skip_last=False)
    print(json.dumps({"run": RUN, "surface": "wikinews-2024-multiref-firstalt",
                      **fascore}, ensure_ascii=False), flush=True)
    (CKPT / "wikinews_multiref_firstalt.json").write_text(
        json.dumps({"run": RUN, **fascore}, ensure_ascii=False, indent=1))
    (CKPT / "EVAL_DONE").touch()


if __name__ == "__main__":
    main()

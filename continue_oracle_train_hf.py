"""WO39 stage 2 — continue-train run-047/best on oracle targets.

Loads /ckpt/run-065-oracle-distill/oracle-targets.jsonl (src, tgt,
pick — targets already strip-alt primary readings), +2 epochs lr 3e-5
from the e6 checkpoint, then the run-047 eval surfaces in-job
(SadeedDiac-25 default protocol + WikiNews-2024 multiref RAW and
first-alt corrected protocol, project_haraqat on stitched windows).

Gate: first-alt WER < 3.5607. Stretch: <= 2.70 (frontier parity).

Env: O_RUN (default run-065-oracle-distill), O_EPOCHS (default 2).

Usage (a100-large, mounts arabic-r8 dataset + isx-training bucket):
    python /ckpt/run-065-oracle-distill/continue_oracle_train_hf.py
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

DATA = Path("/train_data")
CKPT = Path("/ckpt") / os.environ.get("O_RUN", "run-065-oracle-distill")
ORACLE = CKPT / "oracle-targets.jsonl"
INIT = Path("/ckpt/run-047-large-strip/best")
EPOCHS = int(os.environ.get("O_EPOCHS", "2"))
WINDOW = 600
BS = 8
ACCUM = 2

DIACRITICS_RE = __import__("re").compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")


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
    import sys

    sys.path.insert(0, str(CKPT))
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)

    units = [(r["src"], r["tgt"]) for r in
             (json.loads(l) for l in ORACLE.read_text(
                 encoding="utf-8").splitlines() if l.strip())]
    random.Random(43).shuffle(units)
    print(f"[data] oracle units={len(units)}", flush=True)

    tok = AutoTokenizer.from_pretrained(INIT)
    model = AutoModelForSeq2SeqLM.from_pretrained(INIT).cuda().train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    eps = sorted(CKPT.glob("oepoch-*"), key=lambda p: int(p.name.split("-")[1]))
    if eps:
        model = AutoModelForSeq2SeqLM.from_pretrained(eps[-1]).cuda().train()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        print(f"[resume] {eps[-1].name}", flush=True)

    class DS(Dataset):
        def __len__(self):
            return len(units)

        def __getitem__(self, i):
            return units[i]

    def collate(b):
        es = [tok(s, truncation=True, max_length=WINDOW).input_ids
              for s, _ in b]
        ds_ = [tok(t, truncation=True, max_length=WINDOW * 2).input_ids
               for _, t in b]
        mx, md = max(len(e) for e in es), max(len(d) for d in ds_)
        ids = torch.zeros((len(b), mx), dtype=torch.long)
        am = torch.zeros((len(b), mx), dtype=torch.long)
        lb = torch.full((len(b), md), -100, dtype=torch.long)
        for i, (e, d) in enumerate(zip(es, ds_)):
            ids[i, :len(e)] = torch.tensor(e)
            am[i, :len(e)] = 1
            lb[i, :len(d)] = torch.tensor(d)
        return ids, am, lb

    loader = DataLoader(DS(), batch_size=BS, shuffle=True,
                        collate_fn=collate, num_workers=2, drop_last=True)
    total_steps = len(loader) * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, 200, total_steps)

    step, micro = 0, 0
    for ep in range(EPOCHS):
        for ids, am, lb in loader:
            ids, am, lb = ids.cuda(), am.cuda(), lb.cuda()
            with torch.autocast("cuda", torch.bfloat16):
                out = model(input_ids=ids, attention_mask=am, labels=lb)
            if not torch.isfinite(out.loss):
                print(f"[nan] step {step}", flush=True)
                opt.zero_grad()
                continue
            (out.loss / ACCUM).backward()
            micro += 1
            if micro % ACCUM:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f"[train] {step}/{total_steps // ACCUM} "
                      f"loss={float(out.loss):.3f}", flush=True)
        ep_dir = CKPT / f"oepoch-{ep}"
        ep_dir.mkdir(exist_ok=True)
        model.save_pretrained(ep_dir)
        tok.save_pretrained(ep_dir)
        print(f"[ckpt] oepoch-{ep} saved", flush=True)

    # ---- eval surfaces (run-047 protocol) ----
    # NOTE: everything below the checkpoint saves is an epilogue — a
    # missing dependency here must never fail a completed training run
    # (run-080-r8-stage2 died here after 12.7h on `import pandas`).
    model.config.use_cache = True
    model.eval()

    try:
        import pandas as pd  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError as e:
        print(f"[epilogue] skipped: {e} — checkpoints are complete; "
              "run eval_r8_stage2.py separately", flush=True)
        return

    table = pq.read_table(DATA / "sadeeddiac-25.parquet")
    inputs = [DIACRITICS_RE.sub("", t) for t in
              table.column("input").to_pylist()]
    outputs = table.column("output").to_pylist()
    windows, counts = [], []
    for text in inputs:
        ws = split_windows(text)
        counts.append(len(ws))
        windows.extend(ws)
    preds = []
    with torch.no_grad():
        for i in range(0, len(windows), 32):
            enc = tok(windows[i:i + 32], return_tensors="pt", padding=True,
                      truncation=True, max_length=WINDOW).to("cuda")
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=WINDOW * 2,
                                     num_beams=1)
            preds.extend(tok.batch_decode(gen, skip_special_tokens=True))
    k = 0
    paragraphs = []
    for text, c in zip(inputs, counts):
        paragraphs.append(project_haraqat(
            " ".join(preds[k:k + c]), text))
        k += c
    csv_path = Path("/tmp/sadeed_oracle.csv")
    pd.DataFrame({"gt": outputs, "pred": paragraphs}).to_csv(
        csv_path, index=False, header=False)
    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    print("===== SadeedDiac-25 (default protocol) =====", flush=True)
    E.report_errors_on_csv_file(str(csv_path), ground_truth_column_index=0,
                                predicted_column_index=1, has_header=False,
                                gt_missing_diacritic_is_error=False)

    import eval_wikinews_multiref as WN

    bench = [l for l in (DATA / "wikinews/WikiNews_2024_Multi_Ref.txt.diac")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    winputs = [DIACRITICS_RE.sub("", l) for l in bench]
    wwindows, wcounts = [], []
    for text in winputs:
        ws = split_windows(text)
        wcounts.append(len(ws))
        wwindows.extend(ws)
    wpreds = []
    with torch.no_grad():
        for i in range(0, len(wwindows), 32):
            enc = tok(wwindows[i:i + 32], return_tensors="pt", padding=True,
                      truncation=True, max_length=WINDOW).to("cuda")
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=WINDOW * 2,
                                     num_beams=1)
            wpreds.extend(tok.batch_decode(gen, skip_special_tokens=True))
    k = 0
    wparas = []
    for text, c in zip(winputs, wcounts):
        wparas.append(project_haraqat(" ".join(wpreds[k:k + c]), text))
        k += c
    wscore = WN.score(wparas, bench, skip_last=False)
    print(json.dumps({"run": CKPT.name, "surface":
                      "wikinews-2024-multiref", **wscore},
                     ensure_ascii=False), flush=True)
    fa = [" ".join(w.split("/", 1)[0] for w in p.split()) for p in wparas]
    fascore = WN.score(fa, bench, skip_last=False)
    verdict = {"run": CKPT.name, "surface": "wikinews-firstalt",
               "gate": "wer < 3.5607", "teacher": 3.5607, **fascore}
    print(json.dumps(verdict, ensure_ascii=False), flush=True)
    (CKPT / "oracle_verdict.json").write_text(
        json.dumps({"sadeed_csv": "written", "wikinews_raw": wscore,
                    "wikinews_firstalt": verdict}, ensure_ascii=False,
                   indent=1))
    (CKPT / "TRAIN_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

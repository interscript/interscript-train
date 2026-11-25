# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers<5",
#   "accelerate>=0.26",
#   "sentencepiece",
#   "Levenshtein",
#   "huggingface_hub>=0.26",
# ]
# ///
"""WO94 phase-2 — fas-vocalized SFT over the fas-ganjoor curriculum.

The owner-authorized direct-gold arm (gate PASS relayed 2026-11-25;
TODO.sota/94 phase-2 plan, merged via interscript-models#493): ByT5-base
(the fas docs-lineage backbone — the 0.603% docs champion is byt5-base
fp16, the 0.52% docs line is byt5-base 3ep) on the COMBINED curriculum
(the combined-corpus lesson):

  - fas-ganjoor  215,067 train rows (vocalized poetry, CC BY-SA 3.0,
    isx-training:/run-098-fas-acquisition/ganjoor-vocalized, manifest
    sha256 81a25fe02e058fdc112903e76cfc2ce91da109e6728a2a6004f9407986d0cdfb)
  - fas-dia register 445,869 train rows (news-classic gold, CC0-1.0,
    isx-training:/run-076-fas-diac-retrain, corpus record pins train
    sha256 05ea4b33… / test 67513aa1…)

seed-42 shuffle; NO eval contamination: the pinned evals (9,288-row docs
test + the 3,497-verse Ganjoor eval, sha256 94f318f688b556be…) are
EXCLUDED from training by the acquisition-time decontam and are used for
evaluation only. The WO66 arm's "3,497-verse seed x10" oversample is NOT
replicated (it would train on the pinned eval; the acquired vocalized
mass supersedes the tiny-seed oversample ~70x — recorded in the WO94
ledger). Held-out corpus slice: fas-ganjoor shard 7 (30,724 rows) is the
in-job val signal; the CAMPAIGN gates run on the two pinned external
evals.

Recipe (the fas crown recipe verbatim): AdamW lr 1e-4 wd 0.01, cosine,
warmup 500, clip 1.0, bf16 autocast, micro-bs 16 x accum 2, MAXLEN 256,
EPOCHS=3 (the docs 3ep convention; ~62k steps at bs-eff 32), step-1000
checkpoints with resume (the e12 lesson), ByT5 byte-table law (ids from
AutoTokenizer only), greedy decode only (num_beams=1 — the fas lineage;
beam is never used).

Kill criteria (pre-registered, enforced in-job):
  - non-finite loss: skip-step up to 50 occurrences; beyond that ABORT
    (the fp32-autocast-off fallback is a manual relaunch decision, one
    max, per the RMSNorm+bf16 lesson)
  - docs-hold gate FAIL at final eval => the model is a poetry specialist,
    NOT the successor (recorded negative, no successor claim)
  - poetry eval <= 16.55% (the register-arm precedent) => negative record

Final eval (in-job, greedy): row-mean RAW char CER + word accuracy on
  (a) the docs test (9,288) — GATE: row-mean CER < 0.0070 (WO66 register-
      hold law; the 0.603% champion must be held)
  (b) the pinned Ganjoor poetry eval (3,497) — REPORTED vs 33.67%
      baseline / 16.55% register-arm precedent (the register-gap
      reduction is the SOTA metric)
  (c) the fas-ganjoor held-out shard 7 (30,724) — corpus val, reported

Usage (a100-large, detached):
    hf jobs uv run --flavor a100-large -d --timeout 16h --attempts 2 \
      -s HF_TOKEN \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      --name run-099-fas-vocalized-sft train_fas_vocalized_hf.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

CKPT = Path("/ckpt")
OUT = CKPT / os.environ.get("FAS_RUN", "run-099-fas-vocalized-sft")
GANJOOR = CKPT / "run-098-fas-acquisition/ganjoor-vocalized"
DOCS = CKPT / "run-076-fas-diac-retrain"
POETRY_EVAL = CKPT / "run-089-fas-poetry-eval/fas-poetry-eval.jsonl"
BACKBONE = os.environ.get("FAS_BACKBONE", "google/byt5-base")

# ---- run-100 proportion-iteration knobs (disposition 2, relayed) ----
# FAS_GANJOOR_SHARE: the curriculum's ganjoor share target. 1.0 (default)
# = the full post-exclusion ganjoor pool (run-099 behavior). For run-100
# = 0.15: the docs side stays COMPLETE (the register anchor is never
# subsampled) and ganjoor is CAPPED to floor(len(docs) * share/(1-share))
# via a seed-42 sample over the SAME post-exclusion pool — derivation
# (docs 436,599 post-exclusion): ganjoor = floor(436,599 x 0.15/0.85) =
# 77,046 -> combined 513,645, share 77,046/513,645 = 0.149994. Keeping
# the exclusions identical makes PROPORTION the only experimental delta
# vs run-099.
# FAS_DOCS_CLEAN=1: the docs gate measures on the CLEAN instrument
# (wo94-docs-instrument/docs-test-clean, 7,621 rows, sha 61f2b760…)
# instead of the lineage-contaminated 9,288-row test. Threshold
# unchanged (0.0070 row-mean); crown clean reference 0.6527%.
GANJOOR_SHARE = float(os.environ.get("FAS_GANJOOR_SHARE", "1.0"))
DOCS_CLEAN = os.environ.get("FAS_DOCS_CLEAN", "0") == "1"
DOCS_CLEAN_JSONL = CKPT / "wo94-docs-instrument/docs-test-clean/docs-test-clean.jsonl"
DOCS_CLEAN_JSONL_SHA = "61f2b760d5fc5b89569998cc8a2627588495cbdd00d1521d916d61bcc8341dc9"
EPOCHS = int(os.environ.get("FAS_EPOCHS", "3"))
BS = int(os.environ.get("FAS_BS", "16"))
ACCUM = int(os.environ.get("FAS_ACCUM", "2"))
MAXLEN = int(os.environ.get("FAS_MAXLEN", "256"))
LR = float(os.environ.get("FAS_LR", "1e-4"))
WARMUP = int(os.environ.get("FAS_WARMUP", "500"))
CKPT_EVERY = int(os.environ.get("FAS_CKPT_EVERY", "1000"))
DOCS_GATE = 0.0070  # row-mean CER — the WO66 register-hold law (0.70%)

# pinned eval sha256 (verified in-job; the run-089 pin)
POETRY_EVAL_SHA = "94f318f688b556bea7cb6011570c6d64043bf8b38c939e218199426ad49b48e6"
# corpus pins (the models#493 records) — verified in-job via SHA256SUMS.txt
GANJOOR_SUMS_NOTE = "manifest sha256 81a25fe02e058fdc112903e76cfc2ce91da109e6728a2a6004f9407986d0cdfb"


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_pins() -> None:
    """In-job integrity: the pins the corpus records + the eval make."""
    got = sha256_file(POETRY_EVAL)
    if got != POETRY_EVAL_SHA:
        raise SystemExit(f"PIN FAIL: poetry eval sha {got} != {POETRY_EVAL_SHA}")
    log(f"[pins] poetry eval sha ok ({POETRY_EVAL_SHA[:16]}…)")
    sums = (GANJOOR / "SHA256SUMS.txt").read_text(encoding="utf-8")
    for line in sums.splitlines():
        if not line.strip():
            continue
        sha, name = line.split("  ", 1)
        got = sha256_file(GANJOOR / name)
        if got != sha:
            raise SystemExit(f"PIN FAIL: {name} sha {got} != {sha}")
    log("[pins] ganjoor-vocalized SHA256SUMS clean (8 shards + manifest)")
    got = sha256_file(GANJOOR / "manifest.json")
    if got != "81a25fe02e058fdc112903e76cfc2ce91da109e6728a2a6004f9407986d0cdfb":
        raise SystemExit(f"PIN FAIL: ganjoor manifest sha {got}")
    log("[pins] ganjoor manifest sha ok")


def load_pairs(path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            src, tgt = (r.get("src") or "").strip(), (r.get("tgt") or "").strip()
            if src and tgt:
                rows.append((src, tgt))
                if limit and len(rows) >= limit:
                    break
    return rows


def lev(a: str, b: str) -> int:
    try:
        from Levenshtein import distance

        return distance(a, b)
    except ImportError:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


class Rows(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], tok, max_len: int = MAXLEN):
        self.pairs = pairs
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]

    def collate(self, b):
        es = [self.tok(s, truncation=True, max_length=self.max_len).input_ids
              for s, _ in b]
        ds_ = [self.tok(t, truncation=True, max_length=self.max_len).input_ids
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


@torch.no_grad()
def evaluate(model, tok, pairs: list[tuple[str, str]], tag: str,
             bs: int = 16) -> dict:
    model.eval()
    cers, correct = [], 0
    preds: list[str] = []
    for i in range(0, len(pairs), bs):
        batch = [s for s, _ in pairs[i:i + bs]]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAXLEN).to("cuda")
        with torch.autocast("cuda", torch.bfloat16):
            gen = model.generate(**enc, max_new_tokens=MAXLEN, num_beams=1)
        preds.extend(tok.batch_decode(gen, skip_special_tokens=True))
        if (i // bs) % 25 == 0:
            log(f"[eval-{tag}] {i + len(batch)}/{len(pairs)}")
    for (s, gold), pred in zip(pairs, preds):
        cers.append(lev(pred, gold) / max(1, len(gold)))
        if pred.strip() == gold.strip():
            correct += 1
    model.train()
    return {"tag": tag, "rows": len(pairs),
            "cer_row_mean": round(sum(cers) / max(1, len(cers)), 6),
            "word_acc": round(correct / max(1, len(pairs)), 6)}


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"[run] {OUT} backbone={BACKBONE} epochs={EPOCHS} bs={BS}x{ACCUM}")

    verify_pins()

    # ---- curriculum: combined corpus, seed-42 shuffle ----
    # the concat materializes into the RUN dir — the pinned corpus dir
    # stays byte-frozen (the corpus-record pins never see new files)
    concat = OUT / "curriculum-train.jsonl"
    if concat.exists():
        ganjoor_train = load_pairs(concat)
        log(f"[curriculum] loaded ganjoor train concat: {len(ganjoor_train)}")
    else:
        ganjoor_train = []
        for s in range(7):
            ganjoor_train.extend(
                load_pairs(GANJOOR / f"ganjoor-vocalized-{s:05d}-of-00008.jsonl"))
        with concat.open("w", encoding="utf-8") as f:
            for s, t in ganjoor_train:
                f.write(json.dumps({"src": s, "tgt": t}, ensure_ascii=False) + "\n")
        log(f"[curriculum] materialized ganjoor train concat: {len(ganjoor_train)}")
    ganjoor_val = load_pairs(GANJOOR / "ganjoor-vocalized-00007-of-00008.jsonl")
    docs_train = load_pairs(DOCS / "train.jsonl")
    docs_test = load_pairs(DOCS / "test.jsonl")
    poetry_eval = load_pairs(POETRY_EVAL)
    log(f"[data] ganjoor={len(ganjoor_train)} ganjoor_val={len(ganjoor_val)} "
        f"docs={len(docs_train)} docs_test={len(docs_test)} "
        f"poetry_eval={len(poetry_eval)}")
    # ---- eval-exclusion (the pre-registered holdout law, ENFORCED) ----
    # Any curriculum row whose src/tgt (whitespace-stripped) exactly
    # matches a pinned-eval source/target — or a docs-test source/target —
    # is DROPPED and recorded to dropped-rows.jsonl. The run-099 first
    # attempt aborted here: 5 curriculum rows match pinned poetry eval
    # sources (the register corpus predates the eval pin; the
    # acquisition-time decontam covered only the acquired sources).
    eval_srcs = {s.strip() for s, _ in poetry_eval}
    eval_tgts = {t.strip() for _, t in poetry_eval}
    test_srcs = {s.strip() for s, _ in docs_test}
    test_tgts = {t.strip() for _, t in docs_test}

    def _exclude(pairs: list[tuple[str, str]], side: str):
        kept, dropped = [], []
        for s, t in pairs:
            if (s.strip() in eval_srcs or t.strip() in eval_tgts
                    or s.strip() in test_srcs or t.strip() in test_tgts):
                dropped.append({"side": side,
                                "src_sha16": hashlib.sha256(
                                    s.encode()).hexdigest()[:16],
                                "tgt_sha16": hashlib.sha256(
                                    t.encode()).hexdigest()[:16]})
                continue
            kept.append((s, t))
        return kept, dropped

    ganjoor_train, drop_g = _exclude(ganjoor_train, "ganjoor")
    docs_train, drop_d = _exclude(docs_train, "docs")
    dropped = drop_g + drop_d
    if dropped:
        with (OUT / "dropped-rows.jsonl").open("w", encoding="utf-8") as f:
            for d in dropped:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    log(f"[holdout] dropped {len(drop_g)} ganjoor + {len(drop_d)} docs rows "
        f"matching pinned eval/test content (recorded in dropped-rows.jsonl)")

    # ---- proportion knob (run-100): cap ganjoor AFTER the exclusions ----
    if GANJOOR_SHARE < 1.0:
        k = min(len(ganjoor_train),
                int(len(docs_train) * GANJOOR_SHARE / (1 - GANJOOR_SHARE)))
        idx = sorted(random.Random(42).sample(range(len(ganjoor_train)), k))
        ganjoor_train = [ganjoor_train[i] for i in idx]
        log(f"[share] ganjoor subsampled (seed-42) to {k} rows "
            f"(target share {GANJOOR_SHARE})")

    curriculum = docs_train + ganjoor_train
    random.Random(42).shuffle(curriculum)
    log(f"[curriculum] combined={len(curriculum)} "
        f"(ganjoor share {len(ganjoor_train) / len(curriculum):.3f})")

    # ---- holdout assertion: zero residual overlap (hard property) ----
    residual = sum(1 for s, t in curriculum
                   if s.strip() in eval_srcs or t.strip() in eval_tgts)
    if residual:
        raise SystemExit(f"EVAL LEAK persists: {residual} rows remain — "
                         "training aborted")
    log("[holdout] 0 curriculum rows overlap the pinned evals")

    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)

    tok = AutoTokenizer.from_pretrained(BACKBONE)
    model = AutoModelForSeq2SeqLM.from_pretrained(BACKBONE).cuda().train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    ds = Rows(curriculum, tok)
    loader = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=2,
                        collate_fn=ds.collate, drop_last=True)
    steps_per_epoch = len(loader) // ACCUM
    total_steps = steps_per_epoch * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, WARMUP, total_steps)
    log(f"[train] steps/epoch={steps_per_epoch} total={total_steps}")

    # ---- resume (the e12 lesson) ----
    step = 0
    start_ep = 0
    ckpts = sorted(OUT.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    if ckpts:
        resume = ckpts[-1]
        model = AutoModelForSeq2SeqLM.from_pretrained(resume).cuda().train()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        step = int(resume.name.split("-")[1])
        start_ep = step // steps_per_epoch
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        sched = get_cosine_schedule_with_warmup(opt, WARMUP, total_steps)
        for _ in range(step):
            sched.step()
        log(f"[resume] {resume.name} (global step {step}, epoch {start_ep})")

    nan_skips = 0
    t = time.time()
    model.train()
    for _ep in range(start_ep, EPOCHS):
        for ids, am, lb in loader:
            if step >= total_steps:
                break
            ids, am, lb = ids.cuda(), am.cuda(), lb.cuda()
            with torch.autocast("cuda", torch.bfloat16):
                out = model(input_ids=ids, attention_mask=am, labels=lb)
            if not torch.isfinite(out.loss):
                nan_skips += 1
                if nan_skips > 50:
                    raise SystemExit("ABORT: non-finite loss beyond the 50-skip "
                                     "kill criterion (fp32-autocast-off fallback "
                                     "is the single manual relaunch, per plan)")
                opt.zero_grad()
                continue
            (out.loss / ACCUM).backward()
            if (step + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            step += 1
            if step % 200 == 0:
                log(f"[train] {step}/{total_steps} "
                    f"loss={float(out.loss):.3f} "
                    f"({step / max(1, time.time() - t):.2f} it/s)")
            if step % CKPT_EVERY == 0:
                d = OUT / f"step-{step:06d}"
                d.mkdir(exist_ok=True)
                model.save_pretrained(d)
                tok.save_pretrained(d)
                log(f"[ckpt] step-{step:06d} saved")
        ep_dir = OUT / f"epoch-{_ep}"
        ep_dir.mkdir(exist_ok=True)
        model.save_pretrained(ep_dir)
        tok.save_pretrained(ep_dir)
        log(f"[ckpt] epoch-{_ep} saved")

    model.config.use_cache = True

    # ---- final evals (greedy, the fas lineage protocol) ----
    results = {"run": OUT.name, "backbone": BACKBONE,
               "curriculum": {"ganjoor": len(ganjoor_train),
                              "docs": len(docs_train)},
               "epochs": EPOCHS, "steps": step,
               "nan_skips": nan_skips}
    best = OUT / "best"
    best.mkdir(exist_ok=True)
    model.save_pretrained(best)
    tok.save_pretrained(best)
    log("[eval] best checkpoint saved; running the pinned gates")

    # ---- docs gate: the CLEAN instrument (run-100) or the lineage test ----
    if DOCS_CLEAN:
        got = sha256_file(DOCS_CLEAN_JSONL)
        if got != DOCS_CLEAN_JSONL_SHA:
            raise SystemExit(f"PIN FAIL: clean instrument sha {got} != "
                             f"{DOCS_CLEAN_JSONL_SHA}")
        gate_rows = load_pairs(DOCS_CLEAN_JSONL)
        gate_instrument = (f"wo94-docs-instrument/docs-test-clean "
                           f"({len(gate_rows)} rows, sha {DOCS_CLEAN_JSONL_SHA[:16]}…)")
        log(f"[gate] clean instrument rows={len(gate_rows)} (pin ok)")
    else:
        gate_rows = docs_test
        gate_instrument = "run-076 docs test (9,288 rows, lineage-contaminated)"
    docs_res = evaluate(model, tok, gate_rows, "docs")
    results["docs_gate"] = {**docs_res,
                            "instrument": gate_instrument,
                            "gate_row_mean_max": DOCS_GATE,
                            "gate_pass": docs_res["cer_row_mean"] < DOCS_GATE}
    poetry_res = evaluate(model, tok, poetry_eval, "poetry")
    results["poetry_eval"] = {**poetry_res,
                              "baseline": 0.3367,
                              "register_arm_precedent": 0.1655,
                              "run_099_precedent": 0.151375}
    val_res = evaluate(model, tok, ganjoor_val[:5000], "ganjoor-val5k")
    results["ganjoor_val"] = val_res

    verdict = "PASS" if results["docs_gate"]["gate_pass"] else "FAIL"
    results["verdict"] = verdict
    (OUT / "verdict.json").write_text(json.dumps(results, ensure_ascii=False,
                                                 indent=1), encoding="utf-8")
    (OUT / "DONE").write_text("ok\n", encoding="utf-8")
    log("[verdict] " + json.dumps(results, ensure_ascii=False))
    log(f"[done] elapsed={(time.time() - t0) / 3600:.2f}h")


if __name__ == "__main__":
    main()

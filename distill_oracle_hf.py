"""WO39 stage 1 — AR oracle relabel (300K QCRI rows, 3 arms).

Reproduces run-047's corpus recipe (MIX=qcri, GOLD=none, cap 900K,
seed-7 article shuffle, WINDOW=600, strip-alt primary targets),
takes the first 300K shuffled rows, greedy-decodes each src with the
three WO35 arms, picks the per-row oracle (min char edit distance to
stripped silver, ties -> p047), writes arm-preds.jsonl +
oracle-targets.jsonl and prints per-arm/oracle CER + disagreement.

Env: D_RUN (default run-065-oracle-distill), D_SUBSET (default 300000).

Usage (a100-large):
    python /ckpt/run-065-oracle-distill/distill_oracle_hf.py
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

import torch

DATA = Path("/train_data")
CKPT = Path("/ckpt") / os.environ.get("D_RUN", "run-065-oracle-distill")
WINDOW = 600
PSEUDO_CAP = 900_000
SUBSET = int(os.environ.get("D_SUBSET", "300000"))
ARMS = [
    ("p047", "/ckpt/run-047-large-strip/best", 32),
    ("p040", "/ckpt/run-040-strip-alt/best", 64),
    ("p029", "/ckpt/run-029-news-specialist/best", 64),
]
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


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def build_rows() -> list[tuple[str, str]]:
    """run-047's exact qcri-only silver corpus, stripped, shuffled."""
    qpath = DATA / "qcri-wiki.jsonl"
    articles = qpath.read_text(encoding="utf-8").splitlines()
    rng2 = random.Random(7)
    rng2.shuffle(articles)
    pseudo: list[tuple[str, str]] = []

    def add_silver(tgt: str) -> None:
        tgt = tgt.strip()
        src = DIACRITICS_RE.sub("", tgt).strip()
        if len(src) >= 5 and len(src.encode()) <= WINDOW - 50:
            pseudo.append((src, tgt))

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
    stripped = []
    for _src, tgt in pseudo:
        tgt2 = " ".join(t.split("/", 1)[0] for t in tgt.split())
        src2 = DIACRITICS_RE.sub("", tgt2).strip()
        if len(src2) >= 5:
            stripped.append((src2, tgt2))
    rng = random.Random(42)
    rng.shuffle(stripped)
    rows = stripped[:SUBSET]
    print(f"[data] stripped={len(stripped)} subset={len(rows)}", flush=True)
    return rows


def completed_lines(path: Path) -> int:
    if not path.exists():
        return 0
    data = path.read_bytes()
    if not data:
        return 0
    if not data.endswith(b"\n"):
        cut = data.rfind(b"\n")
        with path.open("r+b") as f:
            f.truncate(cut + 1)
        data = data[:cut + 1]
    return data.count(b"\n")


def decode_arm(name: str, ckpt: str, bs: int, srcs: list[str]) -> list[str]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    out = CKPT / f"preds-{name}.jsonl"
    done = completed_lines(out)
    if done:
        print(f"[{name}] resume at row {done}", flush=True)
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16).eval().cuda()
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        with torch.no_grad():
            for i in range(done, len(srcs), bs):
                enc = tok(srcs[i:i + bs], return_tensors="pt", padding=True,
                          truncation=True, max_length=WINDOW).to("cuda")
                gen = model.generate(**enc, max_new_tokens=WINDOW * 2,
                                     num_beams=1)
                for p in tok.batch_decode(gen, skip_special_tokens=True):
                    f.write(json.dumps({"pred": p}, ensure_ascii=False) + "\n")
                f.flush()
                # container death loses appends (3 resume passes lost
                # ~30 GPU-h this way) — force a durable bucket upload
                # every ~4k decoded rows
                if (i // bs) % 125 == 0:
                    try:
                        from huggingface_hub import HfApi
                        HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                            path_or_fileobj=str(out),
                            path_in_repo=str(out.relative_to(Path("/ckpt"))),
                            repo_id="Interscript/isx-training",
                            repo_type=None)
                    except Exception as e:  # noqa: BLE001
                        print(f"[{name}] sync failed: {e}", flush=True)
                if (i // bs) % 50 == 0:
                    r = (i + bs - done) / max(1, time.time() - t0)
                    print(f"[{name}] {i + bs - done}/{len(srcs) - done} "
                          f"({r:.0f} rows/s)", flush=True)
    try:
        from huggingface_hub import HfApi
        HfApi(token=os.environ["HF_TOKEN"]).upload_file(
            path_or_fileobj=str(out),
            path_in_repo=str(out.relative_to(Path("/ckpt"))),
            repo_id="Interscript/isx-training",
            repo_type=None)
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] final sync failed: {e}", flush=True)
    return [json.loads(l)["pred"] for l in
            out.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    srcs = [s for s, _ in rows]
    silvers = [t for _, t in rows]

    preds: dict[str, list[str]] = {}
    for name, ckpt, bs in ARMS:
        # always call decode_arm: it resumes at the synced line count
        # internally and appends the remainder (a file-exists skip here
        # silently shipped partial arms into the consolidation)
        preds[name] = decode_arm(name, ckpt, bs, srcs)

    with (CKPT / "arm-preds.jsonl").open("w", encoding="utf-8") as f, \
         (CKPT / "oracle-targets.jsonl").open("w", encoding="utf-8") as g:
        cers = {n: 0.0 for n, _, _ in ARMS}
        oracle_cer = 0.0
        disagree = 0
        for i, (src, silver) in enumerate(rows):
            row_preds = {n: preds[n][i] for n, _, _ in ARMS}
            if len({row_preds[n] for n, _, _ in ARMS}) > 1:
                disagree += 1
            best_name, best_tgt = "p047", row_preds["p047"]
            best_d = lev(row_preds["p047"], silver)
            for n, _, _ in ARMS[1:]:
                d = lev(row_preds[n], silver)
                if d < best_d:
                    best_name, best_d, best_tgt = n, d, row_preds[n]
            for n in cers:
                cers[n] += lev(row_preds[n], silver) / max(1, len(silver))
            oracle_cer += best_d / max(1, len(silver))
            f.write(json.dumps({"src": src, "silver": silver, **row_preds},
                               ensure_ascii=False) + "\n")
            g.write(json.dumps({"src": src, "tgt": best_tgt,
                                "pick": best_name},
                               ensure_ascii=False) + "\n")
    n = len(rows)
    stats = {"rows": n, "disagreement": round(disagree / n, 4),
             "arm_cer": {k: round(v / n, 4) for k, v in cers.items()},
             "oracle_cer": round(oracle_cer / n, 4),
             "picks": {}, "silver_win": 0}
    picks = [json.loads(l)["pick"] for l in
             (CKPT / "oracle-targets.jsonl").read_text(
                 encoding="utf-8").splitlines()]
    for p in picks:
        stats["picks"][p] = stats["picks"].get(p, 0) + 1
    print(json.dumps(stats), flush=True)
    (CKPT / "relabel_stats.json").write_text(json.dumps(stats, indent=1))
    (CKPT / "RELABEL_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

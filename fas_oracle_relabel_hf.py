"""WO54 — fas multi-arm oracle relabel (the WO39 pattern, fas edition).

Owner item: improve fas-diac beyond run-077's verified 0.603%
row-mean. The proven lever (WO39: oracle CER −25% vs best arm):
greedy-decode each train src with N fas checkpoints, pick the per-row
oracle (min char edit distance to the deterministic gold, ties ->
run-077), write oracle-targets.jsonl for a stage-2 continue-train.

Arms: run-076 (small 3ep) + run-077 (small 6ep, the RC) + run-079
(base docs-LR) — cross-architecture disagreement is the signal.
Subset 120K: ~22h of decode on one a100 (bs 64 small / 32 base),
inside the 23h timeout; durable HfFileSystem sync every ~4k rows.

Env: F_ORACLE_RUN (default run-080-fas-oracle), F_SUBSET (120000).

Usage (a100-large, mounts isx-training:/ckpt:rw):
    python /ckpt/run-080-fas-oracle/fas_oracle_relabel_hf.py
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import torch

try:
    from Levenshtein import distance as _lev_c

    def lev(a: str, b: str) -> int:
        return _lev_c(a, b)
except ImportError:  # pure-python fallback; jobs pip-install the C wheel
    def lev(a: str, b: str) -> int:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                               prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

CKPT = Path("/ckpt") / os.environ.get("F_ORACLE_RUN", "run-080-fas-oracle")
CORPUS = Path("/ckpt/run-076-fas-diac-retrain/train.jsonl")
SUBSET = int(os.environ.get("F_SUBSET", "120000"))
MAXLEN = 256
ARMS = [
    ("p076", "/ckpt/run-076-fas-diac-retrain/best", 64),
    ("p077", "/ckpt/run-077-fas-diac-epochs/best", 64),
    ("p079", "/ckpt/run-079-fas-diac-docslr/best", 32),
]
TIE_BREAK = "p077"


def sync_to_bucket(path: Path) -> None:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    dst = f"buckets/Interscript/isx-training/{path.relative_to(Path('/ckpt'))}"
    with open(path, "rb") as f, fs.open(dst, "wb") as g:
        g.write(f.read())
    if not fs.exists(dst):
        raise RuntimeError(f"sync verify failed: {dst}")


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


def build_rows() -> list[tuple[str, str]]:
    rows = [json.loads(l) for l in CORPUS.read_text(
        encoding="utf-8").splitlines() if l.strip()]
    rows = [(r["src"].strip(), r["tgt"].strip()) for r in rows
            if r.get("src", "").strip() and r.get("tgt", "").strip()]
    rng = random.Random(7)
    rng.shuffle(rows)
    rows = rows[:SUBSET]
    print(f"[data] gold rows={len(rows)}", flush=True)
    return rows


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
                          truncation=True, max_length=MAXLEN).to("cuda")
                gen = model.generate(**enc, max_new_tokens=MAXLEN,
                                     num_beams=1)
                for p in tok.batch_decode(gen, skip_special_tokens=True):
                    f.write(json.dumps({"pred": p}, ensure_ascii=False) + "\n")
                f.flush()
                if (i // bs) % 125 == 0:
                    try:
                        sync_to_bucket(out)
                    except Exception as e:  # noqa: BLE001
                        print(f"[{name}] sync failed: {e}", flush=True)
                if (i // bs) % 50 == 0:
                    r = (i + bs - done) / max(1, time.time() - t0)
                    print(f"[{name}] {i + bs - done}/{len(srcs) - done} "
                          f"({r:.0f} rows/s)", flush=True)
    try:
        sync_to_bucket(out)
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] final sync failed: {e}", flush=True)
    return [json.loads(l)["pred"] for l in
            out.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    srcs = [s for s, _ in rows]
    golds = [t for _, t in rows]

    preds: dict[str, list[str]] = {}
    for name, ckpt, bs in ARMS:
        preds[name] = decode_arm(name, ckpt, bs, srcs)

    with (CKPT / "arm-preds.jsonl").open("w", encoding="utf-8") as f, \
         (CKPT / "oracle-targets.jsonl").open("w", encoding="utf-8") as g:
        cers = {n: 0.0 for n, _, _ in ARMS}
        oracle_cer = 0.0
        disagree = 0
        for i, (src, gold) in enumerate(rows):
            row_preds = {n: preds[n][i] for n, _, _ in ARMS}
            if len(set(row_preds.values())) > 1:
                disagree += 1
            best_name, best_d = TIE_BREAK, lev(row_preds[TIE_BREAK], gold)
            for n, _, _ in ARMS:
                d = lev(row_preds[n], gold)
                cers[n] += d / max(1, len(gold))
                if d < best_d:
                    best_name, best_d = n, d
            best_tgt = row_preds[best_name]
            oracle_cer += best_d / max(1, len(gold))
            f.write(json.dumps({"src": src, "gold": gold, **row_preds},
                               ensure_ascii=False) + "\n")
            g.write(json.dumps({"src": src, "tgt": best_tgt,
                                "pick": best_name},
                               ensure_ascii=False) + "\n")
    n = len(rows)
    stats = {"rows": n, "disagreement": round(disagree / n, 4),
             "arm_cer": {k: round(v / n, 5) for k, v in cers.items()},
             "oracle_cer": round(oracle_cer / n, 5),
             "picks": {}, "convention": "row-mean"}
    picks = [json.loads(l)["pick"] for l in
             (CKPT / "oracle-targets.jsonl").read_text(
                 encoding="utf-8").splitlines()]
    for p in picks:
        stats["picks"][p] = stats["picks"].get(p, 0) + 1
    print(json.dumps(stats), flush=True)
    (CKPT / "relabel_stats.json").write_text(json.dumps(stats, indent=1))
    (CKPT / "RELABEL_DONE").write_text("ok\n")
    try:
        for f in ("relabel_stats.json", "RELABEL_DONE"):
            sync_to_bucket(CKPT / f)
    except Exception as e:  # noqa: BLE001
        print(f"final sync failed: {e}", flush=True)


if __name__ == "__main__":
    main()

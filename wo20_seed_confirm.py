# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "pandas",
#   "pyarrow",
#   "pyarabic",
#   "prettytable",
#   "tqdm",
# ]
# ///
"""WO20u M1 — multi-seed confirmation scorer (issue #441 follow-up).

Scores one WO20 bridge-arm seed run's banked student predictions against
the G2a frontier rung (run-007-r7-muon-6ep, released predictions in
`frontier-predictions-v1`) under the IDENTICAL evaluator and bootstrap
math the seed-1 verdict used:

  - full 1,200-paragraph SadeedDiac-25, windowed zero-skip protocol
    (preds are already window-joined paragraphs banked by the trainer's
    eval epilogue);
  - sadeed_evaluator, `gt_missing_diacritic_is_error=False`;
  - word-count-mismatch paragraphs skipped PAIRWISE (the evaluator
    raises on length mismatch — the skip is its aggregate behavior;
    seed 1: 59 of 1,200 skipped);
  - count-weighted paired bootstrap, documents = resampling unit,
    10k resamples, seed 7 — delta = seed - G2a (negative = seed better).

Also re-scores the seed-1 banked preds when pointed at
WO20_RUN=wo20-tiny-g2w — the calibration that makes seeds 2/3
comparable to the published −0.4385pp verdict.

Usage (cpu-basic; resume-safe via the bucket output marker):
    hf jobs uv run --flavor cpu-basic -d --timeout 4h \\
      -e WO20_RUN=wo20-tiny-g2w-s2 \\
      --secrets HF_TOKEN \\
      -v hf://datasets/Interscript/arabic-r8:/train_data:ro \\
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \\
      -v /tmp/isx-wo20u-code:/code:ro \\
      wo20_seed_confirm.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, "/code")

RUN = os.environ.get("WO20_RUN", "wo20-tiny-g2w-s2")
CKPT = Path("/ckpt") / RUN
OUT = Path("/ckpt") / os.environ.get("WO20U_OUT_DIR", "wo20u-m1")
FPV1 = ("https://github.com/interscript/interscript-models/"
        "releases/download/frontier-predictions-v1")
G2A_PREDS = "run-007-r7-muon-6ep-final_preds.jsonl"
G2A_SHA = "f6f606195144e3e17fdc85eb558ae7d58e30b8c14876e7c26456575bf238e0d7"
BOOT = 10000
BOOT_SEED = 7


def log(*a):
    print("[wo20u-m1]", *a, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"GET {url}")
    with urllib.request.urlopen(url) as r, dest.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


def load_seed_preds(path: Path) -> dict[int, str]:
    preds = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        i = row.get("src", row.get("idx"))
        preds[int(i)] = row.get("student") or row.get("pred") or ""
    return preds


def paired_bootstrap(der_a: np.ndarray, der_b: np.ndarray,
                     counts: np.ndarray, boot: int = BOOT,
                     seed: int = BOOT_SEED):
    """Documents as resampling unit (the trainer's exact math,
    count-weighted). Returns (delta = b - a, lo, hi, p_delta_gt_0)."""
    keep = counts > 0
    a = np.asarray(der_a, np.float64)[keep] * counts[keep]
    b = np.asarray(der_b, np.float64)[keep] * counts[keep]
    c = counts[keep].astype(np.float64)
    delta = b.sum() / c.sum() - a.sum() / c.sum()
    rng = np.random.default_rng(seed)
    n = len(c)
    idx = rng.integers(0, n, size=(boot, n))
    da = a[idx].sum(axis=1)
    db = b[idx].sum(axis=1)
    ct = c[idx].sum(axis=1)
    deltas = (db - da) / ct
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = float((deltas > 0).mean())
    return float(delta), float(lo), float(hi), p


def main():
    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    marker = OUT / f"{RUN}.confirm.json"
    if marker.exists():
        log(f"already scored (bucket marker) — {marker}")
        print(marker.read_text())
        return

    preds_path = CKPT / "student_preds.jsonl"
    if not preds_path.exists():
        raise SystemExit(f"no banked student preds at {preds_path} — "
                         "the seed's eval epilogue has not completed")
    seed_preds = load_seed_preds(preds_path)
    log(f"seed preds: {len(seed_preds)} paragraphs")

    g2a_path = fetch(f"{FPV1}/{G2A_PREDS}", Path("/tmp") / G2A_PREDS)
    got = sha256_file(g2a_path)
    if got != G2A_SHA:
        raise SystemExit(f"sha mismatch: {G2A_PREDS} {got}")
    g2a_rows = {}
    for line in g2a_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        i = row.get("idx", row.get("src"))
        g2a_rows[int(i)] = row.get("student") or ""
    log(f"G2a preds: {len(g2a_rows)} paragraphs (sha {got[:12]} verified)")

    import pyarrow.parquet as pq
    table = pq.read_table(Path("/train_data") / "sadeeddiac-25.parquet")
    gold = table.column("output").to_pylist()
    log(f"gold paragraphs: {len(gold)}")

    ev = E()
    common = sorted(set(seed_preds) & set(g2a_rows)
                    & set(range(len(gold))))
    der_seed = np.zeros(len(common))
    der_g2a = np.zeros(len(common))
    letters = np.zeros(len(common))
    words = np.zeros(len(common))
    n_skip_seed = n_skip_g2a = 0
    for j, i in enumerate(common):
        gt = gold[i]
        counts = {}
        for name, pred in (("seed", seed_preds[i]), ("g2a", g2a_rows[i])):
            try:
                wc, lc, _, _, _, sd, _ = \
                    ev.caculate_error_on_single_sentence(
                        pred, gt, gt_missing_diacritic_is_error=False)
                counts[name] = (wc, lc, sd)
            except RuntimeError:
                counts[name] = None
        if counts["seed"] is None:
            n_skip_seed += 1
        if counts["g2a"] is None:
            n_skip_g2a += 1
        if counts["seed"] is None or counts["g2a"] is None:
            continue  # pairwise skip on word-count mismatch
        wc, lc, sd = counts["seed"]
        words[j], letters[j] = wc, lc
        der_seed[j] = 100.0 * sd / max(lc, 1)
        wc2, lc2, sd2 = counts["g2a"]
        der_g2a[j] = 100.0 * sd2 / max(lc2, 1)

    scorable = letters > 0
    n_scorable = int(scorable.sum())
    # der_* arrays are already in percentage points (per-paragraph
    # DER-CE); the aggregate is their letter-count-weighted mean.
    seed_total = float((der_seed[scorable]
                        * letters[scorable]).sum()
                       / letters[scorable].sum())
    g2a_total = float((der_g2a[scorable]
                       * letters[scorable]).sum()
                      / letters[scorable].sum())
    delta, lo, hi, p = paired_bootstrap(der_g2a, der_seed, letters)
    separated = hi < 0  # CI entirely negative: separated improvement

    result = {
        "stage": "wo20u-m1-confirm",
        "run": RUN,
        "protocol": ("full SadeedDiac-25, windowed zero-skip, paired "
                     "bootstrap vs G2a released preds "
                     "(frontier-predictions-v1 sha-verified), documents "
                     "= resampling unit, count-weighted, 10k, seed 7"),
        "n_scorable_pairs": n_scorable,
        "skipped_seed_wordcount_mismatch": n_skip_seed,
        "skipped_g2a_wordcount_mismatch": n_skip_g2a,
        "g2a_rescored_der_ce": round(g2a_total, 4),
        "seed_der_ce_scorable": round(seed_total, 4),
        "paired_delta_pp_seed_minus_g2a": round(delta, 4),
        "paired_ci95": [round(lo, 4), round(hi, 4)],
        "p_seed_worse": p,
        "separated_improvement": separated,
        "g2a_preds_sha256": got,
    }
    log("RESULT " + json.dumps(result))
    OUT.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

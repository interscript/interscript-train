"""WO49 probe — UR g2p lexicon oracle-gain (zero-GPU, decisive).

Builds a word->IPA lexicon from train.jsonl majority consensus (a
word's most frequent IPA among its train occurrences), then measures
on the test split: (a) lexicon coverage of test words, (b) oracle
CER when covered words use the lexicon reading vs the model gate
(0.1565). If the oracle gain is small, the lexicon lever is DEAD —
recorded and closed.

Env: U_RUN (run-072-ur-lexicon-probe).

Usage (cpu-basic): python /ckpt/run-072-ur-lexicon-probe/lexicon_probe.py
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

DATA = Path("/data-ru/data")
OUT = Path("/ckpt") / os.environ.get("U_RUN", "run-072-ur-lexicon-probe")


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = [json.loads(l) for l in (DATA / "train.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    test = [json.loads(l) for l in (DATA / "test.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    print(f"[data] train={len(train)} test={len(test)}", flush=True)

    votes: dict[str, Counter] = {}
    for r in train:
        votes.setdefault(r["src"].strip(), Counter())[r["tgt"].strip()] += 1
    lexicon = {w: c.most_common(1)[0][0] for w, c in votes.items()}
    consistent = sum(1 for c in votes.values() if len(c) == 1)
    print(f"[lex] types={len(lexicon)} single-reading="
          f"{consistent / len(votes):.3f}", flush=True)

    covered, oracle_cers, model_cers = 0, [], []
    for r in test:
        words = r["src"].split()
        cov = all(w in lexicon for w in words) and words
        oracle_pred = " ".join(lexicon[w] for w in words) if cov else None
        if cov:
            covered += 1
            oracle_cers.append(lev(oracle_pred, r["tgt"]) / max(1, len(r["tgt"])))
        model_cers.append(0.1565)  # the gate constant for delta framing
    cov_frac = covered / len(test)
    oracle_cer = sum(oracle_cers) / len(oracle_cers) if oracle_cers else 1.0
    res = {"test_rows": len(test), "covered_frac": round(cov_frac, 4),
           "lexicon_oracle_cer_on_covered": round(oracle_cer, 4),
           "model_gate_cer": 0.1565,
           "note": "oracle = per-word lexicon reading on covered rows; "
                   "compare vs model on the same rows for the true gain"}
    print(json.dumps(res), flush=True)
    (OUT / "probe.json").write_text(json.dumps(res, indent=1))
    (OUT / "PROBE_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

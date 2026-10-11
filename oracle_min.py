"""Oracle-min complementarity probe for WikiNews-2024 multiref.

Mirrors eval_wikinews_multiref.score exactly, but takes two prediction
sets: a word counts as correct if EITHER arm matches an alternate.
Reports each arm solo plus the oracle ceiling.

Usage (with the scorer on path):
    from oracle_min import oracle_score
    oracle_score(preds_a, preds_b, refs)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eval_wikinews_multiref import get_diac_codes, remove_default_diac, word_matches


def _word_credit(pred_w: str, alts: list[str], skip_last: bool):
    scored = [word_matches(pred_w, a, skip_last) for a in alts]
    matched = next((x for x in scored if x[0]), None)
    if matched is not None:
        return True, matched[1], matched[2]
    best = max(scored, key=lambda x: x[1]) if scored else (False, 0, 0)
    return False, best[1], best[2]


def oracle_score(*pred_sets: list[list[str] | str], refs: list[str],
                 skip_last: bool = False) -> dict:
    """N-arm oracle: any arm matching any alternate counts correct.
    Call as oracle_score(preds_a, preds_b, ..., refs=refs)."""
    arms = list(pred_sets)
    n_arms = len(arms)
    tot_w = 0
    c_w = [0] * n_arms
    c_wo = 0
    tot_l = 0
    c_l = [0] * n_arms
    c_lo = 0
    for row in zip(*arms, refs):
        ref = row[-1]
        preds = row[:-1]
        rw = ref.split()
        tot_w += len(rw)
        toks = [p.split() for p in preds]
        for k, ref_w in enumerate(rw):
            alts = ref_w.split("/") if "/" in ref_w else [ref_w]
            credits = []
            for i, t in enumerate(toks):
                pw = t[k] if k < len(t) else None
                credits.append(_word_credit(pw, alts, skip_last) if pw
                               else (False, 0, 0))
            for i, cr in enumerate(credits):
                c_w[i] += cr[0]
                c_l[i] += cr[1]
            c_wo += any(cr[0] for cr in credits)
            c_lo += max(cr[1] for cr in credits)
            present = [cr for cr, t in zip(credits, toks) if k < len(t)]
            tot_l += (present[0][2] if present
                      else len(get_diac_codes(remove_default_diac(alts[0]))))
    n = max(1, tot_w)
    ln = max(1, tot_l)
    out = {"words": tot_w, "letters": tot_l}
    for i in range(n_arms):
        out[f"WER_{i+1}"] = round(100 * (1 - c_w[i] / n), 4)
    out["WER_oracle"] = round(100 * (1 - c_wo / n), 4)
    for i in range(n_arms):
        out[f"DER_{i+1}"] = round(100 * (1 - c_l[i] / ln), 4)
    out["DER_oracle"] = round(100 * (1 - c_lo / ln), 4)
    return out

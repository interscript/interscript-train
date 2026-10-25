# 04 — Cross-abugida transfer (Thai → Khmer → Lao)

## Goal
Exploit script-family structure: fine-tune/zero-shot the Thai stack for
Khmer (17.9K pairs) and source Lao data for a third point.

## Why groundbreaking
Thai and Khmer share abugida structure (inherent vowels, no spaces,
register/tone complexity). Our Thai evidence: pretrained beats scratch by
10pts; epitran augmentation works. Does *cross-script* pretransfer beat
*in-family* pretraining? Unknown for G2P.

## Steps
1. Baseline (running): Khmer ByT5 from scratch on 17.9K pairs.
2. Khmer: continue-train from the best Thai checkpoint (input-side is
   different script — byte-level model handles it; question is whether
   phonological abugida representations transfer).
3. Optional: transliterate Thai training data into Khmer-aligned
   representation (shared IPA targets!) — the IPA target space IS the
   bridge: Thai→IPA and Khmer→Latin share Latin output space.
4. Lao: source pairs (epitran lao-Latn + Wikipedia), zero-shot then few-shot.

## Success metric
Khmer EM/CER: transfer ≥ 2pt improvement over the from-scratch baseline.

## Commonality
Abugida family; deterministic teachers (epitran tha/lao; none for Khmer —
that gap is itself a finding).

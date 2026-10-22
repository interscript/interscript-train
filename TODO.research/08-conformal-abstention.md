# 08 — Conformal abstention + map fallback runtime

## Goal
Per-word calibrated confidence for the vocalizers; below threshold, the TS
runtime falls back to the deterministic ISC map path (or flags the word).
Distribution-free guarantees via split-conformal prediction.

## Why groundbreaking
interscript.org is a determinism product. "Neural model that can say I don't
know, then defer to rules" is the missing deployment primitive for ML in
transliteration runtimes — and our fallback target (the maps) actually exists,
unlike in classic abstention work.

## Steps
1. Calibrate on held-out: nonconformity = per-word edit distance /
   negative sequence log-prob; pick threshold for 95% coverage.
2. Emit per-word abstain flags in the ONNX/TS inference wrapper.
3. Runtime policy: abstain → bibliographic map transliteration of that word
   (encoding-safe) or markup `<uncertain>`.
4. Measure coverage vs end-to-end error trade-off curve.

## Success metric
95% word-level coverage with measurable end-to-end error reduction over
always-neural; zero silent failures.

## Commonality
Cross-family; pairs with #01 (map fallback needs the readiness matrix).

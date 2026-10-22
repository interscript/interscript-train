# 07 — Protocol cards + two-way benchmark harness

## Goal
Standard + harness: every diacritization/G2P system × every benchmark,
with beam width, domain, and representation stated. One command runs one
system on one benchmark with a protocol card emitted.

## Why groundbreaking
Our documented effect sizes are larger than most modeling choices:
- Hebrew domain: DictaBERT 35.6% Biblical vs ~4% modern (9×)
- Persian representation: ezafe handling shifts HA by 18 points
- Hebrew beam: 12 DER points
- Arabic benchmark: same model 0.99% (our split) vs 2.93% (SadeedDiac-25)
Cross-paper comparisons are currently meaningless; this is a meta-paper.

## Steps
1. Extract `run(system_ckpt, benchmark, protocol) -> card` harness in rababa.
2. Protocol card schema: test set, preprocessing, beam, normalization,
   skipped-example policy (the 326-skip artifact is itself a finding).
3. Populate matrix: ours + Sadeed + DictaBERT + Nakdimon + Homo-GE2PE +
   epitran baselines.
4. Write the meta-analysis section for the umbrella paper.

## Success metric
Published matrix; harness adopted as the repo eval entrypoint.

## Status
First cells already run (SadeedDiac-25 with Misraj's own evaluator).

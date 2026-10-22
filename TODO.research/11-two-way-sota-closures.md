# 11 — Two-way SOTA closures (in-flight executions)

Live executions closing every "are we beating SOTA" question with the
other side's own protocol. Update docs/RESULTS.md + papers as each lands.

## Persian vs Homo-GE2PE (76.89% SentenceBench HA)
- [x] v1 plain-Phoneme model: 59.11% exact / 77.34% ezafe-norm
- [ ] v3 dual-task (plain + Mapped/GE2PE repr), 6 epochs, beam 4 — TRAINING
- [ ] exact-target: train on their repr so no normalization needed
- [ ] evaluate their released Homo-GE2PE / HomoFast on OUR split (other
      direction of the two-way protocol)
- [ ] target: beat 76.89% exact, ≥82% normalized

## Arabic vs Sadeed (7.29% DER CE / 5.26% w/o CE on SadeedDiac-25)
- [x] our ~10M model with Misraj's own evaluator, full 1,200 paragraphs,
      zero skips: **3.25% CE / 1.81% w/o CE** — beats Sadeed 2.2×/2.9×,
      beats GPT-4, splits with Gemini-Flash; Claude-3.7 (1.39/0.77) ahead
- [x] RESULTS.md + arabic/umbrella papers updated and recompiled
- [ ] trie-constrained decoding pass on the same benchmark
- [ ] optional: evaluate Sadeed checkpoint on OUR cleaned split
- [ ] target: <1.5% CE → beats everything incl. Claude, hands down

## Urdu
- [x] no learned SOTA exists; epitran 60.0% vs ours 14.77% (4.1×)
- [ ] cross-abjad transfer (#03) for the diacritization side

## Khmer
- [ ] baseline ByT5 run (in flight) vs legacy net-500-epochs.pth
- [ ] cycle-consistency (#09) as the improvement path

## Thai
- [x] 2.32% PER vs public baseline 6.37% (same test)

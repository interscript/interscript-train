# 09 — Cycle-consistency self-training (Khmer first)

## Goal
Break the 17.9K-pair Khmer ceiling with unlabeled data: forward model
(Khmer→Latin) + backward model (Latin→Khmer); accept pseudo-pairs where
round-trip(backward(forward(x))) ≈ x on unlabeled Khmer Wikipedia.

## Why groundbreaking
Back-translation is the one augmentation we have NOT tried; Khmer is the
ideal test case (tiny labeled set, abundant unlabeled text, and the legacy
net in secryst/data-khmer-translit gives us a 2021 comparison point).

## Steps
1. Baseline: ByT5 from-scratch on 17.9K pairs (running).
2. Train backward Latin→Khmer on the same pairs.
3. Scrape/fetch Khmer Wikipedia dumps; filter Khmer-script ratio + length.
4. Generate forward pseudo-labels; keep top round-trip agreements
   (edit-similarity ≥ 0.95); retrain with mixed real+pseudo.
5. Iterate ×3 (self-training rounds); watch for drift (oracle test set).

## Success metric
Khmer EM +5pts / CER halved vs baseline; no degradation on the held-out
test from the baseline split.

## Commonality
Abugida family; same recipe transfers to Thai/Lao if it works.

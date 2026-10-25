# 06 — Teacher-error structure theory

## Goal
Formalize and test: **a teacher's labels are useful iff its errors are
predictable from the input** (systematic bias), not iff it is accurate.
Deterministic teachers (epitran, ISC maps, graph rules) qualify; LLM
teachers do not (hallucination = input-independent noise).

## Why groundbreaking
We hold unique evidence: Thai epitran augmentation (3.24→2.32 PER),
Urdu IPA-derived labels (3.74% CER from "visibly lossy" conversion),
the 2021 VOA graph-teacher run, and the LLM-hallucination counterfactual.
Nobody has stated or measured the criterion.

## Method
1. Define teacher error predictability: train a small probe to predict
   teacher-vs-gold divergence from input features; measure R².
2. Compute for: epitran (th, ur), IPA→haraqat converter, graph rules,
   an LLM baseline (tone/haraqat labeling).
3. Correlate predictability with downstream student gain across our runs.
4. Derive when distillation-from-teacher beats no-teacher.

## Success metric
A paper section with the predictability-vs-gain plot across ≥4 teachers,
plus the negative LLM case. Feeds TODO.publish papers.

## Commonality
Cross-family; explains why our augmentation worked where ensembles failed.

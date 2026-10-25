# 10 — Weak-supervision ladder formalization

## Goal
Formalize the pattern we used three times: derive labels from a dataset's
AUXILIARY field via a deterministic converter, scale, train.

| Language | Gold? | Auxiliary field | Derived labels | Result |
|---|---|---|---|---|
| Persian | none | HomoRich `Mapped Phoneme` | 464K haraqat pairs | 0.52% CER |
| Urdu | none | G2P dictionary IPA | 597K haraqat pairs | 3.74% CER |
| Thai | 10K dict | unlabeled Wikipedia + epitran | 50K phonemized | 2.32% PER |

## Why groundbreaking
"Derived labels + scale beats no labels" is stated nowhere as a method;
each paper treats its own trick ad hoc. The ladder (gold > curated >
derived > deterministic-teacher > LLM-teacher✗) with cost/quality per
rung is a citable framework, and we have the evidence table.

## Steps
1. Write the ladder formally (label provenance × scale × noise structure).
2. Add the missing rungs' data: quantify noise rate per converter against
   small human-audited samples (200 words each).
3. Cross-ref #06 (teacher-error structure) — same theory, data view.

## Success metric
Umbrella-paper section + reusable converters documented in each repo.

## Commonality
Cross-family; this IS the shared method behind Persian/Urdu/Thai results.

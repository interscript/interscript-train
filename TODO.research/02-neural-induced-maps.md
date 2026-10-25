# 02 — Neural-induced maps (rule extraction)

## Goal
Invert the 2021 "learn graphs" direction: distill our vocalizer+map
composition back into **explicit, auditable context rules** and fold them
into the ISC map source.

## Why groundbreaking
"A student is only justified for rules too large to author by hand" (user,
2026-08-14). Then the highest-value use of the student is as a *rule
discovery tool*: surface the hidden context-dependent rules it learned,
let a linguist review them, and add them to the map. Auditable neural
transliteration — nobody can currently inspect why a ByT5 chose a
romanization.

## Method
1. Run the vocalizer over a large corpus (per language).
2. Align model output ↔ map-expected vocalization; collect divergences.
3. Cluster divergences by (grapheme, left/right context, POS-proxy).
4. Emit candidate ISC rule snippets (context-dependent subst rules) with
   coverage/confidence stats.
5. Linguist review → PR into interscript/maps.

## Success metric
For Arabic trie-lexicon already built: rules recovered by extraction should
match/beat the hand-built lexicon's DER contribution. First target: recover
X% of trie-constrained decoding's gain with explicit rules alone.

## Prior art
interscript/transliteration-learner-from-graphs (2021, expert→code);
we do model→rules.

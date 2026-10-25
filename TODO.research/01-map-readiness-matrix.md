# 01 — Map-readiness matrix

## Goal
Measure, for every ISC map that targets one of our five languages, the
end-to-end error of running **raw (unvocalized) text → neural vocalizer →
ISC map → Latin**, against the reference **human-vocalized → map → Latin**.

## Why groundbreaking
Nobody reports transliteration quality end-to-end from raw input. Published
transliteration eval assumes vocalized input; published diacritization eval
stops at haraqat. The delta between the two pipelines is exactly the cost of
the neural layer — the number interscript.org actually ships against.

## Two-family scoping
- Phonological maps (RTGS, BGN/PCGN Arabic, ALA-LC Hebrew): full pipeline.
- Bibliographic maps (ISO 9 etc.): map alone is near-total on raw script —
  measure as the control/baseline; expected ≈ exact. Any error there is map
  coverage, not model.

## Steps
1. Enumerate ISC maps per language from interscript/maps (289 total; select
   the subset for ar/he/fa/ur/th/kh scripts).
2. Build corpus pairs: sentences with human vocalization (SadeedDiac-25,
   Sefaria, HomoRich-derived, Kaikki Thai).
3. Run map on (a) human-vocalized gold, (b) model-vocalized output.
4. Report per-map: edit distance between the two Latin outputs; bucket by
   vocalizer DER/CER.

## Success metric
A public table: map × language × vocalizer → end-to-end error. Identifies
which maps are production-ready on raw input today.

## Repos
rababa (vocalizers), interscript/maps (maps), interscript-ruby (runtime).

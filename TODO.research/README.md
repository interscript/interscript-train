# TODO.research — groundbreaking & SOTA program

Conceptual frame (corrected 2026-08-14): ISC maps are script→script systems in
**two families**, never conflated:

1. **Bibliographic/encoding** (ISO 9, most ALA-LC): total over *raw* script;
   meant for identification, not pronunciation. No vocalizer needed. A student
   model here is only about runtime/coverage — not indeterminism.
2. **Phonological** (RTGS Thai, BGN/PCGN Arabic, ALA-LC Hebrew): assumes
   vocalized input. THIS is where our neural vocalizers plug in:
   `(vocalize) ∘ (map)`.

Language commonalities (the leverage points):

- **Abjad family** — Arabic, Hebrew, Persian, Urdu: short-vowel restoration,
  homograph disambiguation, lexicon/trie decoding, DER-family metrics, beam
  effects, ezafe (Fa/Ur). One machinery, four corpora.
- **Abugida family** — Thai, Khmer, (Lao): inherent vowels, tones, no word
  segmentation, deterministic-phonemizer augmentation, ByT5 > CTC evidence.
- **Cross-cutting**: byte-level deployment (ByT5 tokenizer = TextEncoder),
  deterministic teachers (never LLM), weak supervision from auxiliary dataset
  fields, two-way protocol discipline.

| # | TODO | Type | Family |
|---|---|---|---|
| 01 | Map-readiness matrix (end-to-end vocalize→map error) | experiment | cross |
| 02 | Neural-induced maps (extract rules back into ISC) | research | cross |
| 03 | Cross-abjad multilingual diacritizer + transfer | experiment | abjad |
| 04 | Cross-abugida transfer (Thai→Khmer→Lao) | experiment | abugida |
| 05 | Homograph retrieval (RAG) module | build | abjad |
| 06 | Teacher-error structure theory | paper | cross |
| 07 | Protocol cards + benchmark harness | paper/build | cross |
| 08 | Conformal abstention + map-fallback runtime | build | cross |
| 09 | Cycle-consistency self-training (Khmer) | experiment | abugida |
| 10 | Weak-supervision ladder formalization | paper | cross |
| 11 | Two-way SOTA closures (in-flight executions) | execution | all |

In-flight as of writing: Khmer ByT5 (task #324), Persian v3 dual-task (#326),
Arabic SadeedDiac-25 rerun (#327).

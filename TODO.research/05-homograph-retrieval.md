# 05 — Homograph retrieval (RAG) for the abjad family

## Goal
Shared inference-time module: for each ambiguous word, retrieve the k-nearest
annotated contexts from a lexicon of (word, context, vocalization) and feed
them as augmentation to the vocalizer.

## Why groundbreaking
Our SentenceBench analysis quantified the ceiling: 50/212 reference
pronunciations never occur in HomoRich's homograph annotations — pure
generalization gap. Retrieval attacks exactly this; no published G2P system
retrieves lexicon evidence at inference.

## Steps
1. Build homograph lexicons: HomoRich homograph columns (Persian), Sefaria
   + Dicta (Hebrew), Tashkeela-derived (Arabic).
2. Index contexts (embedding or BM25 — byte-friendly) per homograph type.
3. Input format: `[evidence] + sentence` or evidence-as-prefix tokens.
4. Measure HA/DER with retrieval on/off; ablate evidence count k.

## Success metric
Persian SentenceBench HA +3–5pts over v3; Hebrew Biblical DER improvement;
Arabic SadeedDiac-25 DER toward <1.5%.

## Commonality
One module, three languages (Persian/Hebrew/Arabic); Urdu via transfer (#03).
Supersedes TODO #312.

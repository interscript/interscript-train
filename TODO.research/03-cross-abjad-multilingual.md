# 03 — Cross-abjad multilingual diacritizer

## Goal
One ByT5 with language-tag prefix, trained jointly on Arabic (2.1M), Hebrew
(~100K), Persian (464K derived), Urdu (597K IPA-derived). Measure per-language
delta vs the monolingual models, especially Urdu (weakest supervision).

## Why groundbreaking
The four scripts share one ancestor and one pathology (short-vowel omission,
homographs). Cross-lingual diacritization among sister scripts is unexplored
at this data scale; our bottleneck taxonomy predicts Urdu benefits most from
Arabic transfer (shared lexicon: Arabic loanwords are a large fraction of
Urdu vocabulary).

## Steps
1. Unified JSONL format {lang, src, tgt} + `<lang>` byte prefix.
2. Joint ByT5-base run (1.2B — saturates anyway) on Modal A100.
3. Ablations: joint vs mono; with/without Urdu; frozen-then-finetune per lang.
4. Ezafe note: Persian/Urdu ezafe machinery is shared; Arabic/Hebrew
   case-ending morphology is shared — tag which transfers.

## Success metric
Urdu diacritization CER 3.74% → below 2.5% from Arabic transfer alone;
no regression >0.1pp on Arabic/Hebrew/Persian.

## Commonality
Abjad family: trie lexicons, RAG homographs (#05), DER metrics all shared.

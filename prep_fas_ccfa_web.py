# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=0.26",
# ]
# ///
"""WO94 phase-1 acquisition — cc-fa-web: Common Crawl Persian web-register
slice (the OSCAR-fa SUBSTITUTE, see the provenance note).

OSCAR-fa (the WO66 dossier's third source) is NOT ACQUIREABLE as of
2026-11-25: every canonical OSCAR release on HF is gated with manual
approval AND the gate text states, verbatim: "We are forced to temporarily
suspend access to OSCAR ... we will not grant any access until the
situation has been clarified." No archive snapshot, no Zenodo mirror,
mc4 removed from HF; third-party mirrors carry unverifiable lineage.
This job reconstructs the dossier row's REGISTER INTENT — its own framing:
"colloquial/web; Common-Crawl-derived; register = modern web; quality
filters needed" — directly from Common Crawl WET through OSCAR-style
filters. Provenance per row: crawl id + WET segment + source URI.

Source: https://data.commoncrawl.org crawl CC-MAIN-2026-47 (Nov 2026;
100,000 WET segments, ~66MB gz each at probe time). A deterministic,
evenly-strided segment sample (stride over the segment list) keeps the
slice reproducible from the crawl id alone.

License: Common-Crawl-derived. The crawl contents carry their own
per-document terms; Common Crawl's policy permits use and redistribution
of the crawl. Registry treatment: license "LicenseRef-CommonCrawl-Terms"
+ license_review record citing the owner's dossier framing (the same
construction the owner reviewed for OSCAR-fa). Weak-pretrain mass only —
NEVER the quality or eval signal (no-teacher-poison law; machine-mass
never gates quality).

Pipeline (all in-job; corpora NEVER touch the laptop):
  1. Fetch wet.paths.gz for the crawl; take every stride-th segment
     (deterministic from the crawl id + FAS_CC_NSEG).
  2. Per segment (resumable .done markers in the bucket): stream-download,
     parse WARC records, keep WARC-Type: conversion docs whose
     WARC-Identified-Content-Language contains fas/pes/fa (header) AND
     whose Persian-letter fraction passes the floor (belt + suspenders).
  3. Clean lines (drop nav junk: short non-letter lines), collapse blanks.
  4. Filters: doc >= 50 chars, <= 100_000; Persian-letter fraction >= 0.6;
     exact-text dedupe first-occurrence-wins; 40-char-window decontam vs
     the two pinned fas evals (uniform with the other acquisition jobs).
  5. Contiguous shards + manifest.json + SHA256SUMS.txt + DONE.

Usage (cpu-upgrade, detached, resumable):
    hf jobs uv run --flavor cpu-upgrade -d --timeout 12h \
      -s HF_TOKEN \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      -v hf://buckets/Interscript/isx-training/run-089-fas-poetry-eval:/evals/poetry:ro \
      -v hf://buckets/Interscript/isx-training/run-076-fas-diac-retrain:/evals/docs:ro \
      --name wo94-ccfa-web prep_fas_ccfa_web.py

This script is a THIN ENTRY POINT over the acquisition factory
(acquisition/): the pipeline stages live there; this file contributes
the WO94 Persian recipe (`acquisition.configs.fas`) and the provenance
docstring below (the pipeline narrative). Behavior-locked to the
pre-refactor scripts by tests/acquisition (byte-identical artifacts on
identical inputs).

"""


from acquisition.configs.fas import ccfa_web
from acquisition.run import run_prep


def main() -> None:
    run_prep(ccfa_web())


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=0.26",
# ]
# ///
"""WO94 phase-1 acquisition — fa-Wikisource classical-prose + poetry corpus.

Source: dumps.wikimedia.org fawikisource-latest-pages-articles.xml.bz2
(the canonical Wikimedia dump channel; observed 47,719,980 B, Last-Modified
2026-11-01 at probe time — the manifest records exactly what was fetched).
License: CC BY-SA 4.0 (Wikimedia site terms; each page's authors attributed
via page history — per-row title + pageid preserved for attribution).

Pipeline (all in-job; corpora NEVER touch the laptop):
  1. Stream-download the bz2 dump; record size + sha256 (bytes streamed).
  2. iterparse the XML; keep ns=0 pages (articles), skip redirects
     (redirect/تغییرمسier markers), skip disambiguation-looking stubs.
  3. Wikitext -> text: drop refs/table magic words/comments; flatten
     templates by extracting argument VALUES (fa-wikisource poetry pages
     wrap verses in {{شعر|بیت|بیت|...}} templates — dropping templates
     whole would delete the poem bodies, so values are kept, names dropped);
     resolve [[a|b]] -> b; strip outer html tags; keep line structure.
  4. Filters: text >= 50 chars, Arabic-letter fraction >= 0.5 of
     non-space; exact-title dedupe; 40-char-window decontam (stride 20)
     vs the two pinned fas evals (uniform with the ganjoor job).
  5. Contiguous shards + manifest.json + SHA256SUMS.txt + DONE.

Usage (cpu-upgrade, detached):
    hf jobs uv run --flavor cpu-upgrade -d --timeout 4h \
      -s HF_TOKEN \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      -v hf://buckets/Interscript/isx-training/run-089-fas-poetry-eval:/evals/poetry:ro \
      -v hf://buckets/Interscript/isx-training/run-076-fas-diac-retrain:/evals/docs:ro \
      --name wo94-fawikisource-dump prep_fas_fawikisource.py

This script is a THIN ENTRY POINT over the acquisition factory
(acquisition/): the pipeline stages live there; this file contributes
the WO94 Persian recipe (`acquisition.configs.fas`) and the provenance
docstring below (the pipeline narrative). Behavior-locked to the
pre-refactor scripts by tests/acquisition (byte-identical artifacts on
identical inputs).

"""


from acquisition.configs.fas import fawikisource
from acquisition.run import run_prep


def main() -> None:
    run_prep(fawikisource())


if __name__ == "__main__":
    main()

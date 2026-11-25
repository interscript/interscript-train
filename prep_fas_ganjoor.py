# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=0.26",
# ]
# ///
"""WO94 phase-1 acquisition — Ganjoor (ganjoor.net) vocalized poetry corpus.

Source: ganjoor.net public REST API (api.ganjoor.net) — the canonical bulk
channel. The official ganjoor/ganjoor-db dump is from 2017 (rejected as
stale, sha256 4623701c5ff54a4f recorded in the WO94 ledger). License:
CC BY-SA 3.0, site-stated (the WO66 dossier record, owner-reviewed
2026-10-25; acquisition owner-approved 2026-11-24 in interscript-models#488).
Per-row attribution: poet_url + poet_name + cat_path + poem_id.

The API exposes, per poem, plainText (site-stripped) and verses[].text —
the VOCALIZED text (erab/haraqat present), e.g. verse 'اَلا اِی آهویِ
وحشی! کجایی؟' vs plainText 'الا ای آهوی وحشی کجایی'. We emit verse-level
pairs {src = strip(tgt), tgt = verse text verbatim, ...} — exactly the
pinned run-089-fas-poetry-eval convention (verified against the pinned
eval: src keeps punctuation, drops the vocalic marks U+064B-U+0652 +
U+0670 while KEEPING the hamza/maddah carriers U+0653-U+0655 (measured
from the pinned rows: U+0654 x130, U+0656 x1 retained); its
tgt/src length-ratio median 1.1724, max 1.8571).

Pipeline (all in-job; corpora NEVER touch the laptop):
  1. GET /api/ganjoor/poets -> published poets (246 as of 2026-11-25).
  2. Per poet: DFS the category tree (GET /cat/{id}?poems=true; the API
     fills children+poems one level per request) -> poem-id inventory.
     Per-poet raw output + .done markers land in the BUCKET run dir, so
     the crawl is resumable across job restarts.
  3. Per poem: GET /poem/{id}?catInfo=false&recitations=false&images=false
     &comments=false&rhymes=false&songs=false (minimal payload).
  4. Assemble: src = strip_marks(tgt); filters + dedupe + decontam.
  5. Contiguous shards (source order) + manifest.json + SHA256SUMS.txt
     + DONE into the bucket run dir.

Filters (pre-registered in WO94 before acquisition):
  - 10 <= len(src) <= 300 chars (verse units; the s45 convention)
  - Arabic-script letter fraction of non-space chars >= 0.5 (pinned-eval
    calibration: median 0.8, p5 0.6)
  - exact-tgt dedupe, first occurrence wins, source order preserved
  - decontam: 40-char windows (stride 20) over Arabic-letter normal forms
    vs BOTH pinned evals (poetry eval src+tgt sides; docs test.jsonl
    src+tgt sides) — the evals must stay held-out: this corpus shares the
    upstream source site with the poetry eval.

Semantic invariants measured in-job, recorded in manifest.json (the
run-093 law); ENFORCED by the separate gate_fas_acquisition.py job:
  - len(tgt)/len(src) ratio in [1.0, 2.0] for every emitted row
  - vocalized share (tgt != src) corpus-level, per-poet share recorded
  - harakat density (marks-per-letter) distribution on vocalized rows
  - degenerate rows (empty src / empty tgt) < 1%

Usage (cpu-upgrade, detached, resumable):
    hf jobs uv run --flavor cpu-upgrade -d --timeout 12h \
      -s HF_TOKEN \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      -v hf://buckets/Interscript/isx-training/run-089-fas-poetry-eval:/evals/poetry:ro \
      -v hf://buckets/Interscript/isx-training/run-076-fas-diac-retrain:/evals/docs:ro \
      --name wo94-ganjoor-crawl prep_fas_ganjoor.py

This script is a THIN ENTRY POINT over the acquisition factory
(acquisition/): the pipeline stages live there; this file contributes
the WO94 Persian recipe (`acquisition.configs.fas`) and the provenance
docstring below (the pipeline narrative). Behavior-locked to the
pre-refactor scripts by tests/acquisition (byte-identical artifacts on
identical inputs).

"""


from acquisition.configs.fas import ganjoor
from acquisition.run import run_prep


def main() -> None:
    run_prep(ganjoor())


if __name__ == "__main__":
    main()

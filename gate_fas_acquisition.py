# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=0.26",
# ]
# ///
"""WO94 phase-1 corpus gates — the independent stats gate (run-093 law).

Re-verifies the acquired artifacts FROM SOURCE (bucket, not job scratch):
SHA256SUMS re-hash of every shard + manifest, row counts, and the
pre-registered semantic invariants per source. Persian script/range law
is spelled out in GATE_DEFS below and in the WO94 ledger.

Pre-registered gates (before acquisition; WO94 ledger):

  ganjoor (src/tgt verse pairs):
    - ratio len(tgt)/len(src) in [1.0, 2.0] for EVERY row (vocalization
      only ever ADDS marks; band calibrated on the pinned eval:
      median 1.1724, max 1.8571)
    - vocalized share (tgt != src) >= 0.30 corpus-level
    - harakat density (marks-per-Arabic-letter) median >= 0.05 on
      vocalized rows (pinned-eval calibration: median 0.2222, p5 0.037)
    - degenerate rows (empty src/tgt) < 1%
    - script sanity: Arabic-letter fraction median >= 0.5 (eval p5 0.6);
      Persian-specific letters (پ چ ژ گ) all present; ZWNJ present

  fawikisource (page rows):
    - rows >= 500; degenerate < 1%; Arabic-letter fraction median >= 0.5;
      Persian-specific letters present

  cc-fa-web (document rows; weak-pretrain mass ONLY — never gates quality):
    - rows >= 10_000; degenerate < 1%; Arabic-letter fraction median
      >= 0.6 (the acquisition floor itself)

Cross-source:
  - SHA256SUMS.txt must re-hash clean for every shard + the manifest
  - manifest.json must exist per source; DONE marker present
  - decontam + dedupe counts recorded in every manifest

Usage (cpu-upgrade, detached; the run dir mounted rw for the verdict):
    hf jobs uv run --flavor cpu-upgrade -d --timeout 2h \
      -s HF_TOKEN \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      --name wo94-acquisition-gate gate_fas_acquisition.py

This script is a THIN ENTRY POINT over the acquisition factory
(acquisition/gate.py): the gate engine lives there; this file contributes
the WO94 pre-registered Persian gate defs (`acquisition.configs.fas`).
Behavior-locked to the pre-refactor script by tests/acquisition.

"""


from acquisition.configs.fas import GATE_CAMPAIGN, GATE_LAW_LINE, gate_dir, gate_sources
from acquisition.run import run_gates


def main() -> None:
    run_gates(gate_dir(), gate_sources(), GATE_CAMPAIGN, GATE_LAW_LINE)


if __name__ == "__main__":
    main()

# WO20u — Thai ByT5 plane: lexicon-lane base arm + schedule arm (branch wo20u-thai)

Campaign doc; the plan issue lives on interscript/interscript-models.
Everything here is execution detail; the gates below are PRE-REGISTERED
and must not be edited after the full runs launch.

## Lane

PER on the ByT5 plane, Thai lexicon lane (Thai orthography ->
phonemic-with-tone fallback learner behind the deterministic RTGS
composite). The CTC plane (plateaus ~70% PER) and from-scratch char
seq2seq (mode collapse) are closed negatives; ByT5 is the validated
direction.

## Baselines to beat (published, quotable)

| model | lane | protocol | PER |
|---|---|---|---|
| tha-phonemic-1.0 (ByT5-small v6a) | lexicon | phoneme-token PER, clean v1 test split (3,210 rows), single-row greedy | **19.29%** |
| tha-phonemic-small-1.0 (tha-g2p-small) | Kaikki KD | greedy, corpus-level PER, 1,219 Kaikki test | 2.85% (beam-4 12.06%) |
| tha-phonemic-base-1.0 (tha-g2p-base) | Kaikki KD | greedy, corpus-level PER, 1,219 Kaikki test | 3.53% (beam-4 9.19%) |

The Kaikki lane and the lexicon lane use different test sets and target
conventions — NEVER cross-compare across lanes. The lexicon lane is
this campaign's comparable surface; the Kaikki numbers are context
only. The FastThaiG2P hybrid lane (latency) is out of scope.

## Established assets (reused unchanged)

- Corpus: `isx-training:/run-080-thai-lexicon-v5/` (tha-phonemic-v5,
  92,669 length-filtered pairs — orst+thai_words+volubilis+wiki-agree;
  the shipped tha-phonemic-1.0 trained_from). Mounted, never local.
- Trainer: `train_thai_lexicon_hf.py` (repo root) — resume from
  `oepoch-*`, epoch checkpoints to the bucket while the job runs.
- Eval: `eval_thai_lex.py` (repo root) — PER over phoneme tokens +
  string CER, on `run-080-thai-evalclean/tha-phonemic-test.jsonl`
  (defect-filtered v1 test, 3,210 rows,
  sha256 dfd748b7ee8940b73368999b7eaedc9b80859fea4ce8444c4952788927cd4f98).
- v6a reference invocation: T_BACKBONE=google/byt5-small, T_EPOCHS=4,
  T_BS=4, T_TRAIN=tha-phonemic-v5-train.jsonl,
  T_DEV=tha-phonemic-v5-dev.jsonl (=> 92,668 updates).

## Arms

- **Arm B — schedule probe (small, 8ep)**: run dir
  `run-wo20u-thai-small-8ep`; v6a protocol doubled to T_EPOCHS=8
  (185,336 updates). Tests the undertrained hypothesis on v5 (v6a
  stopped at oepoch-3; WO67's schedule-confound lesson makes update
  budget the first suspect).
- **Arm A — capacity probe (base, 4ep)**: run dir
  `run-wo20u-thai-base-4ep`; EXACT v6a protocol, backbone
  google/byt5-base. The base tier has never been trained on the
  lexicon lane — this is the missing plane cell.

Both arms: a100-large, BS4, lr 3e-4 AdamW cosine (trainer defaults),
`--timeout 4h`, detached, `--attempts` for cross-window resume (the
trainer resumes from the last `oepoch-*` on restart).

## Pre-registered gates (fixed before full launch — do not edit)

Eval protocol for both arms: `eval_thai_lex.py` unchanged,
E_DATA=run-080-thai-evalclean, latest oepoch, single-row greedy,
phoneme-token PER on the 3,210-row clean split.

- **Arm B PASS iff clean-split PER < 0.1929** (strictly better than
  the shipped v6a small-4ep). PASS => the 8-epoch recipe becomes the
  lexicon-lane reference recipe (release = owner version decision).
- **Arm A PASS iff clean-split PER < 0.1929** (strictly better than
  the shipped small arm). PASS => base-tier lexicon candidate staged
  for owner inspection; artifact cost is an owner decision.
- Attribution note recorded either way: A-vs-B orders capacity vs
  schedule on this corpus.
- Negative results are recorded as negatives in RESULTS.md lineage;
  no version numbers are picked by the campaign.

## Fixture-scale test (before any full launch)

`make_fixture.py` slices 400 train / 80 dev rows into
`run-wo20u-thai-fixture/`, prints counts + sha256 of the real v5
files (provenance only, no corpus text). Fixture job: 1 epoch,
BS8, byt5-small, then the eval harness on the clean split.
PASS = job completes, `oepoch-0` saved, `eval-verdict.json` written.
The fixture's PER value is meaningless (400-row training).

## Discipline

- HF Jobs substrate, detached, `--timeout 4h`, retry-wrapped launch;
  `hf jobs ps` checked before each launch — if the urdu/arabic jobs
  saturate the queue, queue via retry.
- Checkpoints land in the bucket mid-job (oepoch-N dirs); any 4h kill
  resumes from the last one.
- Corpus text never appears in prompts, PRs, issues, or logs — counts,
  hashes, and paths only.
- No LLM teachers anywhere (standing rule). No RL.

## Contingency (declared now)

If the base epoch does not fit a 4h window (no oepoch-0 after 3
attempts), the base arm falls back to T_BS=8 / T_EPOCHS=8 (same
92,668-update budget, declared schedule deviation) rather than
burning windows on zero-checkpoint rounds.

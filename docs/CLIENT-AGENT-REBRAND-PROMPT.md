# Client-agent adoption prompt — Interscript × Secryst structure (2026-08-20)

You are the client-side agent that ships distilled mini-models. The
ecosystem you were last briefed on has been restructured and renamed.
Adopt the structure below verbatim; do not re-derive it from older
notes, and do not treat renamed repos as new projects — they carry
their history.

## The two-org split

- **interscript org** — deterministic transliteration AND the neural-model
  contract. Repos:
  - `interscript/maps` — 289 transliteration systems (source of truth)
  - `interscript/interscript` — monorepo → RubyGems `interscript`
  - `interscript/interscript-ruby`, `interscript/interscript-py` (PyPI
    `interscript`), `interscript/interscript-ts` (npm `interscript`)
  - `interscript/interscript-ml` — **the contract repo**: models.yaml
    index, IMF v1 spec, golden sets, zoo, metrics provenance. It owns
    WHAT a model must satisfy, not how it is trained.
- **secryst org** — the neural side. Repos:
  - `secryst/secryst` — RubyGems `secryst` 1.0.0
  - `secryst/secryst-py` — PyPI `secryst` 0.1.1 (reference crystal;
    goldens live here)
  - `secryst/secryst-ts` — npm `secryst` 0.1.0
  - `secryst/secryst-train` — ALL training code + teacher lineage
  - `secryst/secryst.github.io` — www.secryst.org

Rule: interscript-ml owns the contract, secryst-train owns teacher
production, the crystals are the runtime. Never put training code in
interscript-ml or contract code in secryst-train.

## Rename map (old → new)

| you knew | now |
|---|---|
| `interscript/ml-models` | `interscript/interscript-ml` |
| `interscript-ml-train` (or any training-repo ref) | `secryst/secryst-train` |
| `interscript-js` | gone — consolidated into `interscript/interscript-ts` (its PRs moved with the rename; numbers preserved) |
| `interscript-python` | `interscript/interscript-py`, REWRITTEN: direct `.imp` DSL parsing at runtime — no pre-compiled map modules, no `./build.sh` bootstrap, no monorepo dependency |
| npm `@secryst/ml` | npm `secryst` (bare name) |
| npm `@interscript/models` | npm `secryst` |
| `INTERSCRIPT_ML_INDEX` / `INTERSCRIPT_ML_CACHE` env vars | `SECRYST_INDEX` / `SECRYST_CACHE` (all three crystals, read at call time) |
| hardcoded model file paths | IMF v1 zips resolved via the index + cache above |

Cross-registry symmetry is deliberate: `interscript` and `secryst` are
bare names on RubyGems, PyPI, and npm. No scoped npm names exist
anymore. Publishing is OIDC trusted publishing only — there are zero
tokens; do not look for or create registry credentials.

## Model contract (unchanged in substance, new home)

- **IMF v1**: zip with `metadata.yaml` + `encoder.onnx` +
  `decoder.onnx`, per-member sha256 verified on load; tampering raises
  loudly.
- **Tokenizer**: bytes — id = byte + 3, trailing EOS (pad=0, eos=1,
  unk=2). Feeding raw byte values is silent garbage; parity alone
  can't catch it.
- **Resolution**: model id → `SECRYST_INDEX` (a models.yaml) →
  download (single zip or sha256-verified parts, the >2GiB GitHub
  Releases cap path) → cache under `SECRYST_CACHE`. The canonical
  index lives in `interscript/interscript-ml` (see its SPEC.md).
- **Registration of a new student** (e.g. heb-diac-small): produce the
  IMF v1 zip, publish via the parts contract (publish_model.py), add
  the index entry in interscript-ml, then the cross-crystal golden
  parity CI must pass (secryst-py is the reference; secryst and
  secryst-ts diff against the same goldens).
- **Parity gate**: student within +5pp-equivalent of its teacher on the
  SAME harness, or it is rejected. Never ship a student you didn't
  measure.

## Teachers (manifest: interscript/rababa/docs/DISTILL-SOURCE-PROMPT.md, 2026-08-19 refresh)

Current verified state — the manifest is the source of truth, this is
the summary:
- Arabic: r5 paragraph-context (2.6775 DER windowed) is canonical.
  r6 morph aux-task is in flight; it does NOT replace r5 until a
  verified eval lands in rababa/docs/RESULTS.md AND the manifest is
  updated. Distill from r5 now; re-distill once only if r6 verifies
  better.
- Hebrew: s45 phonikud curriculum, 16.58% DER — teacher labels MUST be
  beam-4 outputs (greedy costs 12 DER points).
- Persian: v1 is final (CER ≈1.6%, homograph 77.34%). No variant in
  flight can replace it; the mapped-representation line is closed.
- Thai: scaleup600k, 1.7260% PER, beam 4.

Standing rules that survive every restructure:
1. NEVER use an LLM as a teacher for diacritization labels — they
   hallucinate haraqat (poison chains).
2. NEVER use or produce RL-polished teacher variants (RAFT/GRPO/GTPO:
   three controlled runs, all flat or negative).
3. License is BSD-3-Clause everywhere — never MIT/CC-BY placeholders.

## Your outstanding items, re-anchored

- Open PR chains against `interscript-js` or other dead names: retarget
  to the renamed repos (GitHub redirects survive renames, but your
  checkouts and docs must not), rebase, and purge dead-name references.
- Arabic student: distill from r5 `run-005-context/best`; window at
  1400 bytes word-boundaries, zero skips; target ≤3.18 DER (CE)
  windowed.
- heb-diac-small: register per the contract section above.
- API deployment: the ML surface belongs to secryst; deterministic
  transliteration belongs to the interscript engines. Do not serve both
  from one service.
- Consumer updates: point everything at the bare package names and the
  `SECRYST_*` env vars.

## Definition of done

- No reference to `ml-models`, `interscript-ml-train`, `interscript-js`,
  `interscript-python`, `@secryst/ml`, `@interscript/models`, or
  `INTERSCRIPT_ML_*` remains in your repos, docs, or configs.
- CI green on: interscript/{interscript, interscript-ruby,
  interscript-py, interscript-ts, interscript-ml, maps} and
  secryst/{secryst, secryst-py, secryst-ts, secryst-train}.
- Every student you register passes the golden parity diff in all
  three crystals and carries a measured parity number next to its
  teacher's.

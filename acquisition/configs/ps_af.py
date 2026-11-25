"""DRAFT — Pashto (pus) acquisition recipe. DO NOT LAUNCH.

STATUS: ILLUSTRATIVE STARTER ONLY. Sources UNVERIFIED per ROADMAP-2027
R4 (TODO.sota/ROADMAP-2027.md): large UNDIACRITIZED monolingual mass
exists (e.g. a 1.5B-word community corpus on HF, cited by the roadmap),
but NO diacritized gold corpus surfaced — the binding constraint. The
undiacritized mass is useful ONLY as curriculum weak-pretrain IF a
licensed machine labeler appears; never self-labeled for eval. This
config wires the crawl adapter as the shape a future weak-pretrain
slice would take; the community-corpus source below is a PLACEHOLDER
(the roadmap's HF reference, unverified here).

This draft demonstrates the Common-Crawl WET adapter wired for a
second Arabic-script language — "a new language = a new config module".

Open verification items (the R4 eval-slice track — start FIRST, the
annotation campaign is the lead-time item):
  - human-annotated gold eval is the prerequisite; quality control per
    the urdu relabel-input lesson (semantic invariants at load time)
  - dialect split (Southern/Northern) must be declared PER SPLIT
    before any slice is staged
  - the ScriptLaw inventory below needs verification against real
    Pashto text (the letter set is the standard inventory, unconfirmed
    here)
"""

from __future__ import annotations

from ..charlaw import ScriptLaw
from ..config import (
    AcquisitionConfig,
    AssembleConfig,
    CcWetConfig,
    CharCeiling,
    DedupeFilter,
    DecontamFilter,
    EvalMount,
    ExclusiveReject,
    LenFilter,
    ScriptFilter,
)

STATUS = ("DRAFT — UNVERIFIED SOURCES (ROADMAP-2027 R4): illustrative "
          "starter, not a launch; human-annotated gold eval is the "
          "prerequisite (the lead-time item)")

# Arabic base + the Pashto extensions (standard inventory; TODO(R4):
# verify against real Pashto text before launch).
ARABIC_BASE = ((0x0621, 0x063A), (0x0641, 0x064A), (0x0671, 0x06D3))
PASHTO_EXTENSIONS = (
    (0x0681, 0x0681),  # ځ dzhe
    (0x0685, 0x0685),  # څ tse
    (0x0689, 0x0689),  # ډ ddal
    (0x0696, 0x0696),  # ږ rre
    (0x069A, 0x069A),  # ښ kheen
    (0x06AF, 0x06AF),  # ګ gaf
    (0x06BC, 0x06BC),  # ڼ nune
    (0x06CD, 0x06CD),  # ۍ kheh
    (0x06D0, 0x06D0),  # ې e
    (0x067F, 0x067F),  # ټ tte
)

PUSHTO_LAW = ScriptLaw(
    name="pus-draft",
    letter_ranges=ARABIC_BASE + PASHTO_EXTENSIONS,
    # Pashto writes long vowels as letters; the harakat set is the
    # Arabic one (used sparingly — TODO(R4): verify usage share)
    strip_ranges=((0x064B, 0x0652), (0x0670, 0x0670)),
    density_ranges=((0x064B, 0x0652), (0x0670, 0x0670)),
)

# TODO(R4): the eval mount does not exist — human annotation is the
# prerequisite. Placeholder mount for the annotated eval slice.
EVAL_MOUNTS = (
    EvalMount(env="PUS_EVAL_GOLD", default="/evals/pus-gold/eval.jsonl"),
)

# Neighbor-script letters Pashto never uses — Persian-check: Pashto
# shares پ چ ژ گ with Persian, so the rejects target URDU and KURDISH
# specific letters (verify the inventory at probe time).
PUS_EXCLUSIVE_REJECTS = (
    ExclusiveReject(name="urdu-draft", letters="ٹڈڑںے", threshold=3),
    ExclusiveReject(name="kurdish-draft", letters="ڕڵۏ", threshold=3),
)


def pus_web_weak_draft() -> AcquisitionConfig:
    """Weak-pretrain web register slice (the OSCAR-substitute pattern).
    DRAFT: curriculum mass only — never the quality or eval signal; the
    dialect split below is a placeholder pending the R4 declaration."""
    return AcquisitionConfig(
        source_name="pus-web-draft",
        output_prefix="pus-web",
        law=PUSHTO_LAW,
        adapter=CcWetConfig(
            base_url="https://data.commoncrawl.org",
            crawl_id="CC-MAIN-PROBE",  # probe pins the crawl id
            n_segments=50,
            user_agent="interscript-acquisition/0.1 (pus DRAFT, unverified)",
            # ps/pus/pbu(=NC)/pst: the allowlist is a PLACEHOLDER — the
            # Southern/Northern split must be declared per split (R4)
            lang_allowlist=frozenset({"ps", "pus", "pst"}),
            letter_floor=0.6,
            doc_min=50,
            doc_max=100_000,
            clean_doc_min=50,
            nav_junk_min_letters=3,
            nav_junk_max_len=60,
            exclusive_rejects=PUS_EXCLUSIVE_REJECTS,
            # Arabic ta-marbuta shares a ceiling rationale (Arabic-heavy
            # docs are not Pashto); the threshold is a probe question
            char_ceilings=(CharCeiling(char="ة", max_ratio=0.01),),
        ),
        assemble=AssembleConfig(
            steps=(
                DedupeFilter(field_name="text"),
                DecontamFilter(),
            ),
            text_field="text",
            partner_field=None,
            raw_row_counter="docs_raw",
            count_chars_as="chars_raw",
            kept_log_every=5_000,
        ),
        campaign="R4 Pashto weak-pretrain DRAFT (ROADMAP-2027 R4; unverified)",
        source_meta={
            "status": STATUS,
            "origin": "Common Crawl WET, crawl UNPINNED (probe decides)",
            "license": "LicenseRef-CommonCrawl-Terms (weak-pretrain mass "
                       "ONLY — never the quality or eval signal)",
            "dialect_split": "UNDECLARED — Southern/Northern must be "
                             "declared per split (the R4 risk line)",
        },
        policy_meta={
            "unit": "web document (WARC-Type: conversion)",
            "langid": "WARC language header (allowlist) AND "
                      "Pashto-letter fraction >= 0.6 floor; script-exclusive "
                      "rejection of Urdu/Kurdish letters",
            "clean": "nav junk lines dropped; blank runs collapsed",
            "filters": "50 <= len <= 100_000 chars",
            "dedupe": "exact-text dedupe, first occurrence wins",
            "decontam": "40-char windows (stride 20) vs the annotated "
                        "gold eval slice (once it exists)",
            "sharding": "contiguous source-order slices",
        },
        run_dir_default="run-pus-draft/web",
        raw_subdir="segments",
        eval_mounts=EVAL_MOUNTS,
        count_extras=("kept_unique",),
    )

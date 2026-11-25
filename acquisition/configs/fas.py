"""WO94 Persian (fas) acquisition recipes — the factory's first instance.

These configs reproduce the committed phase-1 scripts byte-for-byte on
identical inputs (proven by tests/acquisition). The license/provenance
prose below is the WO94 ledger record: ganjoor CC BY-SA 3.0
(owner-reviewed 2026-10-25; acquisition approved interscript-models#488
2026-11-24), fa-Wikisource CC BY-SA 4.0, cc-fa-web as the OSCAR-fa
SUBSTITUTE (canonical OSCAR upstream-suspended).

Character-class law (the WO94 gate definitions):
  - letters: the Arabic block base ranges
  - PINNED-EVAL strip set (prep): U+064B-0652 + U+0670 — the hamza/
    maddah carriers U+0653-0655 STAY in the src (they change the letter
    itself; the pinned eval src keeps U+0654, 130 occurrences)
  - GATE density set: U+064B-065F + U+0670 — WIDER than the prep strip
    set. This dual law is pre-existing (the prep job and the gate job
    each define their own mark predicate) and preserved here as two
    ScriptLaw instances.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from ..charlaw import ScriptLaw
from ..config import (
    AcquisitionConfig,
    ApiWalkConfig,
    AssembleConfig,
    CcWetConfig,
    CharCeiling,
    DedupeFilter,
    DecontamFilter,
    EvalMount,
    ExclusiveReject,
    InvariantBands,
    LenFilter,
    RatioFilter,
    ScriptFilter,
    WikiDumpConfig,
)
from ..gate import GateCheck, GateDef, GateSource

CAMPAIGN = "WO94 fas poetry acquisition (TODO.sota/94, interscript-models#488)"
GATE_CAMPAIGN = "WO94 fas acquisition corpus gates"
GATE_LAW_LINE = ("run-093 semantic invariants; pre-registered thresholds "
                 "documented in gate_fas_acquisition.py + the WO94 ledger")

EVAL_MOUNTS = (
    EvalMount(env="FAS_EVAL_POETRY",
              default="/evals/poetry/fas-poetry-eval.jsonl"),
    EvalMount(env="FAS_EVAL_DOCS", default="/evals/docs/test.jsonl"),
)

# ---- character-class laws ----

ARABIC_LETTER_RANGES = ((0x0621, 0x063A), (0x0641, 0x064A), (0x0671, 0x06D3))

FAS_PREP_LAW = ScriptLaw(
    name="fas-prep",
    letter_ranges=ARABIC_LETTER_RANGES,
    # the PINNED-EVAL strip set, measured from the pinned rows: the vocalic
    # marks U+064B-U+0652 (tanwin..sukun) + dagger alif U+0670. The
    # hamza/maddah carriers U+0653-U+0655 STAY. Prep density uses the same
    # set (the prep script's own predicate).
    strip_ranges=((0x064B, 0x0652), (0x0670, 0x0670)),
    density_ranges=((0x064B, 0x0652), (0x0670, 0x0670)),
)

FAS_GATE_LAW = ScriptLaw(
    name="fas-gate",
    letter_ranges=ARABIC_LETTER_RANGES,
    # the gate's own mark predicate: WIDER (U+064B-U+065F + U+0670)
    density_ranges=((0x064B, 0x065F), (0x0670, 0x0670)),
)

PERSIAN_SPECIFIC = "پچژگ"      # Persian-specific letters (gate presence check)
ZWNJ = "‌"

# cc-fa-web script-exclusive rejection (pre-registered): letters the
# NEIGHBOR scripts use that Persian never does. Kurdish Sorani: ڕ ڵ ۆ ێ ڤ ە;
# Urdu: ٹ ڈ ڑ ں ے. Threshold >= 3 occurrences — a stray glyph in a quote is
# fine, a Kurdish/Urdu document is not.
KURDISH_ONLY = frozenset("ڕڵۏێە")
URDU_ONLY = frozenset("ٹڈڑںے")


# ---- ganjoor (ganjoor.net API; verse-level src/tgt pairs) ----

def ganjoor() -> AcquisitionConfig:
    api = ApiWalkConfig(
        base_url="https://api.ganjoor.net/api/ganjoor",
        user_agent="interscript-wo94-acquisition/0.1 "
                   "(ganjoor CC BY-SA 3.0 attribution corpus)",
        list_endpoint="/poets",
        list_published_key="published",
        cat_endpoint="/cat/{cid}?poems=true&mainSections=false",
        item_endpoint=("/poem/{pid}?catInfo=false&recitations=false"
                       "&images=false&comments=false&rhymes=false"
                       "&songs=false&mainSections=false"
                       "&top6QuotedPoems=false&geoDateTags=false"),
        workers=int(os.environ.get("FAS_WORKERS", "6")),
    )
    return AcquisitionConfig(
        source_name="ganjoor",
        output_prefix="ganjoor",
        law=FAS_PREP_LAW,
        adapter=api,
        assemble=AssembleConfig(
            steps=(
                LenFilter(min_len=10, max_len=300, count_as="filtered_len"),
                ScriptFilter(min_fraction=0.5),
                RatioFilter(lo=1.0, hi=2.0),
                DedupeFilter(field_name="tgt"),
                DecontamFilter(),
            ),
            text_field="src",
            partner_field="tgt",
            raw_row_counter="verses_raw",
            kept_log_every=200_000,
        ),
        campaign=CAMPAIGN,
        source_meta={
            "site": "ganjoor.net",
            "api": api.base_url,
            "channel_note": "live API = canonical bulk channel; the official "
                            "ganjoor/ganjoor-db dump.sql.gz is from 2017 (stale, rejected)",
            "license": "CC BY-SA 3.0 (site-stated; WO66 dossier record, "
                       "owner-reviewed 2026-10-25; acquisition approved #488 2026-11-24)",
            "attribution_fields": ["poet_url", "poet_name", "cat_path", "poem_id",
                                   "poem_title"],
            "sharealike": "derived corpus inherits CC BY-SA 3.0; weights carry the "
                          "copyleft dependency note (the tashkeela lane)",
            "crawl_snapshot_utc": os.environ.get(
                "FAS_CRAWL_SNAPSHOT",
                datetime.now(timezone.utc).date().isoformat()),
        },
        policy_meta={
            "unit": "verse (API verses[].text = vocalized; src = pinned-eval "
                    "strip convention: drop U+064B-U+0652 + U+0670, keep "
                    "U+0653-U+0655 hamza/maddah carriers (pinned-eval measured), "
                    "keep punctuation/ZWNJ)",
            "filters": "10 <= len(src) <= 300; Arabic-letter fraction >= 0.5 of "
                       "non-space; ratio len(tgt)/len(src) in [1.0, 2.0]",
            "dedupe": "exact-tgt dedupe, first occurrence wins, source order preserved",
            "decontam": "40-char windows (stride 20) over Arabic-letter normal forms "
                        "vs run-089-fas-poetry-eval (src+tgt) AND run-076 docs "
                        "test.jsonl (src+tgt)",
            "sharding": "contiguous source-order slices",
        },
        run_dir_default="run-098-fas-acquisition/ganjoor",
        raw_subdir="raw",
        shards_in_raw_dir=True,
        eval_mounts=EVAL_MOUNTS,
        crawl_snapshot_env="FAS_CRAWL_SNAPSHOT",
        decontam_log_paths=True,
        measure_invariants=True,
        invariant_bands=InvariantBands(
            vocalized_share_min=0.30,
            ratio_band=(1.0, 2.0),
            degenerate_max=0.01,
            harakat_density_median_min=0.05,
        ),
        group_field="poet_url",
        count_extras=("poet_files", "kept_unique", "degenerate"),
    )


# ---- fawikisource (Wikimedia dump; page rows) ----

FAWIKISOURCE_DUMP_URL = ("https://dumps.wikimedia.org/fawikisource/latest/"
                         "fawikisource-latest-pages-articles.xml.bz2")
FAWIKISOURCE_PROBED_BYTES = 47_719_980  # observed at probe 2026-11-25


def fawikisource() -> AcquisitionConfig:
    return AcquisitionConfig(
        source_name="fawikisource",
        output_prefix="fawikisource",
        law=FAS_PREP_LAW,
        adapter=WikiDumpConfig(
            dump_url=FAWIKISOURCE_DUMP_URL,
            user_agent="interscript-wo94-acquisition/0.1 "
                       "(wikisource CC BY-SA 4.0 corpus)",
            probed_bytes=FAWIKISOURCE_PROBED_BYTES,
            probed_bytes_key="probed_bytes_2026-11-25",
            redirect_prefixes=("#redirect", "#تغییرمسیر"),
        ),
        assemble=AssembleConfig(
            steps=(
                LenFilter(min_len=50, max_len=None, count_as="filtered_short"),
                ScriptFilter(min_fraction=0.5),
                DecontamFilter(),
            ),
            text_field="text",
            partner_field=None,
            kept_log_every=2_000,
        ),
        campaign=CAMPAIGN,
        source_meta={
            "site": "fa.wikisource.org",
            "dump": FAWIKISOURCE_DUMP_URL,
            "license": "CC BY-SA 4.0 (Wikimedia site terms; per-page attribution "
                       "via page history — title + pageid preserved per row)",
        },
        policy_meta={
            "unit": "page (ns=0 article, redirects excluded)",
            "wikitext": "comments/refs/tables dropped; templates flattened to "
                        "argument VALUES (poetry templates {{شعر|بیت|بیت}} keep "
                        "their verses); [[a|b]] -> b; headers unmarked; line "
                        "structure preserved",
            "filters": "text >= 50 chars; Arabic-letter fraction >= 0.5 of "
                       "non-space; exact-title dedupe",
            "decontam": "40-char windows (stride 20) over Arabic-letter normal "
                        "forms vs run-089-fas-poetry-eval + run-076 docs test "
                        "(both sides) — uniform with the ganjoor job",
            "sharding": "contiguous source-order slices",
        },
        run_dir_default="run-098-fas-acquisition/fawikisource",
        eval_mounts=EVAL_MOUNTS,
    )


# ---- cc-fa-web (Common Crawl WET slice; the OSCAR-fa SUBSTITUTE) ----

def ccfa_web() -> AcquisitionConfig:
    crawl = os.environ.get("FAS_CC_CRAWL", "CC-MAIN-2026-47")
    nseg = int(os.environ.get("FAS_CC_NSEG", "120"))
    cc = CcWetConfig(
        base_url="https://data.commoncrawl.org",
        crawl_id=crawl,
        n_segments=nseg,
        user_agent="interscript-wo94-acquisition/0.1 "
                   "(common-crawl-derived fa register slice)",
        # NOTE: "pbu" is a PASHTO code sitting in the Persian allowlist —
        # a pre-refactor quirk preserved verbatim (behavior-locked by the
        # equivalence test). Pashto docs passing this header check are
        # still subject to the letter floor + exclusive rejections.
        lang_allowlist=frozenset({"fas", "pes", "fa", "pbu"}),
        letter_floor=0.6,
        doc_min=50,
        doc_max=100_000,
        clean_doc_min=50,
        nav_junk_min_letters=3,
        nav_junk_max_len=60,
        exclusive_rejects=(
            ExclusiveReject(name="kurdish", letters="ڕڵۏێە", threshold=3),
            ExclusiveReject(name="urdu", letters="ٹڈڑںے", threshold=3),
        ),
        char_ceilings=(CharCeiling(char="ة", max_ratio=0.01),),
    )
    return AcquisitionConfig(
        source_name="cc-fa-web",
        output_prefix="cc-fa-web",
        law=FAS_PREP_LAW,
        adapter=cc,
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
        campaign=CAMPAIGN,
        source_meta={
            "origin": "Common Crawl WET, crawl " + crawl,
            "channel": f"{cc.base_url}/crawl-data/"
                       f"{crawl}/wet.paths.gz (deterministic stride sample, "
                       f"NSEG={nseg})",
            "oscar_note": "OSCAR-F SUBSTITUTE — canonical OSCAR releases are "
                          "upstream-suspended (gate text recorded verbatim in the "
                          "WO94 ledger); this slice reconstructs the dossier row's "
                          "register intent from the same underlying source "
                          "(Common-Crawl-derived) through OSCAR-style filters",
            "license": "LicenseRef-CommonCrawl-Terms (per-document terms; CC use/"
                       "redistribution policy) + license_review record citing the "
                       "owner's OSCAR-fa dossier framing — weak-pretrain mass ONLY, "
                       "never the quality/eval signal",
        },
        policy_meta={
            "unit": "web document (WARC-Type: conversion)",
            "langid": "WARC-Identified-Content-Language header (fas/pes/fa/pbu) "
                      "AND Persian-letter fraction >= 0.6 floor; script-exclusive "
                      "rejection of Kurdish (ڕڵۏێە) / Urdu (ٹڈڑںے) letters and "
                      "Arabic ta-marbuta-heavy docs",
            "clean": "lines with <3 Arabic letters and <60 chars dropped (nav junk); "
                     "blank runs collapsed",
            "filters": "50 <= len <= 100_000 chars",
            "dedupe": "exact-text dedupe, first occurrence wins (OSCAR-style "
                      "shingle dedupe deferred to training prep)",
            "decontam": "40-char windows (stride 20) vs both pinned fas evals "
                        "(uniform with the other acquisition jobs)",
            "sharding": "contiguous source-order slices",
        },
        run_dir_default="run-098-fas-acquisition/cc-fa-web",
        raw_subdir="segments",
        eval_mounts=EVAL_MOUNTS,
        count_extras=("kept_unique",),
    )


# ---- the gate (run-093 law; pre-registered WO94 thresholds) ----

def gate_dir():
    """The acquisition run-dir root the gate verifies (note: the gate's
    FAS_RUN_DIR default has NO source subpath — pre-refactor behavior)."""
    from pathlib import Path
    bucket = Path(os.environ.get("FAS_CKPT", "/ckpt"))
    return bucket / os.environ.get("FAS_RUN_DIR", "run-098-fas-acquisition")


def gate_sources() -> list[GateSource]:
    """The three pre-registered gate defs (WO94 ledger)."""
    zwnj = ZWNJ
    ganjoor_gate = GateDef(
        source_name="ganjoor",
        shard_prefix="ganjoor",
        pair_mode=True,
        text_key="src",
        partner_key="tgt",
        letter_floor=0.5,
        min_rows=0,
        checks=(
            GateCheck(key="ratio_band_all_rows", kind="ratio_band_all_rows",
                      band=(1.0, 2.0)),
            GateCheck(key="vocalized_share_min_0.30", kind="vocalized_share",
                      threshold=0.30),
            GateCheck(key="harakat_density_median_min_0.05",
                      kind="harakat_density_median", threshold=0.05),
            GateCheck(key="degenerate_lt_1pct", kind="degenerate_lt_1pct",
                      threshold=0.01),
            GateCheck(key="script_median_min_0.5", kind="script_median",
                      threshold=0.5),
            GateCheck(key="persian_specific_all_present",
                      kind="specific_all_present"),
            GateCheck(key="zwnj_present", kind="zwnj_present"),
        ),
        specific_letters=PERSIAN_SPECIFIC,
        zwnj=zwnj,
    )
    fawikisource_gate = GateDef(
        source_name="fawikisource",
        shard_prefix="fawikisource",
        pair_mode=False,
        text_key="text",
        partner_key=None,
        letter_floor=0.5,
        min_rows=500,
        checks=(
            GateCheck(key="rows_min_500", kind="rows_min", threshold=500),
            GateCheck(key="degenerate_lt_1pct", kind="degenerate_lt_1pct",
                      threshold=0.01),
            GateCheck(key="script_median_min", kind="script_median",
                      threshold=0.5),
            GateCheck(key="persian_specific_all_present",
                      kind="specific_all_present"),
        ),
        specific_letters=PERSIAN_SPECIFIC,
        zwnj=zwnj,
    )
    ccfa_gate = GateDef(
        source_name="cc-fa-web",
        shard_prefix="cc-fa-web",
        pair_mode=False,
        text_key="text",
        partner_key=None,
        letter_floor=0.6,
        min_rows=10_000,
        checks=(
            GateCheck(key="rows_min_10000", kind="rows_min", threshold=10_000),
            GateCheck(key="degenerate_lt_1pct", kind="degenerate_lt_1pct",
                      threshold=0.01),
            GateCheck(key="script_median_min", kind="script_median",
                      threshold=0.6),
            GateCheck(key="persian_specific_all_present",
                      kind="specific_all_present"),
        ),
        specific_letters=PERSIAN_SPECIFIC,
        zwnj=zwnj,
    )
    return [
        GateSource(subdir="ganjoor", gate=ganjoor_gate, law=FAS_GATE_LAW),
        GateSource(subdir="fawikisource", gate=fawikisource_gate,
                   law=FAS_GATE_LAW),
        GateSource(subdir="cc-fa-web", gate=ccfa_gate, law=FAS_GATE_LAW),
    ]

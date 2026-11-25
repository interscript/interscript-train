"""DRAFT — Jawi / Malay (ms-arab) acquisition recipe. DO NOT LAUNCH.

STATUS: ILLUSTRATIVE STARTER ONLY. Every source below is UNVERIFIED per
ROADMAP-2027 R3 (TODO.sota/ROADMAP-2027.md): no dedicated
diacritized-Jawi corpus surfaced in the survey; the plausible path is
mining fully-vocalized Jawi Quran editions (the Tashkeela pattern),
which MUST pass a mining-verification probe (zero-GPU) before anything
is staged. The URLs are placeholders shaped to the adapter contracts —
they demonstrate the factory wiring, not verified endpoints.

This config demonstrates "a new language = a new config module": the
factory machinery (adapters, chain, gate engine) is unchanged; only the
recipe differs from the Persian instance.

Open verification items (the R3 mining probe):
  - does a public, redistributable, FULLY-VOCALIZED Jawi Quran text
    exist, and how many unique vocalized units survive dedupe?
  - regional orthographic conventions (Malay vs Pattani vs Kelantan-
    Pallava legacy spellings) — the ScriptLaw below needs the probe's
    letter inventory before the ranges are final.
  - register coverage: a Quran-only model is not a Jawi model — a
    second, non-Quranic eval slice is mandatory before any SOTA claim.

License posture (per the R3 line): public-domain religious text is the
target; every source's license must be re-verified at probe time and
recorded here BEFORE acquisition runs.
"""

from __future__ import annotations

from ..charlaw import ScriptLaw
from ..config import (
    AcquisitionConfig,
    ApiWalkConfig,
    AssembleConfig,
    DedupeFilter,
    DecontamFilter,
    EvalMount,
    LenFilter,
    ScriptFilter,
    WikiDumpConfig,
)

STATUS = ("DRAFT — UNVERIFIED SOURCES (ROADMAP-2027 R3): illustrative "
          "starter, not a launch; requires the mining-verification probe")

# Arabic base + the Jawi extensions (TODO(R3-probe): verify the letter
# inventory — the extension set below is the commonly cited Jawi
# consonant inventory (pa ny nga ga va) and MUST be confirmed against
# mined text before launch).
ARABIC_BASE = ((0x0621, 0x063A), (0x0641, 0x064A), (0x0671, 0x06D3))
JAWI_EXTENSIONS = (
    (0x067E, 0x067E),  # پ pa
    (0x0683, 0x0683),  # ڃ ny
    (0x06AC, 0x06AC),  # ڬ ga
    (0x06AD, 0x06AD),  # ڭ nga
    (0x06A4, 0x06A4),  # ڤ va
    (0x06CF, 0x06CF),  # ۏ va (final form usage; probe to confirm)
)

JAWI_LAW = ScriptLaw(
    name="jawi-draft",
    letter_ranges=ARABIC_BASE + JAWI_EXTENSIONS,
    # Jawi vowel marks = the Arabic harakat set (probe to confirm the
    # dotted-vowel conventions before launch)
    strip_ranges=((0x064B, 0x0652), (0x0670, 0x0670)),
    density_ranges=((0x064B, 0x0652), (0x0670, 0x0670)),
)

# TODO(R3-probe): the eval mounts do not exist yet — an eval slice must
# be built from the mined source with strict held-out rules FIRST (the
# R3 eval line). The defaults below are the conventional mount paths a
# future probe would fill in.
EVAL_MOUNTS = (
    EvalMount(env="JAWI_EVAL_MINED", default="/evals/jawi-mined/eval.jsonl"),
)


def jawi_quran_mining() -> AcquisitionConfig:
    """The mined-vocalized-Jawi source (pairs). DRAFT: placeholder API
    endpoint; the R3 probe decides the real channel (API vs dump)."""
    return AcquisitionConfig(
        source_name="jawi-quran-draft",
        output_prefix="jawi-quran",
        law=JAWI_LAW,
        adapter=ApiWalkConfig(
            # PLACEHOLDER endpoint — replace with the probe's verified source
            base_url="https://jawi-source.invalid/api",
            user_agent="interscript-acquisition/0.1 (jawi DRAFT, unverified)",
            list_endpoint="/surahs",
            list_published_key="published",
            cat_endpoint="/surah/{cid}?verses=true",
            item_endpoint="/verse/{pid}",
            workers=4,
        ),
        assemble=AssembleConfig(
            steps=(
                LenFilter(min_len=10, max_len=300, count_as="filtered_len"),
                ScriptFilter(min_fraction=0.5),
                DedupeFilter(field_name="tgt"),
                DecontamFilter(),
            ),
            text_field="src",
            partner_field="tgt",
            raw_row_counter="verses_raw",
            kept_log_every=100_000,
        ),
        campaign="R3 Jawi mining DRAFT (ROADMAP-2027 R3; unverified source)",
        source_meta={
            "status": STATUS,
            "site": "UNVERIFIED — probe required before launch",
            "license": "UNVERIFIED — public-domain target (the R3 line); "
                       "re-verify per source at probe time",
        },
        policy_meta={
            "unit": "verse (pairs: src = undiacritized law strip; "
                    "tgt = fully-vocalized text)",
            "filters": "10 <= len(src) <= 300; Jawi-letter fraction >= 0.5; "
                       "ratio band set by the probe's calibration",
            "dedupe": "exact-tgt dedupe, first occurrence wins",
            "decontam": "40-char windows (stride 20) vs the mined eval slice "
                        "(held-out rules per the R3 eval line)",
            "sharding": "contiguous source-order slices",
        },
        run_dir_default="run-jawi-draft/quran",
        raw_subdir="raw",
        eval_mounts=EVAL_MOUNTS,
        measure_invariants=False,  # bands calibrated by the probe, not before
    )


def ms_wikisource_draft() -> AcquisitionConfig:
    """ms.wikisource dump — the Jawi-script pages layer. DRAFT: the
    Malay wikisource is predominantly Rumi; the Jawi-script share is an
    open question for the probe."""
    return AcquisitionConfig(
        source_name="ms-wikisource-draft",
        output_prefix="ms-wikisource",
        law=JAWI_LAW,
        adapter=WikiDumpConfig(
            dump_url="https://dumps.wikimedia.org/mswikisource/latest/"
                     "mswikisource-latest-pages-articles.xml.bz2",
            user_agent="interscript-acquisition/0.1 (ms wikisource DRAFT)",
            probed_bytes=-1,  # probe required (never launched with -1)
            probed_bytes_key="probed_bytes_PROBE",
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
        campaign="R3 Jawi mining DRAFT (ROADMAP-2027 R3; unverified share)",
        source_meta={
            "status": STATUS,
            "site": "ms.wikisource.org",
            "license": "CC BY-SA 4.0 (Wikimedia site terms)",
            "open_question": "Jawi-script page share in the Rumi-dominant "
                             "wikisource — probe measures before launch",
        },
        policy_meta={
            "unit": "page (ns=0 article, redirects excluded)",
            "wikitext": "comments/refs/tables dropped; templates flattened "
                        "to argument VALUES; line structure preserved",
            "filters": "text >= 50 chars; Jawi-letter fraction >= 0.5; "
                       "exact-title dedupe",
            "decontam": "40-char windows (stride 20) vs the Jawi eval slice",
            "sharding": "contiguous source-order slices",
        },
        run_dir_default="run-jawi-draft/wikisource",
        eval_mounts=EVAL_MOUNTS,
    )


# A Common-Crawl register slice (the cc-fa-web pattern) would follow the
# probe: `CcWetConfig(lang_allowlist=frozenset({"msa", "zsm", "ms"}), ...)
# with neighbor-script ExclusiveRejects (Urdu/Pashto letters never occur
# in Jawi) finalized from the probe's letter inventory. Not drafted
# until the mined source verifies.

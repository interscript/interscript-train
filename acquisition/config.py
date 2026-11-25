"""The acquisition config schema — the language recipe.

A corpus acquisition run is fully described by one `AcquisitionConfig`:
the script law, the adapter (and its parameters), the row shape, the
counted filter chain (its ORDER is load-bearing — Counter insertion order
becomes the manifest `counts` key order), the invariant measurement, the
shard/packaging layout, and the provenance blocks written verbatim into
manifest.json.

A new language acquisition = a new config module (see
`acquisition/configs/`); a new SOURCE TYPE additionally needs an adapter.
The Persian recipes in `acquisition/configs/fas.py` are the first
instance and the behavior-reference.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .charlaw import ScriptLaw


# ---- filter-chain steps (order in `AssembleConfig.steps` is the Counter
#      insertion order — the manifest counts key order) ----

@dataclass(frozen=True)
class LenFilter:
    """Length window on the row text field."""
    min_len: int
    max_len: int | None
    count_as: str  # counter key: "filtered_len" (ganjoor) / "filtered_short" (wikisource)


@dataclass(frozen=True)
class ScriptFilter:
    """Letter-fraction floor on the row text field."""
    min_fraction: float


@dataclass(frozen=True)
class RatioFilter:
    """len(partner)/len(text) band — pairs only (vocalization only ever
    ADDS marks, so the band is bounded below by 1.0)."""
    lo: float
    hi: float


@dataclass(frozen=True)
class DedupeFilter:
    """Exact-duplicate drop on `field_name`, first occurrence wins."""
    field_name: str


@dataclass(frozen=True)
class DecontamFilter:
    """Held-out window rejection on the letter normal form of the text field."""
    pass


FilterStep = LenFilter | ScriptFilter | RatioFilter | DedupeFilter | DecontamFilter


@dataclass(frozen=True)
class AssembleConfig:
    """The counted assemble-stage filter chain (two-phase sources) or the
    inline single-pass chain (wikisource dumps)."""
    steps: tuple[FilterStep, ...]
    text_field: str              # field the len/script/decontam steps apply to
    partner_field: str | None    # "tgt" for src/tgt pairs; None for text rows
    raw_row_counter: str | None = None  # per-raw-row counter ("verses_raw"/"docs_raw")
    count_chars_as: str | None = None   # per-row raw char counter ("chars_raw") or None
    kept_counter: str = "kept"
    kept_log_every: int = 0      # 0 = no progress logging


# ---- adapters ----

@dataclass(frozen=True)
class ApiWalkConfig:
    """REST collection walker (list -> category-tree DFS -> items), the
    ganjoor.net shape: per-entity raw jsonl + .done resume markers, a
    thread pool over item fetches, retrying GETs."""
    base_url: str
    user_agent: str
    list_endpoint: str            # e.g. "/poets"
    list_published_key: str       # entities must have this key truthy
    cat_endpoint: str             # "/cat/{cid}?poems=true&mainSections=false"
    item_endpoint: str            # "/poem/{pid}?catInfo=false&..."
    root_cat_key: str = "rootCatId"
    entity_url_key: str = "fullUrl"
    entity_id_key: str = "id"
    entity_name_keys: tuple[str, ...] = ("nickname", "name")
    cat_wrapper_key: str = "cat"
    cat_poems_key: str = "poems"
    cat_children_key: str = "children"
    poem_id_key: str = "id"
    poem_title_key: str = "title"
    child_slug_key: str = "urlSlug"
    verses_key: str = "verses"
    verse_text_key: str = "text"
    verse_order_key: str = "vOrder"
    verse_position_key: str = "versePosition"
    workers: int = 6
    http_timeout: int = 45
    http_tries: int = 6


@dataclass(frozen=True)
class WikiDumpConfig:
    """Wikimedia XML dump (bz2) single-pass adapter: ns=0 articles, skip
    redirects, dedupe by title (uncounted, pre-conversion), wikitext
    flattened to visible text."""
    dump_url: str
    user_agent: str
    probed_bytes: int             # probe-time size; manifest records the actual
    probed_bytes_key: str         # manifest key embedding the probe date
    media_ns: str = "{http://www.mediawiki.org/xml/export-0.11/}"
    article_ns: str = "0"
    redirect_prefixes: tuple[str, ...] = ("#redirect",)
    title_dedupe: bool = True
    cleanup_workdir: bool = True


@dataclass(frozen=True)
class ExclusiveReject:
    """Script-exclusive letter rejection: neighbor-script letters the
    target language never uses; >= threshold occurrences rejects the doc."""
    name: str
    letters: str
    threshold: int


@dataclass(frozen=True)
class CharCeiling:
    """Per-character share ceiling vs letters (e.g. Arabic ta-marbuta)."""
    char: str
    max_ratio: float


@dataclass(frozen=True)
class CcWetConfig:
    """Common Crawl WET slice adapter: deterministic stride sample of the
    crawl's wet.paths.gz, per-segment resume, WARC conversion records,
    header+script language law."""
    base_url: str
    crawl_id: str
    n_segments: int
    user_agent: str
    lang_allowlist: frozenset[str]
    letter_floor: float
    doc_min: int
    doc_max: int
    clean_doc_min: int
    nav_junk_min_letters: int
    nav_junk_max_len: int
    exclusive_rejects: tuple[ExclusiveReject, ...] = ()
    char_ceilings: tuple[CharCeiling, ...] = ()
    http_timeout: int = 300


AdapterConfig = ApiWalkConfig | WikiDumpConfig | CcWetConfig


# ---- invariant measurement (pairs sources; recorded in the manifest) ----

@dataclass(frozen=True)
class InvariantBands:
    """Pre-registered semantic-invariant thresholds, measured in-job and
    recorded in manifest.json under `invariants.gates`; ENFORCED by the
    separate gate job (whose own GateDef must mirror these)."""
    vocalized_share_min: float
    ratio_band: tuple[float, float]
    degenerate_max: float
    harakat_density_median_min: float


# ---- packaging / layout ----

@dataclass(frozen=True)
class EvalMount:
    """A held-out eval jsonl mounted read-only into the job."""
    env: str
    default: str


@dataclass(frozen=True)
class AcquisitionConfig:
    """One corpus acquisition run — the language recipe instance."""
    source_name: str              # "ganjoor" / "fawikisource" / "cc-fa-web"
    output_prefix: str            # shard file prefix
    law: ScriptLaw
    adapter: AdapterConfig
    assemble: AssembleConfig
    campaign: str                 # manifest "campaign" string
    source_meta: Mapping[str, object]   # manifest "source" block (verbatim)
    policy_meta: Mapping[str, object]   # manifest "policy" block (verbatim)

    # run layout (bucket envs — corpora never touch the laptop)
    bucket_env: str = "FAS_CKPT"
    bucket_default: str = "/ckpt"
    run_dir_env: str = "FAS_RUN_DIR"
    run_dir_default: str = ""
    raw_subdir: str | None = None       # "raw" (ganjoor) / "segments" (cc) / None
    n_shards_env: str = "FAS_SHARDS"
    n_shards_default: int = 8
    eval_mounts: tuple[EvalMount, ...] = ()
    crawl_snapshot_env: str | None = None  # env carrying the crawl snapshot date
    decontam_log_paths: bool = False  # per-eval-path window log lines (ganjoor)

    # manifest extras
    measure_invariants: bool = False
    invariant_bands: InvariantBands | None = None
    group_field: str | None = None      # per-group manifest block ("poet_url")
    count_extras: tuple[str, ...] = ()  # ("poet_files","kept_unique","degenerate")

    # layout quirks preserved per source
    shards_in_raw_dir: bool = False     # ganjoor writes shards next to the raw files

    # resolved paths (filled by `resolve_run`); kept out of equality
    _resolved: dict = field(default_factory=dict, repr=False, compare=False)

    def bucket(self) -> Path:
        import os
        return Path(os.environ.get(self.bucket_env, self.bucket_default))

    def run_dir(self) -> Path:
        import os
        rel = os.environ.get(self.run_dir_env, "") or self.run_dir_default
        return self.bucket() / rel

    def shard_dir(self) -> Path:
        run = self.run_dir()
        if self.shards_in_raw_dir:
            return run / (self.raw_subdir or "")
        return run

    def eval_paths(self) -> tuple[Path, ...]:
        import os
        return tuple(
            Path(os.environ.get(m.env, m.default)) for m in self.eval_mounts
        )

    def n_shards(self) -> int:
        import os
        return int(os.environ.get(self.n_shards_env, str(self.n_shards_default)))

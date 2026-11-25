"""Generic pipeline stages: filter chain, invariants, sharding, packaging.

The stages reproduce the pre-refactor behavior exactly, including the
manifest key order (dict insertion order) and the Counter-key order of
`counts` (driven by the filter-chain step order in the config).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from .charlaw import ScriptLaw
from .config import (
    AcquisitionConfig,
    AssembleConfig,
    DecontamFilter,
    DedupeFilter,
    LenFilter,
    RatioFilter,
    ScriptFilter,
)
from .decontam import hits_windows


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl_line(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False) + "\n"


# ---- the counted filter chain ----

def apply_chain(row: dict, asm: AssembleConfig, law: ScriptLaw,
                windows: set[str], counters: Counter,
                seen: set[str]) -> bool:
    """Run one raw row through the configured chain. Returns True when the
    row survives; increments the stage counters in chain order (which is
    the manifest counts key order). On survival the dedupe key enters
    `seen` (first occurrence wins)."""
    text = row[asm.text_field]
    for step in asm.steps:
        if isinstance(step, LenFilter):
            n = len(text)
            if not (step.min_len <= n and
                    (step.max_len is None or n <= step.max_len)):
                counters[step.count_as] += 1
                return False
        elif isinstance(step, ScriptFilter):
            letters = sum(1 for c in text if law.is_letter(c))
            if letters / max(1, len(text.replace(" ", ""))) < step.min_fraction:
                counters["filtered_script"] += 1
                return False
        elif isinstance(step, RatioFilter):
            partner = row[asm.partner_field]  # type: ignore[index]
            ratio = len(partner) / max(1, len(text))
            if not (step.lo <= ratio <= step.hi):
                counters["filtered_ratio"] += 1
                return False
        elif isinstance(step, DedupeFilter):
            if row[step.field_name] in seen:
                counters["dup_dropped"] += 1
                return False
        elif isinstance(step, DecontamFilter):
            norm = law.letter_norm(text)
            if hits_windows(norm, windows):
                counters["decontam_dropped"] += 1
                return False
    # survival: first occurrence wins — the dedupe key enters `seen`
    # AFTER the whole chain passes (the pre-refactor add position)
    for step in asm.steps:
        if isinstance(step, DedupeFilter):
            seen.add(row[step.field_name])
    return True


def assemble_from_raw(cfg: AcquisitionConfig, windows: set[str],
                      progress: Callable[[Counter], None] | None = None,
                      ) -> tuple[list[dict], Counter, int]:
    """Two-phase sources (api / crawl): read the adapter's raw jsonl files
    (sorted glob — source order), run the chain, return survivors,
    counters, and the raw-file count (`poet_files`)."""
    asm = cfg.assemble
    raw_dir = cfg.run_dir() / (cfg.raw_subdir or "")
    survivors: list[dict] = []
    counters: Counter = Counter()
    seen: set[str] = set()
    n_files = 0
    for raw_path in sorted(raw_dir.glob("*.jsonl")):
        n_files += 1
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if asm.raw_row_counter:
                    counters[asm.raw_row_counter] += 1
                if asm.count_chars_as:
                    counters[asm.count_chars_as] += len(r[asm.text_field])
                if apply_chain(r, asm, cfg.law, windows, counters, seen):
                    survivors.append(r)
                    counters[asm.kept_counter] += 1
                    if progress is not None:
                        progress(counters)
    return survivors, counters, n_files


# ---- pair-source invariants (recorded in the manifest, enforced by the gate) ----

def measure_pair_invariants(survivors: list[dict], law: ScriptLaw,
                            text_field: str, partner_field: str,
                            bands) -> tuple[dict, int]:
    """The ganjoor invariant block + the degenerate count (an extra
    counter). Density uses the law's density ranges (the prep job's own
    law — the gate law is separate and may be wider)."""
    vocalized = [r for r in survivors if r[partner_field] != r[text_field]]
    ratios = [len(r[partner_field]) / max(1, len(r[text_field]))
              for r in survivors]
    dens: list[float] = []
    for r in vocalized:
        letters = sum(1 for c in r[partner_field] if law.is_letter(c))
        m = sum(1 for c in r[partner_field] if law.is_density_mark(c))
        dens.append(m / max(1, letters))
    dens.sort()
    dens_mid = dens[len(dens) // 2] if dens else 0.0
    n = len(survivors)
    degenerate = sum(1 for r in survivors
                     if not r[text_field] or not r[partner_field])
    invariants = {
        "vocalized_share": round(len(vocalized) / max(1, n), 6),
        "ratio_min": round(min(ratios), 4) if ratios else None,
        "ratio_max": round(max(ratios), 4) if ratios else None,
        "harakat_density_median_vocalized": round(dens_mid, 6),
        "gates": {
            "vocalized_share_min": bands.vocalized_share_min,
            "ratio_band": list(bands.ratio_band),
            "degenerate_max": bands.degenerate_max,
            "harakat_density_median_min": bands.harakat_density_median_min,
        },
    }
    return invariants, degenerate


def group_block(survivors: list[dict], group_field: str,
                text_field: str, partner_field: str) -> dict:
    """Per-group rows + vocalized share (the ganjoor per_poet block)."""
    groups: dict[str, dict] = {}
    for r in survivors:
        e = groups.setdefault(r[group_field], {"rows": 0, "vocalized": 0})
        e["rows"] += 1
        e["vocalized"] += 1 if r[partner_field] != r[text_field] else 0
    return {k: {"rows": v["rows"],
                "vocalized_share": round(v["vocalized"] / max(1, v["rows"]), 4)}
            for k, v in sorted(groups.items())}


# ---- sharding + packaging ----

def write_shards(out_dir: Path, survivors: list[dict], prefix: str,
                 n_shards: int) -> list[dict]:
    """Contiguous source-order slices; empty shards are still written."""
    n = len(survivors)
    shard_meta = []
    for s in range(n_shards):
        lo, hi = n * s // n_shards, n * (s + 1) // n_shards
        name = f"{prefix}-{s:05d}-of-{n_shards:05d}.jsonl"
        p = out_dir / name
        with p.open("w", encoding="utf-8") as f:
            for r in survivors[lo:hi]:
                f.write(jsonl_line(r))
        shard_meta.append({"file": name, "rows": hi - lo,
                           "sha256": sha256_file(p)})
        log(f"[shard] {name} rows={hi - lo}")
    return shard_meta


def build_manifest(cfg: AcquisitionConfig, counters: Counter,
                   survivors: list[dict], shard_meta: list[dict],
                   n_files: int = 0) -> dict:
    """Manifest dict in the exact pre-refactor key order."""
    extras: dict = {}
    for key in cfg.count_extras:
        if key == "poet_files":
            extras["poet_files"] = n_files
        elif key == "kept_unique":
            extras["kept_unique"] = len(survivors)
        elif key == "degenerate":
            asm = cfg.assemble
            extras["degenerate"] = sum(
                1 for r in survivors
                if not r[asm.text_field] or not r[asm.partner_field]) \
                if asm.partner_field else 0
    manifest: dict = {
        "campaign": cfg.campaign,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": dict(cfg.source_meta),
        "policy": dict(cfg.policy_meta),
        "counts": dict(counters) | extras,
    }
    if cfg.measure_invariants and cfg.invariant_bands is not None:
        asm = cfg.assemble
        inv, _ = measure_pair_invariants(
            survivors, cfg.law, asm.text_field, asm.partner_field,  # type: ignore[arg-type]
            cfg.invariant_bands)
        manifest["invariants"] = inv
    if cfg.group_field:
        asm = cfg.assemble
        manifest["per_poet"] = group_block(
            survivors, cfg.group_field, asm.text_field,
            asm.partner_field)  # type: ignore[arg-type]
    manifest["shards"] = shard_meta
    return manifest


def write_packaging(out_dir: Path, manifest: dict,
                    shard_meta: list[dict]) -> None:
    """manifest.json + SHA256SUMS.txt (shards, then the manifest itself)
    + the DONE marker."""
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    sums = "".join(f"{m['sha256']}  {m['file']}\n" for m in shard_meta)
    sums += f"{sha256_file(out_dir / 'manifest.json')}  manifest.json\n"
    (out_dir / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    (out_dir / "DONE").write_text("ok\n", encoding="utf-8")

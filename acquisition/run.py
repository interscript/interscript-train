"""Run drivers — orchestrate adapter -> windows -> assemble -> packaging.

The flow order per source type reproduces the pre-refactor mains exactly
(windows before fetch for api/crawl; fetch before windows for dumps;
packaging after assemble; work-dir cleanup for dumps).
"""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from pathlib import Path

from .adapters.api_walk import fetch_stage as api_fetch_stage
from .adapters.cc_wet import fetch_stage as cc_fetch_stage
from .adapters.wiki_dump import fetch_dump, run_single_pass
from .charlaw import ScriptLaw
from .config import AcquisitionConfig, ApiWalkConfig, CcWetConfig, WikiDumpConfig
from .decontam import build_eval_windows
from .pipeline import (
    assemble_from_raw,
    build_manifest,
    log,
    sha256_file,
    write_packaging,
    write_shards,
)


def run_prep(cfg: AcquisitionConfig) -> dict:
    """Run one acquisition end-to-end; returns the manifest dict."""
    t0 = time.time()
    law: ScriptLaw = cfg.law
    run_dir = cfg.run_dir()
    raw_dir = run_dir / (cfg.raw_subdir or "")
    raw_dir.mkdir(parents=True, exist_ok=True)
    log(f"[work] {run_dir}")

    if isinstance(cfg.adapter, WikiDumpConfig):
        manifest = _run_dump(cfg, t0)
    else:
        # held-out windows first (the pre-refactor order for api/crawl)
        windows = build_eval_windows(
            cfg.eval_paths(), law,
            log=log if cfg.decontam_log_paths else None)
        log(f"[decontam] eval windows: {len(windows)}")
        if isinstance(cfg.adapter, ApiWalkConfig):
            api_fetch_stage(cfg, t0=t0)
        else:
            assert isinstance(cfg.adapter, CcWetConfig)
            cc_fetch_stage(cfg, t0)
        survivors, counters, n_files = assemble_from_raw(
            cfg, windows,
            progress=(_progress_logger(cfg.assemble.kept_log_every)
                      if cfg.assemble.kept_log_every else None))
        manifest = _package(cfg, survivors, counters, n_files)

    if cfg.measure_invariants and "invariants" in manifest:
        log("[manifest] " + json.dumps(
            {k: manifest[k] for k in ("counts", "invariants")},
            ensure_ascii=False))
    else:
        log("[manifest] counts="
            + json.dumps(manifest["counts"], ensure_ascii=False))
    log(f"[done] elapsed={time.time() - t0:.0f}s")
    return manifest


def _run_dump(cfg: AcquisitionConfig, t0: float) -> dict:
    dump: WikiDumpConfig = cfg.adapter  # type: ignore[assignment]
    assert isinstance(dump, WikiDumpConfig)
    run_dir = cfg.run_dir()
    work = run_dir / "work"
    work.mkdir(exist_ok=True)

    # 1. download + pin the dump (streamed to the bucket work dir)
    dump_path = work / dump.dump_url.rsplit("/", 1)[-1]
    fetch_dump(dump, dump_path)
    dump_sha = sha256_file(dump_path)
    dump_size = dump_path.stat().st_size
    if dump_size != dump.probed_bytes:
        log(f"[warn] dump size {dump_size} != probed {dump.probed_bytes} — "
            "upstream moved on; manifest records what was used")

    # 2. held-out windows (after fetch — the pre-refactor order)
    windows = build_eval_windows(cfg.eval_paths(), cfg.law)
    log(f"[decontam] eval windows: {len(windows)}")

    # 3. single-pass extract + counted chain
    survivors, counters = run_single_pass(cfg, windows, t0)

    # 4-5. packaging (manifest source block gains the runtime pin fields)
    source_meta = dict(cfg.source_meta)
    source_meta["sha256"] = dump_sha
    source_meta["bytes"] = dump_size
    source_meta[dump.probed_bytes_key] = dump.probed_bytes
    manifest = _package(cfg, survivors, counters, 0,
                        source_meta=source_meta)
    if dump.cleanup_workdir:
        shutil.rmtree(work, ignore_errors=True)
    return manifest


def _package(cfg: AcquisitionConfig, survivors: list[dict],
             counters: Counter, n_files: int,
             source_meta: dict | None = None) -> dict:
    shard_dir = cfg.shard_dir()
    shard_meta = write_shards(shard_dir, survivors, cfg.output_prefix,
                              cfg.n_shards())
    meta = dict(cfg.source_meta) if source_meta is None else source_meta
    cfg_view = cfg
    if source_meta is not None:
        # rebuild the manifest with the runtime source block (wiki pins)
        from dataclasses import replace
        cfg_view = replace(cfg, source_meta=meta)
    manifest = build_manifest(cfg_view, counters, survivors, shard_meta,
                              n_files)
    write_packaging(shard_dir, manifest, shard_meta)
    return manifest


def _progress_logger(every: int):
    def progress(counters: Counter) -> None:
        kept = counters.get("kept", 0)
        if every and kept and kept % every == 0:
            log(f"[assemble] kept={kept} dup={counters.get('dup_dropped', 0)} "
                f"decontam={counters.get('decontam_dropped', 0)}")
    return progress


def run_gates(acq_dir: Path, sources, campaign: str, law_line: str) -> None:
    """Run the independent gate; exits nonzero on FAIL (the run-093 law)."""
    from .gate import run_gate
    if not run_gate(acq_dir, list(sources), campaign, law_line):
        raise SystemExit(1)

"""The independent stats gate (the run-093 law).

Re-verifies the acquired artifacts FROM SOURCE (the bucket, not job
scratch): SHA256SUMS re-hash of every shard + manifest, then the
pre-registered semantic invariants per source. The gate definitions are
config (GateDef) — the Persian ones pre-registered in the WO94 ledger
are in `acquisition.configs.fas`.

Two verifier modes, exactly as pre-refactor:
  - pair mode (src/tgt verse rows): ratio band over EVERY non-degenerate
    row, vocalized share, harakat density (mean over vocalized rows —
    the pre-refactor `median`-keyed name preserved), degenerate share,
    letter-fraction median (nsp-guarded), specific-letter presence,
    ZWNJ presence.
  - text mode (document/page rows): row floor, degenerate share,
    letter-fraction median (appended unconditionally — pre-refactor
    behavior), specific-letter presence.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .charlaw import ScriptLaw
from .pipeline import log, sha256_file


@dataclass(frozen=True)
class GateCheck:
    """One named boolean check — the key is the LITERAL verdict key
    (thresholds are embedded in the pre-registered key strings)."""
    key: str
    kind: str  # ratio_band_all_rows | vocalized_share | harakat_density_median
               # | degenerate_lt_1pct | script_median | rows_min
               # | specific_all_present | zwnj_present
    threshold: float | None = None
    band: tuple[float, float] | None = None


@dataclass(frozen=True)
class GateDef:
    source_name: str
    shard_prefix: str          # shard glob prefix
    pair_mode: bool
    text_key: str              # "src" (pairs) / "text" (docs)
    partner_key: str | None    # "tgt" (pairs) / None
    letter_floor: float        # script_median check threshold
    min_rows: int              # 0 = no row-floor check
    checks: tuple[GateCheck, ...]
    specific_letters: str = ""
    zwnj: str | None = None
    letter_fraction_key: str = "arabic_letter_fraction_median"
    specific_letters_key: str = "persian_specific_letters"


@dataclass(frozen=True)
class GateSource:
    subdir: str                # run-dir subdirectory
    gate: GateDef
    law: ScriptLaw             # the GATE law (may differ from the prep law)


def check_sums(run_dir: Path) -> tuple[bool, list[str]]:
    problems = []
    sums_path = run_dir / "SHA256SUMS.txt"
    if not sums_path.exists():
        return False, ["SHA256SUMS.txt missing"]
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sha, name = line.split("  ", 1)
        p = run_dir / name
        if not p.exists():
            problems.append(f"{name}: missing")
            continue
        got = sha256_file(p)
        if got != sha:
            problems.append(f"{name}: sha mismatch {got} != {sha}")
    return not problems, problems


def _iter_shard_rows(run_dir: Path, prefix: str):
    for shard in sorted(run_dir.glob(f"{prefix}-*.jsonl")):
        with shard.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)


def _apply_checks(gdef: GateDef, med: float, dens_med: float,
                  ratio_min: float, ratio_max: float, rows: int,
                  vocalized: int, degenerate: int, zwnj: int,
                  spec: dict) -> dict:
    checks: dict[str, bool] = {}
    for chk in gdef.checks:
        if chk.kind == "ratio_band_all_rows":
            assert chk.band is not None
            val = ratio_min >= chk.band[0] and ratio_max <= chk.band[1]
        elif chk.kind == "vocalized_share":
            assert chk.threshold is not None
            val = (vocalized / rows) >= chk.threshold if rows else False
        elif chk.kind == "harakat_density_median":
            assert chk.threshold is not None
            val = dens_med >= chk.threshold
        elif chk.kind == "degenerate_lt_1pct":
            assert chk.threshold is not None
            val = (degenerate / rows) < chk.threshold if rows else False
        elif chk.kind == "script_median":
            assert chk.threshold is not None
            val = med >= chk.threshold
        elif chk.kind == "rows_min":
            assert chk.threshold is not None
            val = rows >= chk.threshold
        elif chk.kind == "specific_all_present":
            val = all(v > 0 for v in spec.values())
        elif chk.kind == "zwnj_present":
            val = zwnj > 0
        else:  # pragma: no cover — config error
            raise ValueError(f"unknown gate check kind: {chk.kind}")
        checks[chk.key] = val
    return checks


def verify_pairs(run_dir: Path, gdef: GateDef, law: ScriptLaw) -> dict:
    assert gdef.partner_key is not None
    res: dict = {"source": gdef.source_name, "checks": {}, "counts": {}}
    rows = vocalized = dens_sum = degenerate = zwnj = 0
    letters_all: list[float] = []
    spec = {c: 0 for c in gdef.specific_letters}
    ratio_min, ratio_max = 10.0, 0.0
    for r in _iter_shard_rows(run_dir, gdef.shard_prefix):
        src, tgt = r[gdef.text_key], r[gdef.partner_key]
        rows += 1
        if not src or not tgt:
            degenerate += 1
            continue
        ratio = len(tgt) / max(1, len(src))
        ratio_min = min(ratio_min, ratio)
        ratio_max = max(ratio_max, ratio)
        nsp = len(src.replace(" ", ""))
        if nsp:
            letters_all.append(law.letter_fraction(src))
        if tgt != src:
            vocalized += 1
            ls = sum(1 for c in tgt if law.is_letter(c))
            dens_sum += sum(1 for c in tgt if law.is_density_mark(c)) / max(1, ls)
        for c in spec:
            spec[c] += tgt.count(c)
        if gdef.zwnj is not None:
            zwnj += tgt.count(gdef.zwnj)
    letters_all.sort()
    med = letters_all[len(letters_all) // 2] if letters_all else 0.0
    dens_med = (dens_sum / vocalized) if vocalized else 0.0
    res["counts"] = {"rows": rows, "vocalized": vocalized,
                     "degenerate": degenerate, "zwnj": zwnj,
                     "ratio_min": round(ratio_min, 4),
                     "ratio_max": round(ratio_max, 4),
                     "harakat_density_median_vocalized": round(dens_med, 6),
                     gdef.letter_fraction_key: round(med, 6),
                     gdef.specific_letters_key: spec}
    res["checks"] = _apply_checks(gdef, med, dens_med, ratio_min, ratio_max,
                                  rows, vocalized, degenerate, zwnj, spec)
    return res


def verify_text_rows(run_dir: Path, gdef: GateDef, law: ScriptLaw) -> dict:
    res: dict = {"source": gdef.source_name, "checks": {}, "counts": {}}
    rows = degenerate = zwnj = 0
    letters_all: list[float] = []
    spec = {c: 0 for c in gdef.specific_letters}
    for r in _iter_shard_rows(run_dir, gdef.shard_prefix):
        text = r[gdef.text_key]
        rows += 1
        if not text or not text.strip():
            degenerate += 1
            continue
        nsp = len(text.replace(" ", ""))
        letters_all.append(sum(1 for c in text if law.is_letter(c)) / max(1, nsp))
        for c in spec:
            spec[c] += text.count(c)
        if gdef.zwnj is not None:
            zwnj += text.count(gdef.zwnj)
    letters_all.sort()
    med = letters_all[len(letters_all) // 2] if letters_all else 0.0
    res["counts"] = {"rows": rows, "degenerate": degenerate, "zwnj": zwnj,
                     gdef.letter_fraction_key: round(med, 6),
                     gdef.specific_letters_key: spec}
    res["checks"] = _apply_checks(gdef, med, 0.0, 10.0, 0.0, rows, 0,
                                  degenerate, zwnj, spec)
    return res


def run_gate(acq_dir: Path, sources: list[GateSource], campaign: str,
             law_line: str, write: Callable[[str], None] = log) -> bool:
    """Verify every source run dir; write gate-verdict.json + the
    GATE_PASS/GATE_FAIL marker. Returns the overall verdict."""
    write(f"[gate] {acq_dir}")
    verdict: dict = {
        "campaign": campaign,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": [],
        "law": law_line,
    }
    all_pass = True
    for src in sources:
        run_dir = acq_dir / src.subdir
        verifier = verify_pairs if src.gate.pair_mode else verify_text_rows
        entry: dict = {"run_dir": str(run_dir)}
        if not (run_dir / "DONE").exists():
            entry["verdict"] = "FAIL"
            entry["problems"] = ["DONE missing (acquisition incomplete)"]
            all_pass = False
            verdict["sources"].append(entry)
            continue
        sums_ok, problems = check_sums(run_dir)
        entry["sha256sums_verified"] = sums_ok
        entry["sha256_problems"] = problems
        entry["manifest_sha256"] = sha256_file(run_dir / "manifest.json")
        entry["manifest"] = {
            k: json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")).get(k)
            for k in ("counts", "invariants")}
        stats = verifier(run_dir, src.gate, src.law)
        entry["stats"] = stats
        gates_pass = all(stats["checks"].values())
        entry["checks"] = stats["checks"]
        entry["verdict"] = "PASS" if (sums_ok and gates_pass) else "FAIL"
        if entry["verdict"] == "FAIL":
            all_pass = False
        verdict["sources"].append(entry)
        write(f"[gate] {run_dir.name}: {entry['verdict']} "
              f"checks={json.dumps(stats['checks'])}")
    verdict["verdict"] = "PASS" if all_pass else "FAIL"
    (acq_dir / "gate-verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    (acq_dir / ("GATE_PASS" if all_pass else "GATE_FAIL")).write_text(
        json.dumps(verdict["verdict"]) + "\n", encoding="utf-8")
    write(f"[gate] VERDICT {verdict['verdict']}")
    return all_pass

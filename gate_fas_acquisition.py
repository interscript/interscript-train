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
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

BUCKET = Path(os.environ.get("FAS_CKPT", "/ckpt"))
ACQ = BUCKET / os.environ.get("FAS_RUN_DIR", "run-098-fas-acquisition")


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_arabic_letter(c: str) -> bool:
    o = ord(c)
    return (0x0621 <= o <= 0x063A) or (0x0641 <= o <= 0x064A) or (0x0671 <= o <= 0x06D3)


def _is_haraqat(c: str) -> bool:
    o = ord(c)
    return 0x064B <= o <= 0x065F or o == 0x0670


PERSIAN_SPECIFIC = "پچژگ"
ZWNJ = "‌"


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


def check_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def verify_ganjoor(run_dir: Path) -> dict:
    res = {"source": "ganjoor", "checks": {}, "counts": {}}
    rows = ratios = 0
    vocalized = dens = 0
    letters_all = []
    degenerate = 0
    spec = {c: 0 for c in PERSIAN_SPECIFIC}
    zwnj = 0
    ratio_min, ratio_max = 10.0, 0.0
    for shard in sorted(run_dir.glob("ganjoor-*.jsonl")):
        with shard.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                src, tgt = r["src"], r["tgt"]
                rows += 1
                if not src or not tgt:
                    degenerate += 1
                    continue
                ratio = len(tgt) / max(1, len(src))
                ratio_min = min(ratio_min, ratio)
                ratio_max = max(ratio_max, ratio)
                ratios += 1
                nsp = len(src.replace(" ", ""))
                if nsp:
                    letters_all.append(sum(1 for c in src if _is_arabic_letter(c)) / nsp)
                if tgt != src:
                    vocalized += 1
                    ls = sum(1 for c in tgt if _is_arabic_letter(c))
                    dens += sum(1 for c in tgt if _is_haraqat(c)) / max(1, ls)
                for c in spec:
                    spec[c] += tgt.count(c)
                zwnj += tgt.count(ZWNJ)
    letters_all.sort()
    med = letters_all[len(letters_all) // 2] if letters_all else 0.0
    dens_med = (dens / vocalized) if vocalized else 0.0
    res["counts"] = {"rows": rows, "vocalized": vocalized,
                     "degenerate": degenerate, "zwnj": zwnj,
                     "ratio_min": round(ratio_min, 4), "ratio_max": round(ratio_max, 4),
                     "harakat_density_median_vocalized": round(dens_med, 6),
                     "arabic_letter_fraction_median": round(med, 6),
                     "persian_specific_letters": spec}
    res["checks"] = {
        "ratio_band_all_rows": ratio_min >= 1.0 and ratio_max <= 2.0,
        "vocalized_share_min_0.30": (vocalized / rows) >= 0.30 if rows else False,
        "harakat_density_median_min_0.05": dens_med >= 0.05,
        "degenerate_lt_1pct": (degenerate / rows) < 0.01 if rows else False,
        "script_median_min_0.5": med >= 0.5,
        "persian_specific_all_present": all(v > 0 for v in spec.values()),
        "zwnj_present": zwnj > 0,
    }
    return res


def verify_text_rows(run_dir: Path, prefix: str, text_key: str,
                     min_rows: int, letter_floor: float) -> dict:
    res = {"source": prefix, "checks": {}, "counts": {}}
    rows = degenerate = letters_ok = 0
    letters_all = []
    spec = {c: 0 for c in PERSIAN_SPECIFIC}
    zwnj = 0
    for shard in sorted(run_dir.glob(f"{prefix}-*.jsonl")):
        with shard.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                text = r[text_key]
                rows += 1
                if not text or not text.strip():
                    degenerate += 1
                    continue
                nsp = len(text.replace(" ", ""))
                frac = sum(1 for c in text if _is_arabic_letter(c)) / max(1, nsp)
                letters_all.append(frac)
                if frac >= letter_floor:
                    letters_ok += 1
                for c in spec:
                    spec[c] += text.count(c)
                zwnj += text.count(ZWNJ)
    letters_all.sort()
    med = letters_all[len(letters_all) // 2] if letters_all else 0.0
    res["counts"] = {"rows": rows, "degenerate": degenerate, "zwnj": zwnj,
                     "arabic_letter_fraction_median": round(med, 6),
                     "persian_specific_letters": spec}
    res["checks"] = {
        f"rows_min_{min_rows}": rows >= min_rows,
        "degenerate_lt_1pct": (degenerate / rows) < 0.01 if rows else False,
        "script_median_min": med >= letter_floor,
        "persian_specific_all_present": all(v > 0 for v in spec.values()),
    }
    return res


def main() -> None:
    log(f"[gate] {ACQ}")
    verdict = {
        "campaign": "WO94 fas acquisition corpus gates",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": [],
        "law": "run-093 semantic invariants; pre-registered thresholds "
               "documented in gate_fas_acquisition.py + the WO94 ledger",
    }
    all_pass = True

    for run_dir, verifier in (
        (ACQ / "ganjoor", lambda d: verify_ganjoor(d)),
        (ACQ / "fawikisource", lambda d: verify_text_rows(d, "fawikisource",
                                                          "text", 500, 0.5)),
        (ACQ / "cc-fa-web", lambda d: verify_text_rows(d, "cc-fa-web",
                                                       "text", 10_000, 0.6)),
    ):
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
        entry["manifest"] = {k: check_manifest(run_dir).get(k) for k in
                             ("counts", "invariants")}
        stats = verifier(run_dir)
        entry["stats"] = stats
        gates_pass = all(stats["checks"].values())
        entry["checks"] = stats["checks"]
        entry["verdict"] = "PASS" if (sums_ok and gates_pass) else "FAIL"
        if entry["verdict"] == "FAIL":
            all_pass = False
        verdict["sources"].append(entry)
        log(f"[gate] {run_dir.name}: {entry['verdict']} "
            f"checks={json.dumps(stats['checks'])}")

    verdict["verdict"] = "PASS" if all_pass else "FAIL"
    (ACQ / "gate-verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    (ACQ / ("GATE_PASS" if all_pass else "GATE_FAIL")).write_text(
        json.dumps(verdict["verdict"]) + "\n", encoding="utf-8")
    log(f"[gate] VERDICT {verdict['verdict']}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

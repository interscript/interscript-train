# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=0.26",
# ]
# ///
"""WO94 corpus gate — ganjoor-vocalized (FRESH DEF, not an amendment).

A new gate definition, pre-registered BEFORE the extraction run
(checkpoint /tmp/factory-notes.md, WO94-SUBSET section) for the carved
vocalized subset corpus. The parent corpus gate (gate_fas_acquisition.py,
frozen) FAILED its vocalized_share floor on the parent and that verdict
stays verbatim as the falsification record; nothing here amends it.

Pre-registered checks (verbatim from the pre-registration):
  1. vocalized_share == 1.0 — every row's tgt carries >= 1 harakat of
     the FAS_GATE_LAW set U+064B–U+065F + U+0670 (the same set the
     extraction used for membership; by-construction invariant)
  2. ratio len(tgt)/len(src) in [1.0, 2.0] for every row (band preserved
     from the parent corpus)
  3. harakat_density_median >= 0.05 — TRUE median of marks-per-Arabic-
     letter over ALL rows (law set)
  4. degenerate (empty src/tgt) < 1%
  5. script sanity: Arabic-letter fraction median >= 0.5; Persian-specific
     letters (پ چ ژ گ) all present; ZWNJ present
  6. rows >= 100,000
  7. integrity: SHA256SUMS.txt re-hash clean (every shard + manifest);
     manifest.json present; DONE marker present

Writes gate-verdict.json + GATE_PASS / GATE_FAIL into the corpus dir;
exits 1 on FAIL.

Usage (cpu-upgrade, detached):
    hf jobs uv run --flavor cpu-upgrade -d --timeout 2h \
      -s HF_TOKEN \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      --name wo94-ganjoor-vocalized-gate gate_fas_ganjoor_vocalized.py
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

BUCKET = Path(os.environ.get("FAS_CKPT", "/ckpt"))
ACQ = BUCKET / os.environ.get("FAS_RUN_DIR", "run-098-fas-acquisition")
RUN_DIR = ACQ / "ganjoor-vocalized"


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_haraqat_law(c: str) -> bool:
    o = ord(c)
    return 0x064B <= o <= 0x065F or o == 0x0670


def _is_arabic_letter(c: str) -> bool:
    o = ord(c)
    return (0x0621 <= o <= 0x063A) or (0x0641 <= o <= 0x064A) or (0x0671 <= o <= 0x06D3)


PERSIAN_SPECIFIC = "پچژگ"
ZWNJ = "‌"


def main() -> None:
    log(f"[gate] {RUN_DIR}")
    entry: dict = {"source": "ganjoor-vocalized",
                   "law": "fresh pre-registered def (WO94-SUBSET), NOT an "
                          "amendment of the parent gate"}
    problems = []

    if not (RUN_DIR / "DONE").exists():
        problems.append("DONE missing (extraction incomplete)")
    manifest_path = RUN_DIR / "manifest.json"
    if not manifest_path.exists():
        problems.append("manifest.json missing")
    else:
        entry["manifest_sha256"] = sha256_file(manifest_path)

    # ---- integrity: re-hash every pinned file ----
    sums_ok = False
    if (RUN_DIR / "SHA256SUMS.txt").exists():
        sums_ok = True
        for line in (RUN_DIR / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sha, name = line.split("  ", 1)
            p = RUN_DIR / name
            if not p.exists():
                sums_ok = False
                problems.append(f"{name}: missing")
            elif sha256_file(p) != sha:
                sums_ok = False
                problems.append(f"{name}: sha mismatch")
    else:
        problems.append("SHA256SUMS.txt missing")
    entry["sha256sums_verified"] = sums_ok

    # ---- semantic checks over every row ----
    rows = vocalized = ratios = degenerate = zwnj = 0
    ratio_min, ratio_max = 10.0, 0.0
    dens: list[float] = []
    letters_all: list[float] = []
    spec = {c: 0 for c in PERSIAN_SPECIFIC}
    for shard in sorted(RUN_DIR.glob("ganjoor-vocalized-*.jsonl")):
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
                if any(_is_haraqat_law(c) for c in tgt):
                    vocalized += 1
                ratio = len(tgt) / max(1, len(src))
                ratio_min = min(ratio_min, ratio)
                ratio_max = max(ratio_max, ratio)
                ratios += 1
                nsp = len(src.replace(" ", ""))
                if nsp:
                    letters_all.append(sum(1 for c in src if _is_arabic_letter(c)) / nsp)
                ls = sum(1 for c in tgt if _is_arabic_letter(c))
                dens.append(sum(1 for c in tgt if _is_haraqat_law(c)) / max(1, ls))
                for c in spec:
                    spec[c] += tgt.count(c)
                zwnj += tgt.count(ZWNJ)

    letters_all.sort()
    dens.sort()
    letter_med = letters_all[len(letters_all) // 2] if letters_all else 0.0
    dens_med = dens[len(dens) // 2] if dens else 0.0

    entry["stats"] = {"rows": rows, "vocalized": vocalized, "degenerate": degenerate,
                      "zwnj": zwnj,
                      "ratio_min": round(ratio_min, 4), "ratio_max": round(ratio_max, 4),
                      "harakat_density_median_lawset": round(dens_med, 6),
                      "arabic_letter_fraction_median": round(letter_med, 6),
                      "persian_specific_letters": spec}
    entry["checks"] = {
        "vocalized_share_eq_1.0": rows > 0 and vocalized == rows,
        "ratio_band_all_rows": ratios == rows and ratio_min >= 1.0 and ratio_max <= 2.0,
        "harakat_density_median_min_0.05": dens_med >= 0.05,
        "degenerate_lt_1pct": (degenerate / rows) < 0.01 if rows else False,
        "script_median_min_0.5": letter_med >= 0.5,
        "persian_specific_all_present": all(v > 0 for v in spec.values()),
        "zwnj_present": zwnj > 0,
        "rows_min_100000": rows >= 100_000,
        "sha256sums_clean": sums_ok,
    }
    verdict = "PASS" if all(entry["checks"].values()) and not problems else "FAIL"
    entry["problems"] = problems
    out = {"campaign": "WO94 ganjoor-vocalized corpus gate",
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "verdict": verdict,
           "sources": [entry]}
    (RUN_DIR / "gate-verdict.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (RUN_DIR / ("GATE_PASS" if verdict == "PASS" else "GATE_FAIL")).write_text(
        verdict + "\n", encoding="utf-8")
    log("[gate] " + json.dumps(entry["checks"]))
    log(f"[gate] stats={json.dumps(entry['stats'], ensure_ascii=False)}")
    if problems:
        log(f"[gate] problems={json.dumps(problems)}")
    log(f"[gate] VERDICT {verdict}")
    if verdict == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

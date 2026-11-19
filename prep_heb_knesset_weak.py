# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "py7zr>=0.21",
#   "huggingface_hub>=0.26",
# ]
# ///
"""WO92 prep — Phonikud knesset_nikud_v6 weak-pretrain corpus builder.

Builds the stage-1 WEAK-PRETRAIN corpus for the Hebrew plane campaign
(TODO.sota/92): Dicta machine labels, so it is weak supervision ONLY —
never the quality/eval signal (no-teacher-poison rule).

Pipeline (all inside the job; corpora NEVER touch the laptop):
  1. Fetch knesset_nikud_v6.txt.7z from Phonikud/phonikud-data (MIT,
     verified via the dataset card). Record sha256 + size.
  2. Extract with py7zr; stream the single-column v6 lines
     (nikud + '|' syllable separators + stress/teamim marks).
  3. Normalize to plane-consumer targets: keep Hebrew letters, the
     nikud plane mark set (nikud_planes.NIKUD_MARKS — the trainer's own
     definition), ASCII, whitespace; drop '|' / teamim / stress /
     phonikud custom marks. src = target minus nikud marks.
  4. Filters (the s45 recipe, VERIFIED 2026-08-20): 10 <= len(src)
     <= 300 chars, Hebrew fraction >= 0.6 of non-space, nikud density
     >= len(src non-space)/12.
  5. Exact-input dedupe on the normalized skeleton (knesset is
     formulaic — the prior pass measured 2.48M of 5.3M exact
     duplicates). First occurrence wins; source order preserved.
  6. Decontamination vs the human-verified eval set (hebrew-v4
     test.jsonl): 40-char windows (stride 20) over Hebrew-letter-only
     normal forms of BOTH sides.
  7. Emit contiguous shards (source order), manifest.json with every
     count, and a SHA256SUMS sidecar for every emitted file.
  8. Upload to hf://datasets/{HEB_WEAK_REPO} (private) + an eval-set
     census (re-counted in-job from the mounted hebrew-v4).

Usage:
    hf jobs uv run --flavor cpu-upgrade -d --timeout 4h -s HF_TOKEN \
      -v hf://datasets/Interscript/hebrew-v4:/train_data:ro \
      --name heb-weakprep-data prep_heb_knesset_weak.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

WORK = Path(os.environ.get("HEB_WORK", "/scratch"))
TEST_DATA = Path(os.environ.get("HEB_TEST_DATA", "/train_data"))
REPO_ID = os.environ.get("HEB_WEAK_REPO", "Interscript/he-knesset-v6-weak")
N_SHARDS = int(os.environ.get("HEB_SHARDS", "8"))
ARCHIVE = "knesset_nikud_v6.txt.7z"
UPSTREAM = "Phonikud/phonikud-data"
UPSTREAM_URL = f"https://huggingface.co/datasets/{UPSTREAM}/resolve/main/{ARCHIVE}"

sys.path.insert(0, "/code")
from nikud_planes import NIKUD_MARKS  # noqa: E402  (the trainer's own mark set)

# s45 mark policy, aligned to the plane set: keep plane nikud marks,
# drop '|' syllable separators, teamim/stress, phonikud custom marks.
# (maqqaf U+05BE stays — nikud_planes keeps it in the skeleton.)
# The v6 separators, per phonikud's own source: PREFIX_CHAR = "|"
# (ASCII U+007C) and HATAMA_CHAR = U+05AB (stress). '|' is inside the
# ASCII-printable band, so it is dropped EXPLICITLY — the s45 comment
# said drop but its ASCII range accidentally kept it.
KEEP_PREDICATE = NIKUD_MARKS
DROP_CHARS = frozenset("|")


def _is_hebrew_letter(c: str) -> bool:
    return "א" <= c <= "ת"


def normalize_line(line: str) -> tuple[str, str] | None:
    """Raw v6 line -> (src skeleton, tgt plane target) or None if filtered."""
    tgt_chars = []
    for c in line:
        if c in DROP_CHARS:
            continue
        if c in KEEP_PREDICATE or _is_hebrew_letter(c) or c.isspace() or (33 <= ord(c) <= 126):
            tgt_chars.append(c)
        # drop: '|' (explicit), teamim/stress, phonikud custom marks, anything else
    target = "".join(tgt_chars).strip()
    src = "".join(c for c in target if c not in NIKUD_MARKS).strip()
    if not (10 <= len(src) <= 300):
        return None
    letters = sum(1 for c in src if _is_hebrew_letter(c))
    if letters / max(1, len(src.replace(" ", ""))) < 0.6:
        return None
    nikud = sum(1 for c in target if c in NIKUD_MARKS)
    if nikud < len(src.replace(" ", "")) / 12:
        return None
    return src, target


def _letter_norm(text: str) -> str:
    """Hebrew letters + single-spaced whitespace — the decontam normal form."""
    return re.sub(r"\s+", " ", "".join(c for c in text if _is_hebrew_letter(c) or c == " ")).strip()


def build_test_windows(path: Path) -> set[str]:
    """40-char windows (stride 20) over letter-normal forms of the eval set."""
    windows: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for side in (row.get("src", ""), row.get("tgt", "")):
                norm = _letter_norm(side)
                for i in range(0, max(1, len(norm) - 40), 20):
                    windows.add(norm[i : i + 40])
    return windows


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download
    import py7zr

    WORK.mkdir(parents=True, exist_ok=True)
    out_dir = WORK / "out"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    # ---- 1. fetch + pin the upstream archive ----
    archive_path = Path(hf_hub_download(UPSTREAM, ARCHIVE, repo_type="dataset"))
    arch_sha = sha256_file(arch_path)
    arch_size = archive_path.stat().st_size
    print(f"[fetch] {ARCHIVE} size={arch_size} sha256={arch_sha}", flush=True)
    KNOWN_SIZE = 142_845_067  # observed 2026-11-19; upstream re-upload changes this
    if arch_size != KNOWN_SIZE:
        print(f"[warn] archive size {arch_size} != pinned {KNOWN_SIZE} — "
              "upstream may have changed; manifest records what we used", flush=True)

    # ---- 2. extract ----
    txt_path = WORK / "knesset_nikud_v6.txt"
    if not txt_path.exists():
        print("[extract] ...", flush=True)
        with py7zr.SevenZipFile(archive_path, mode="r") as z:
            z.extractall(WORK)

    # ---- 3-6. normalize / filter / dedupe / decontam ----
    test_path = TEST_DATA / "test.jsonl"
    test_windows = build_test_windows(test_path)
    print(f"[decontam] {len(test_windows)} test windows", flush=True)

    seen_src: set[str] = set()
    seen_lines: set[str] = set()
    survivors: list[tuple[str, str]] = []
    n_lines = n_kept = n_dup = n_rawdup = n_contam = n_filtered = 0
    with txt_path.open(encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            stripped = line.strip()
            if not stripped:
                continue
            if stripped in seen_lines:
                n_rawdup += 1
            else:
                seen_lines.add(stripped)
            pair = normalize_line(stripped)
            if pair is None:
                n_filtered += 1
                continue
            src, target = pair
            if src in seen_src:
                n_dup += 1
                continue
            norm = _letter_norm(src)
            if any(norm[i : i + 40] in test_windows for i in range(0, max(1, len(norm) - 40), 20)):
                n_contam += 1
                continue
            seen_src.add(src)
            survivors.append((src, target))
            n_kept += 1
            if n_kept % 250_000 == 0:
                print(f"[prep] kept={n_kept} dup={n_dup} rawdup={n_rawdup} "
                      f"contam={n_contam} filtered={n_filtered}", flush=True)

    print(f"[prep] lines={n_lines} kept={n_kept} dup={n_dup} rawdup={n_rawdup} "
          f"contam={n_contam} filtered={n_filtered}", flush=True)

    # ---- 7. contiguous shards + manifest + sha256 sidecar ----
    shard_names = []
    shard_meta = []
    total = len(survivors)
    for s in range(N_SHARDS):
        lo = total * s // N_SHARDS
        hi = total * (s + 1) // N_SHARDS
        name = f"heb-knesset-weak-{s:05d}-of-{N_SHARDS:05d}.jsonl"
        p = out_dir / name
        with p.open("w", encoding="utf-8") as f:
            for src, target in survivors[lo:hi]:
                f.write(json.dumps({"src": src, "tgt": target}, ensure_ascii=False) + "\n")
        shard_names.append(name)
        shard_meta.append({"file": name, "rows": hi - lo, "sha256": sha256_file(p)})
        print(f"[shard] {name} rows={hi - lo}", flush=True)

    # eval-set census, re-counted in-job from the mounted human-verified set
    def count_rows(p: Path) -> int:
        n = 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    eval_census = {
        "test_rows": count_rows(test_path),
        "train_rows": count_rows(TEST_DATA / "train.jsonl"),
        "note": "test.jsonl is the held-out HUMAN-VERIFIED eval set "
                "(Nakdimon-derived; the crown heb-vocalized-plane-2.0 protocol)",
    }
    print(f"[census] {eval_census}", flush=True)

    manifest = {
        "campaign": "WO92 heb weak-pretrain (TODO.sota/92, interscript-models)",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "dataset": UPSTREAM,
            "file": ARCHIVE,
            "url": UPSTREAM_URL,
            "license": "MIT (dataset card cardData.license, verified 2026-11-19)",
            "sha256": arch_sha,
            "bytes": arch_size,
            "labels": "Dicta machine diacritizer + ~1k manual high-frequency "
                      "corrections — WEAK supervision only, never eval/quality",
            "format": "single-column v6: nikud + '|' syllable separators + "
                      "stress/teamim marks",
        },
        "policy": {
            "normalize": "keep Hebrew letters + nikud_planes.NIKUD_MARKS + "
                         "ASCII + whitespace; drop '|'/teamim/stress/custom marks",
            "filters": "10 <= len(src) <= 300 chars; Hebrew fraction >= 0.6 of "
                       "non-space; nikud density >= len(src non-space)/12 (s45)",
            "dedupe": "exact-input (normalized skeleton) dedupe, first "
                      "occurrence wins, source order preserved",
            "decontam": "40-char windows (stride 20) over Hebrew-letter normal "
                        "forms vs hebrew-v4 test.jsonl both sides",
            "sharding": "contiguous source-order slices",
        },
        "counts": {
            "lines_total": n_lines,
            "kept_unique": n_kept,
            "skeleton_dup_dropped": n_dup,
            "raw_line_dup_seen": n_rawdup,
            "contam_dropped": n_contam,
            "filter_dropped": n_filtered,
        },
        "shards": shard_meta,
        "eval_census": eval_census,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                             encoding="utf-8")

    sums_lines = [f"{m['sha256']}  {m['file']}" for m in shard_meta]
    sums_lines.append(f"{sha256_file(manifest_path)}  manifest.json")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sums_lines) + "\n",
                                            encoding="utf-8")

    readme = f"""---
license: mit
task_categories:
- text-classification
language:
- he
---

# heb-knesset-v6-weak — WO92 weak-pretrain corpus

Derived from [{UPSTREAM}]({UPSTREAM_URL}) `knesset_nikud_v6.txt.7z`
(MIT; sha256 `{arch_sha[:16]}…`). Labels are Dicta MACHINE diacritization
(+ ~1k manual high-frequency corrections): **weak supervision only —
never the quality or eval signal** (no-teacher-poison rule).

Normalization/filters/dedupe/decontam policy and every count are in
`manifest.json`; `SHA256SUMS.txt` pins all files.

- rows kept (unique skeletons): {n_kept}
- skeleton dups dropped: {n_dup}; raw line dups seen: {n_rawdup}
- decontam drops vs hebrew-v4 test.jsonl: {n_contam}
- eval census: {eval_census["test_rows"]} human-verified test rows

Consumed by the Hebrew plane trainer (run-093) via HfFileSystem mount.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    # ---- 8. upload ----
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=REPO_ID, repo_type="dataset", folder_path=out_dir)
    print(f"[upload] {REPO_ID} done", flush=True)

    summary = {"repo": REPO_ID, "kept": n_kept, "dup": n_dup,
               "rawdup": n_rawdup, "contam": n_contam,
               "filtered": n_filtered, "lines": n_lines,
               "test_rows": eval_census["test_rows"],
               "archive_sha256": arch_sha}
    print("[summary] " + json.dumps(summary), flush=True)
    (out_dir.parent / "PREP_DONE").write_text(json.dumps(summary, indent=1) + "\n",
                                              encoding="utf-8")


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=0.26",
# ]
# ///
"""WO94 subset extraction — ganjoor-vocalized (owner decision, 2026-11-25).

Carves the VOCALIZED rows out of the acquired Ganjoor corpus into a new
corpus dir (isx-training:/run-098-fas-acquisition/ganjoor-vocalized/).
The parent gate FAILED exactly one pre-registered check
(vocalized_share_min_0.30: measured 0.054141; verdict.json sha256
b20776e319e88011bb1d08c78c1eaeaeecba23b7e59c0d38ded2e3fc5973e7a5 — the
falsification record stays verbatim). Owner ruling: the vocalized subset
becomes the fas-poetry corpus; the plain verses stay in the bucket as an
UNSUPERVISED REGISTER (never deleted).

MEMBERSHIP SET, and why: a row is vocalized iff its tgt carries >= 1
harakat of the FAS_GATE_LAW set U+064B–U+065F + U+0670 — the SAME set the
frozen gate (gate_fas_acquisition.py / its frozen reference twin) counts
with. NOT the narrower pinned-eval STRIP set (U+064B–U+0652 + U+0670,
which keeps the hamza/maddah carriers): membership and the gate's density
measurement must use ONE population definition, otherwise the
share==1.0 check and the density-median check would measure different
row sets. The strip set remains only the src-derivation convention.

Packaging contract (mirrored from the parent, run-096b conventions):
contiguous shards in parent source order, manifest.json with the SAME
counter key order as the parent ganjoor manifest plus the invariants
block, SHA256SUMS.txt sidecar (every shard + the manifest), DONE marker.

Provenance recorded in the manifest: parent corpus manifest sha256,
gate-verdict.json sha256, license CC BY-SA 3.0 carried over (site-stated,
WO66 dossier record), per-row attribution fields passed through verbatim
(poet_url, poet_name, cat_path, poem_id, poem_title, vorder,
verse_position).

Pre-registered gate for this corpus (fresh def, NOT an amendment):
share == 1.0; ratio [1.0, 2.0] every row; harakat-density TRUE median
>= 0.05 (law set); degenerate < 1%; Arabic-letter median >= 0.5;
Persian-specific present; ZWNJ present; rows >= 100,000; sums re-hash
clean. Enforced by gate_fas_ganjoor_vocalized.py (separate job).

Usage (cpu-upgrade, detached):
    hf jobs uv run --flavor cpu-upgrade -d --timeout 4h \
      -s HF_TOKEN \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      --name wo94-ganjoor-vocalized-extract prep_fas_ganjoor_subset.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

BUCKET = Path(os.environ.get("FAS_CKPT", "/ckpt"))
ACQ = BUCKET / os.environ.get("FAS_RUN_DIR", "run-098-fas-acquisition")
PARENT = ACQ / "ganjoor"
OUT = ACQ / "ganjoor-vocalized"
N_SHARDS = int(os.environ.get("FAS_SHARDS", "8"))

# ---- the law sets (see docstring) ----
def _is_haraqat_law(c: str) -> bool:
    """FAS_GATE_LAW counting set: U+064B–U+065F + U+0670 (the frozen
    gate's set — membership is measured with what the gate measures)."""
    o = ord(c)
    return 0x064B <= o <= 0x065F or o == 0x0670


def _is_arabic_letter(c: str) -> bool:
    o = ord(c)
    return (0x0621 <= o <= 0x063A) or (0x0641 <= o <= 0x064A) or (0x0671 <= o <= 0x06D3)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    from huggingface_hub import hf_hub_download  # noqa: F401  (env sanity)

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[work] parent={PARENT} out={OUT}", flush=True)

    # ---- provenance pins from the parent tree ----
    parent_manifest_sha = sha256_file(PARENT / "manifest.json")
    verdict_sha = sha256_file(ACQ / "gate-verdict.json")
    print(f"[provenance] parent manifest sha256={parent_manifest_sha}", flush=True)
    print(f"[provenance] gate-verdict sha256={verdict_sha}", flush=True)

    # ---- filter pass over the parent shards (source order preserved) ----
    counts = {
        "verses_raw": 0,      # parent rows scanned (== parent kept 3,078,058)
        "kept": 0,            # vocalized rows emitted
        "filtered_len": 0,    # re-assert: len filter already held in parent
        "dup_dropped": 0,     # re-assert: parent exact-tgt dedupe held
        "decontam_dropped": 0,  # re-assert: parent decontam held
        "filtered_script": 0,   # re-assert: parent script filter held
        "poet_files": 0,      # poets with >= 1 vocalized row
        "kept_unique": 0,
        "degenerate": 0,
    }
    survivors: list[dict] = []
    seen_tgt: set[str] = set()
    poets: set[str] = set()
    for shard in sorted(PARENT.glob("ganjoor-*.jsonl")):
        with shard.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                counts["verses_raw"] += 1
                r = json.loads(line)
                tgt = r["tgt"]
                if not any(_is_haraqat_law(c) for c in tgt):
                    continue
                src = r["src"]
                if not (10 <= len(src) <= 300):
                    counts["filtered_len"] += 1
                    continue
                if tgt in seen_tgt:
                    counts["dup_dropped"] += 1
                    continue
                letters = sum(1 for c in src if _is_arabic_letter(c))
                if letters / max(1, len(src.replace(" ", ""))) < 0.5:
                    counts["filtered_script"] += 1
                    continue
                seen_tgt.add(tgt)
                survivors.append(r)
                poets.add(r["poet_url"])
                counts["kept"] += 1
                if counts["kept"] % 20_000 == 0:
                    print(f"[extract] kept={counts['kept']}", flush=True)
    counts["poet_files"] = len(poets)
    counts["kept_unique"] = len(survivors)
    counts["degenerate"] = sum(1 for r in survivors if not r["src"] or not r["tgt"])

    # ---- contiguous shards (parent source order) ----
    n = len(survivors)
    shard_meta = []
    for s in range(N_SHARDS):
        lo, hi = n * s // N_SHARDS, n * (s + 1) // N_SHARDS
        name = f"ganjoor-vocalized-{s:05d}-of-{N_SHARDS:05d}.jsonl"
        p = OUT / name
        with p.open("w", encoding="utf-8") as f:
            for r in survivors[lo:hi]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        shard_meta.append({"file": name, "rows": hi - lo, "sha256": sha256_file(p)})
        print(f"[shard] {name} rows={hi - lo}", flush=True)

    # ---- true-median density over ALL rows (law set) for the manifest ----
    dens = []
    for r in survivors:
        ls = sum(1 for c in r["tgt"] if _is_arabic_letter(c))
        m = sum(1 for c in r["tgt"] if _is_haraqat_law(c))
        dens.append(m / max(1, ls))
    dens.sort()
    dens_med = dens[len(dens) // 2] if dens else 0.0

    manifest = {
        "campaign": "WO94 subset extraction (owner decision 2026-11-25; "
                    "parent gate FAIL falsification record retained verbatim)",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "parent": "run-098-fas-acquisition/ganjoor (staged tree)",
            "parent_manifest_sha256": parent_manifest_sha,
            "gate_verdict_sha256": verdict_sha,
            "membership_set": "FAS_GATE_LAW haraqat U+064B-U+065F + U+0670 "
                              "(the frozen gate's counting set — NOT the "
                              "narrower pinned-eval strip set; membership and "
                              "gate density must measure one population)",
            "license": "CC BY-SA 3.0 (site-stated; WO66 dossier record, "
                       "carried over from the parent corpus)",
            "attribution_fields": ["poet_url", "poet_name", "cat_path", "poem_id",
                                   "poem_title"],
            "plain_register_note": "the unvocalized rows stay in the parent "
                                   "corpus dir as an UNSUPERVISED REGISTER "
                                   "(owner ruling; never deleted)",
        },
        "policy": {
            "unit": "verse (parent rows with >= 1 law-set harakat in tgt)",
            "filters": "re-assert parent invariants: 10 <= len(src) <= 300; "
                       "Arabic-letter fraction >= 0.5; exact-tgt dedupe "
                       "(expected 0 new drops — parent already applied)",
            "sharding": "contiguous parent-source-order slices",
        },
        "counts": counts,
        "invariants": {
            "vocalized_share": 1.0 if n else 0.0,
            "harakat_density_median_all_rows_lawset": round(dens_med, 6),
            "gates": {
                "vocalized_share": 1.0,
                "ratio_band": [1.0, 2.0],
                "harakat_density_median_min": 0.05,
                "degenerate_max": 0.01,
                "rows_min": 100000,
                "arabic_letter_fraction_median_min": 0.5,
            },
        },
        "shards": shard_meta,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    sums = "".join(f"{m['sha256']}  {m['file']}\n" for m in shard_meta)
    sums += f"{sha256_file(OUT / 'manifest.json')}  manifest.json\n"
    (OUT / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    (OUT / "DONE").write_text("ok\n", encoding="utf-8")
    print("[manifest] " + json.dumps({"counts": counts, "invariants": manifest["invariants"]},
                                     ensure_ascii=False), flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()

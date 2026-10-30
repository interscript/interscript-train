"""WO64/B3 Option-B stage 1 — oracle relabel via the two surviving SHIPPED arms.

The WO39 script (distill_oracle_hf.py) decodes PyTorch arm checkpoints that
the storage layer no longer holds (the WO64 verdict); the Option-B arms are
the shipped IMF artifacts themselves:

  r7   = /ckpt/b3-arms/ara-diac-2.0-fp32.zip  ByT5-base fp32, kv decoder
  news = /ckpt/b3-arms/ara-diac-news-1.0.zip  ByT5-small int8, plain decoder

Decoding is ONNX with LENGTH-BUCKETED batches: every row in a batch has an
identical encoded length, so the encoder never sees padding and the WO51
unmasked-encoder hazard cannot apply. Rows follow run-047's corpus recipe
exactly (article shuffle seed 7, row shuffle seed 42, WINDOW=600,
morpheme-slash strip); the per-row oracle is the arm prediction with minimum
char edit distance to the corpus silver, ties -> r7.

Env: D_RUN (default run-084-oracle-relabel-b), D_SUBSET (default 300000),
D_SMOKE (set -> run ~100 rows end to end and exit), D_BS (max batch,
default 48).

Durability: per-arm raw preds are appended keyed by row index and
force-synced to the bucket every ~4k decoded rows; on relaunch each arm
resumes from its synced raw file. Consolidation runs only when both arms
cover all rows.

Usage (a100-large):
    python /job/oracle_relabel_onnx.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import zipfile
from pathlib import Path

import numpy as np

DATA = Path("/train_data")
CKPT = Path("/ckpt") / os.environ.get("D_RUN", "run-084-oracle-relabel-b")
WINDOW = 600
PSEUDO_CAP = 900_000
SUBSET = int(os.environ.get("D_SUBSET", "300000"))
SMOKE = bool(os.environ.get("D_SMOKE"))
BS = int(os.environ.get("D_BS", "48"))
ARMS = [
    ("r7", "/ckpt/b3-arms/ara-diac-2.0-fp32.zip"),
    ("news", "/ckpt/b3-arms/ara-diac-news-1.0.zip"),
]
DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")

BYTE_OFFSET = 3
PAD_ID = 0
EOS_ID = 1


def encode(text: str) -> list[int]:
    return [b + BYTE_OFFSET for b in text.encode("utf-8")] + [EOS_ID]


def decode_ids(ids: list[int]) -> str:
    out = bytearray()
    for t in ids:
        if t == EOS_ID:
            break
        if t in (PAD_ID, 2):
            continue
        out.append((t - BYTE_OFFSET) % 256)
    return out.decode("utf-8", errors="replace")


def lev(a: str, b: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein

        return Levenshtein.distance(a, b)
    except ImportError:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                               prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


class Arm:
    def __init__(self, name: str, zip_path: str, force_cpu: bool = False,
                 intra_threads: int = 0):
        import onnxruntime as ort

        self.name = name
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_threads:
            opts.intra_op_num_threads = intra_threads
        graphs, digests = {}, {}
        with zipfile.ZipFile(zip_path) as z:
            for member in z.namelist():
                if member.endswith(".onnx"):
                    graphs[member] = z.read(member)
                elif member == "metadata.yaml":
                    meta = z.read(member).decode("utf-8")
        in_sha = False
        for line in meta.splitlines():
            if line.strip() == "sha256:":
                in_sha = True
                continue
            if in_sha:
                m = re.match(r"\s+([\w.-]+): ([0-9a-f]{64})", line)
                if m:
                    digests[m.group(1)] = m.group(2)
                elif line.strip() and not line.startswith(" "):
                    in_sha = False
        for fname, blob in graphs.items():
            want = digests.get(fname)
            if want:
                got = hashlib.sha256(blob).hexdigest()
                if got != want:
                    raise RuntimeError(f"{zip_path}:{fname} sha256 mismatch")
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        if force_cpu:
            providers = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(graphs["encoder.onnx"], opts, providers)
        dname = ("decoder-kv.onnx"
                 if "decoder-kv.onnx" in graphs else "decoder.onnx")
        self.kv = "decoder-kv.onnx" in graphs
        self.dec = ort.InferenceSession(graphs[dname], opts, providers)
        print(f"[{name}] EP={self.enc.get_providers()} kv={self.kv}",
              flush=True)
        self.out_names = [o.name for o in self.dec.get_outputs()]
        self.pasts = {}
        for m in self.dec.get_inputs():
            if m.name.startswith("past_"):
                shape = m.shape  # [batch, heads, past_seq, d_kv]
                heads = shape[1] if isinstance(shape[1], int) else 4
                d_kv = shape[3] if isinstance(shape[3], int) else 8
                self.pasts[m.name] = np.zeros((1, heads, 0, d_kv), np.float32)

    def decode_equal(self, srcs: list[str]) -> list[str]:
        """Greedy decode a batch of identical-encoded-length rows."""
        ids = [encode(s) for s in srcs]
        n = len(srcs)
        enc = np.array(ids, dtype=np.int64)
        hidden = self.enc.run(None, {"input_ids": enc})[0]
        max_len = min(1300, 2 * enc.shape[1] + 8)
        outs: list[list[int]] = [[] for _ in range(n)]
        alive = np.ones(n, dtype=bool)
        if self.kv:
            pasts = {k: np.repeat(z, n, axis=0) for k, z in self.pasts.items()}
            current = np.full((n, 1), PAD_ID, dtype=np.int64)
            for _ in range(max_len):
                res = dict(zip(self.out_names, self.dec.run(
                    None, {"input_ids": current,
                           "encoder_hidden_states": hidden, **pasts})))
                tok = np.argmax(res["logits"][:, -1, :], axis=1)
                for b in np.nonzero(alive)[0]:
                    t = int(tok[b])
                    if t == EOS_ID:
                        alive[b] = False
                    else:
                        outs[b].append(t)
                if not alive.any():
                    break
                pasts = {name: res[name.replace("past_", "present_", 1)]
                         for name in pasts}
                current = tok.reshape(n, 1)
        else:
            current = np.full((n, 1), PAD_ID, dtype=np.int64)
            for _ in range(max_len):
                logits = self.dec.run(
                    None, {"input_ids": current,
                           "encoder_hidden_states": hidden})[0]
                tok = np.argmax(logits[:, -1, :], axis=1)
                for b in np.nonzero(alive)[0]:
                    t = int(tok[b])
                    if t == EOS_ID:
                        alive[b] = False
                    else:
                        outs[b].append(t)
                if not alive.any():
                    break
                current = np.concatenate([current, tok.reshape(n, 1)], axis=1)
        return [decode_ids(o) for o in outs]


def split_windows(text: str, budget: int = WINDOW) -> list[str]:
    if len(text.encode("utf-8")) <= budget:
        return [text]
    words, cur, n, wins = text.split(), [], 0, []
    for w in words:
        c = len(w.encode("utf-8")) + 1
        if cur and n + c > budget:
            wins.append(" ".join(cur))
            cur, n = [], 0
        cur.append(w)
        n += c
    if cur:
        wins.append(" ".join(cur))
    return wins


def build_rows() -> list[tuple[str, str]]:
    """run-047's exact qcri-only silver corpus, stripped, shuffled."""
    qpath = DATA / "qcri-wiki.jsonl"
    articles = qpath.read_text(encoding="utf-8").splitlines()
    rng2 = random.Random(7)
    rng2.shuffle(articles)
    pseudo: list[tuple[str, str]] = []

    def add_silver(tgt: str) -> None:
        tgt = tgt.strip()
        src = DIACRITICS_RE.sub("", tgt).strip()
        if len(src) >= 5 and len(src.encode()) <= WINDOW - 50:
            pseudo.append((src, tgt))

    for aline in articles:
        if len(pseudo) >= PSEUDO_CAP:
            break
        if not aline.strip():
            continue
        art = json.loads(aline).get("text", "")
        for para in art.split("\n"):
            para = para.strip()
            if not para:
                continue
            if len(para.encode()) <= WINDOW - 50:
                add_silver(para)
            else:
                for w in split_windows(para, WINDOW - 50):
                    add_silver(w)
            if len(pseudo) >= PSEUDO_CAP:
                break
    stripped = []
    for _src, tgt in pseudo:
        tgt2 = " ".join(t.split("/", 1)[0] for t in tgt.split())
        src2 = DIACRITICS_RE.sub("", tgt2).strip()
        if len(src2) >= 5:
            stripped.append((src2, tgt2))
    rng = random.Random(42)
    rng.shuffle(stripped)
    rows = stripped[:SUBSET]
    print(f"[data] stripped={len(stripped)} subset={len(rows)}", flush=True)
    return rows


def sync_to_bucket(path: Path) -> None:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    dst = f"buckets/Interscript/isx-training/{path.relative_to(Path('/ckpt'))}"
    with open(path, "rb") as f, fs.open(dst, "wb") as g:
        g.write(f.read())
    if not fs.exists(dst):
        raise RuntimeError(f"sync verify failed: {dst}")


def decode_groups(nbytes: list[int], done: set[int],
                  bs: int = BS) -> list[list[int]]:
    """Deterministic length-grouped decode plan over remaining rows."""
    by_len: dict[int, list[int]] = {}
    for i, nb in enumerate(nbytes):
        if i not in done:
            by_len.setdefault(nb, []).append(i)
    groups = []
    for nb in sorted(by_len):
        chunk = by_len[nb]
        for j in range(0, len(chunk), bs):
            groups.append(chunk[j:j + bs])
    sizes = [len(g) for g in groups]
    print(f"[plan] groups={len(groups)} rows={sum(sizes)} "
          f"bs={bs} mean={sum(sizes) / max(1, len(sizes)):.1f}", flush=True)
    return groups


def decode_arm(arm: Arm, srcs: list[str], skip: set[int] | None = None,
               pool=None) -> None:
    """Rotating part files (the khm-labeler pattern): the mount only
    reliably flushes closed files, and HfFileSystem API sync 401s from
    inside jobs — so durability = new part file every CHUNK rows.

    skip: rows that need no decode (news cannot beat an exact r7 match;
    ties go to r7) — they are treated as decoded for planning/resume.
    pool: optional multiprocessing pool of CPU workers (int8 plain arm
    is fastest as parallel CPU single-row; see the bench verdict).
    """
    CHUNK = 2000
    skip = skip or set()
    parts_dir = CKPT / f"parts-{arm.name}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    for part in sorted(parts_dir.glob("part-*.jsonl")):
        for line in part.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["i"])
    n_decodable = len(srcs) - len(skip)
    if done:
        print(f"[{arm.name}] resume: {len(done)}/{n_decodable} rows decoded",
              flush=True)
    nbytes = [len(s.encode()) + 1 for s in srcs]
    # plain-decoder exports are batch=1-traced (parity test: batched
    # plain decode corrupts; only the kv graph is batch-safe) — so the
    # int8 news arm always decodes single-row, exactly like the runtime
    groups = decode_groups(nbytes, done | skip, BS if arm.kv else 1)
    t0 = time.time()
    n_done = 0
    part_idx = len(list(parts_dir.glob("part-*.jsonl")))
    f = (parts_dir / f"part-{part_idx:03d}.jsonl").open(
        "w", encoding="utf-8")

    def write(i: int, p: str) -> None:
        nonlocal n_done, part_idx, f
        f.write(json.dumps({"i": i, "p": p}, ensure_ascii=False) + "\n")
        n_done += 1
        if n_done % CHUNK == 0:
            f.close()
            part_idx += 1
            f = (parts_dir / f"part-{part_idx:03d}.jsonl").open(
                "w", encoding="utf-8")
            print(f"[{arm.name}] part {part_idx - 1:03d} flushed",
                  flush=True)

    if pool is not None:
        jobs = [(i, srcs[i]) for g in groups for i in g]
        for i, p in pool.imap_unordered(_pool_worker, jobs, chunksize=4):
            write(i, p)
            if n_done % 100 == 0:
                r = n_done / max(1, time.time() - t0)
                print(f"[{arm.name}] {n_done}/{n_decodable} "
                      f"({r:.1f} rows/s)", flush=True)
        f.close()
        print(f"[{arm.name}] done ({n_done} this session)", flush=True)
        return

    for gi, g in enumerate(groups):
        preds = arm.decode_equal([srcs[i] for i in g])
        for i, p in zip(g, preds):
            write(i, p)
        if gi % 20 == 0:
            r = n_done / max(1, time.time() - t0)
            rem = n_decodable - len(done) - n_done
            print(f"[{arm.name}] {n_done}/{n_decodable} "
                  f"({r:.1f} rows/s, eta {rem / max(r, 1) / 3600:.1f}h)",
                  flush=True)
    f.close()
    print(f"[{arm.name}] done ({n_done} this session)", flush=True)


_WORKER_ARM = None


def _pool_worker(job: tuple[int, str]) -> tuple[int, str]:
    global _WORKER_ARM
    if _WORKER_ARM is None:
        _WORKER_ARM = Arm("news", ARMS[-1][1], force_cpu=True,
                          intra_threads=2)
    i, src = job
    return i, _WORKER_ARM.decode_equal([src])[0]


def raw_preds(name: str, n: int, skip: set[int] | None = None):
    skip = skip or set()
    parts_dir = CKPT / f"parts-{name}"
    by_i: dict[int, str] = {}
    for part in sorted(parts_dir.glob("part-*.jsonl")):
        for line in part.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                by_i[r["i"]] = r["p"]
    have = len(by_i) + len(skip - set(by_i))
    if have < n:
        print(f"[{name}] raw covers {have}/{n} rows", flush=True)
        return None
    return [by_i.get(i) for i in range(n)]


def consolidate(rows: list[tuple[str, str]],
                preds: dict[str, list[str | None]]) -> None:
    with (CKPT / "arm-preds.jsonl").open("w", encoding="utf-8") as f, \
         (CKPT / "oracle-targets.jsonl").open("w", encoding="utf-8") as g:
        cers = {n: 0.0 for n, _ in ARMS}
        eval_n = {n: 0 for n, _ in ARMS}
        oracle_cer = 0.0
        disagree = 0
        for i, (src, silver) in enumerate(rows):
            row_preds = {n: preds[n][i] for n, _ in ARMS}
            if len({p for p in row_preds.values() if p is not None}) > 1:
                disagree += 1
            primary = ARMS[0][0]
            best_name, best_d = primary, lev(row_preds[primary], silver)
            best_tgt = row_preds[primary]
            for n, _ in ARMS[1:]:
                if row_preds[n] is None:  # skipped: cannot beat the tie
                    continue
                d = lev(row_preds[n], silver)
                if d < best_d:
                    best_name, best_d, best_tgt = n, d, row_preds[n]
            for n in cers:
                if row_preds[n] is None:
                    continue
                cers[n] += lev(row_preds[n], silver) / max(1, len(silver))
                eval_n[n] += 1
            oracle_cer += best_d / max(1, len(silver))
            f.write(json.dumps({"src": src, "silver": silver,
                                **row_preds}, ensure_ascii=False) + "\n")
            g.write(json.dumps({"src": src, "tgt": best_tgt,
                                "pick": best_name}, ensure_ascii=False) + "\n")
    n = len(rows)
    picks: dict[str, int] = {}
    for l in (CKPT / "oracle-targets.jsonl").read_text(
            encoding="utf-8").splitlines():
        p = json.loads(l)["pick"]
        picks[p] = picks.get(p, 0) + 1
    stats = {"rows": n, "disagreement": round(disagree / n, 4),
             "arm_cer": {k: round(v / max(1, eval_n[k]), 4)
                         for k, v in cers.items()},
             "arm_eval_rows": eval_n,
             "oracle_cer": round(oracle_cer / n, 4), "picks": picks}
    print(json.dumps(stats), flush=True)
    (CKPT / "relabel_stats.json").write_text(json.dumps(stats, indent=1))


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    srcs = [s for s, _ in rows]
    silvers = [t for _, t in rows]
    primary_name, primary_zip = ARMS[0]
    primary = Arm(primary_name, primary_zip)
    decode_arm(primary, srcs)
    p_primary = raw_preds(primary_name, len(rows))
    if p_primary is None:
        print(f"[consolidate] waiting on arm {primary_name}", flush=True)
        return
    skip = {i for i in range(len(rows)) if lev(p_primary[i], silvers[i]) == 0}
    print(f"[skip] {len(skip)}/{len(rows)} rows exact under {primary_name} "
          f"— news cannot beat a tie; skipped", flush=True)
    (CKPT / "skip.json").write_text(json.dumps(sorted(skip)))
    mode = os.environ.get("D_NEWS_MODE", "gpu")
    if mode == "cpu":
        news = Arm(ARMS[-1][0], ARMS[-1][1], force_cpu=True,
                   intra_threads=4)
    else:
        news = Arm(ARMS[-1][0], ARMS[-1][1])
    pool = None
    if mode == "cpu_par":
        import multiprocessing as mp

        # spawn, not fork: the parent already initialized the CUDA
        # context for the primary arm; forked children must not inherit it
        pool = mp.get_context("spawn").Pool(
            int(os.environ.get("D_NEWS_PROCS", "8")))
    try:
        decode_arm(news, srcs, skip=skip, pool=pool)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    preds = {primary_name: p_primary}
    for name, _ in ARMS[1:]:
        p = raw_preds(name, len(rows), skip=skip)
        if p is None:
            print(f"[consolidate] waiting on arm {name}", flush=True)
            return
        preds[name] = p
    consolidate(rows, preds)
    (CKPT / "RELABEL_DONE").write_text("ok\n")
    try:
        sync_to_bucket(CKPT / "RELABEL_DONE")
    except Exception as e:  # noqa: BLE001
        print(f"sync failed: {e}", flush=True)


if __name__ == "__main__":
    if SMOKE:
        os.environ["D_SUBSET"] = "100"
        SUBSET = 100
    main()
    if SMOKE:
        (CKPT / "SMOKE_DONE").write_text("ok\n")
        try:
            sync_to_bucket(CKPT / "SMOKE_DONE")
        except Exception:
            pass

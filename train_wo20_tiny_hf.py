# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.5.1",
#   "transformers==4.46.3",
#   "accelerate>=1.1.0",
#   "pandas",
#   "pyarrow",
#   "pyarabic",
#   "prettytable",
# ]
# ///
"""WO20 campaign A — r7 -> tiny student via the microkimi recipe (issue #441).

Stage scan (WO20_STAGE=scan, t4/a10g):
  downloads + verifies the r7 teacher (release teachers-arabic-v1,
  per-part sha256) and the canonical labels (teacher_labels_r7.jsonl,
  sha e70ce991), captures decoder hidden streams of teacher and
  byt5-small student on the same teacher-forced positions, runs the
  closed-form directional ridge-R2 scan over every (student layer,
  teacher layer) pairing, applies the PRE-REGISTERED kill rule
  (best R2 >= 0.30 AND >= 2x its scrambled control), and when armed
  solves the low-rank bridges (ridge W, SVD-truncated, bias refit)
  for the best teacher layer per student layer. Writes
  bridges.npz + scan.json to the bucket. Kill = the negative result;
  no distillation is armed.

Stage distill (WO20_STAGE=distill, a100-large; requires bridges.npz):
  the 2.0 rung verbatim (byt5-small, Muon lr 0.01 + AdamW 1e-4 group,
  sequence KD alpha 0.5 T 2.0, BS 8, 3 epochs, labels <= 384 B/side,
  cosine warmup total//20, clip 1.0, step ckpts every 500 + resume)
  plus the single pre-registered delta: beta * MSE through the frozen
  bridges at the scan-selected pairs. Epilogue (never fails the run):
  full 1,200-paragraph SadeedDiac-25 windowed zero-skip decode for
  BOTH student and in-job teacher, per-paragraph DER counts, paired
  bootstrap vs r7 (documents = resampling unit), final_eval.json.

Usage:
    hf jobs uv run --flavor t4-small -d --timeout 4h \\
      -e WO20_STAGE=scan -e WO20_RUN=wo20-tiny-scan \\
      --secrets HF_TOKEN \\
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \\
      -v /tmp/isx-wo20-code:/code:ro \\
      train_wo20_tiny_hf.py

    hf jobs uv run --flavor a100-large -d --timeout 4h \\
      -e WO20_STAGE=distill -e WO20_RUN=wo20-tiny-g2w \\
      --secrets HF_TOKEN \\
      -v hf://datasets/Interscript/arabic-r8:/train_data:ro \\
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \\
      -v /tmp/isx-wo20-code:/code:ro \\
      train_wo20_tiny_hf.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/code")

RUN = os.environ.get("WO20_RUN", "wo20-tiny")
STAGE = os.environ.get("WO20_STAGE", "scan")
SCAN_RUN = os.environ.get("WO20_SCAN_RUN", "wo20-tiny-scan")
SEED = int(os.environ.get("WO20_SEED", "0"))  # 0 = unseeded (the seed-1 run)
CKPT = Path("/ckpt") / RUN
SCAN_CKPT = Path("/ckpt") / SCAN_RUN
WORK = Path("/tmp/wo20")

RELEASE = ("https://github.com/interscript/interscript-models/"
           "releases/download/teachers-arabic-v1")
TEACHER_PARTS = ["run-007-news.zip.part-00", "run-007-news.zip.part-01"]
LABELS_FILE = "teacher_labels_r7.jsonl"
LABELS_SHA = None  # pinned from teachers-sha256.txt at fetch time
STUDENT_INIT = "google/byt5-small"
TEACHER_TOK = "google/byt5-base"

PAIRS_MAX_BYTES = 1450    # the ara spec's max_len: units <= 1450 B
LABEL_CAP_BYTES = 2900    # the lineage's label cap = 2 x max_len
BS = 8
EPOCHS = int(os.environ.get("WO20_EPOCHS", "3"))
ALPHA = float(os.environ.get("WO20_ALPHA", "0.5"))
TEMP = float(os.environ.get("WO20_TEMP", "2.0"))
BETA = float(os.environ.get("WO20_BETA", "0.1"))
BRIDGE_RANK = 256
REL_LAMBDA = 1e-4
CAL_POSITIONS = 30000     # per-side cap on captured decoder positions
KILL_BAR = 0.30           # pre-registered: TODO.sota-2026/06
KILL_MARGIN = 2.0
WINDOW = 1400             # the published windowed protocol budget
SAVE_EVERY = 500

DIACRITICS_RE = __import__("re").compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")


def log(*a):
    print(f"[wo20:{STAGE}]", *a, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"GET {url}")
    with urllib.request.urlopen(url) as r, dest.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


def fetch_teacher_and_labels() -> tuple[Path, Path]:
    """Download + verify the r7 teacher zip (parts) and the labels.
    Digests pinned from the release's own teachers-sha256.txt."""
    manifest = {}
    for line in fetch(f"{RELEASE}/teachers-sha256.txt",
                      WORK / "teachers-sha256.txt").read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            manifest[name.strip()] = digest
    for part in TEACHER_PARTS:
        expect = manifest[part]
        got = sha256_file(fetch(f"{RELEASE}/{part}", WORK / part))
        if got != expect:
            raise SystemExit(f"sha mismatch: {part} {got} != {expect}")
        log(f"verified {part}")
    zpath = WORK / "run-007-news.zip"
    if not zpath.exists():
        with zpath.open("wb") as out:
            for part in TEACHER_PARTS:
                out.write((WORK / part).read_bytes())
    tdir = WORK / "run-007-news" / "best"
    if not tdir.exists():
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(WORK)
    global LABELS_SHA
    LABELS_SHA = manifest[LABELS_FILE]
    labels = fetch(f"{RELEASE}/{LABELS_FILE}", WORK / LABELS_FILE)
    got = sha256_file(labels)
    if got != LABELS_SHA:
        raise SystemExit(f"sha mismatch: {LABELS_FILE} {got} != {LABELS_SHA}")
    log(f"verified {LABELS_FILE} sha {LABELS_SHA[:12]}")
    return tdir, labels


class Gram:
    """Chunk-accumulated rectangular Gram sufficient statistics (the
    microkimi scan's mergeable-by-addition statistics)."""

    def __init__(self, dx: int, dy: int):
        self.dx, self.dy = dx, dy
        self.gxx = np.zeros((dx, dx), np.float64)
        self.gxd = np.zeros((dx, dy), np.float64)
        self.gdd = 0.0
        self.n = 0

    def add(self, X: np.ndarray, D: np.ndarray):
        X = np.ascontiguousarray(X, np.float64)
        D = np.ascontiguousarray(D, np.float64)
        self.gxx += X.T @ X
        self.gxd += X.T @ D
        self.gdd += float((D * D).sum())
        self.n += len(X)

    def solve(self) -> tuple[np.ndarray, float]:
        lam = REL_LAMBDA * (np.trace(self.gxx) / max(self.dx, 1))
        W = np.linalg.solve(self.gxx + lam * np.eye(self.dx), self.gxd)
        res = (self.gdd - 2.0 * float((W * self.gxd).sum())
               + float(np.einsum("ij,ik,kj->", W, self.gxx, W)))
        return W, 1.0 - res / max(self.gdd, 1e-30)


def load_pairs(labels_path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = (row.get("src") or "").strip()
        t = (row.get("tgt") or row.get("teacher") or "").strip()
        if s and t and len(s.encode()) <= PAIRS_MAX_BYTES \
                and len(t.encode()) <= LABEL_CAP_BYTES:
            rows.append((s, t))
    if not rows:
        raise SystemExit("no usable label pairs — check the labels file "
                         "schema (src + tgt|teacher)")
    log(f"usable label pairs: {len(rows)}")
    return rows


def decoder_blocks(model):
    """transformers 4.x keeps the stack at model.decoder, 5.x at
    model.model.decoder — read both."""
    inner = getattr(model, "model", None)
    stack = getattr(inner, "decoder", None) if inner else None
    if stack is None:
        stack = model.decoder
    return stack.block


def capture_streams(teacher, student, tok, pairs):
    """Teacher-forced decoder hidden streams at shared byte positions."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    class P(Dataset):
        def __len__(self):
            return len(pairs)

        def __getitem__(self, i):
            return pairs[i]

    def collate(batch):
        src = tok([s for s, _ in batch], padding=True, return_tensors="pt")
        labels = tok([t for _, t in batch], padding=True,
                     return_tensors="pt").input_ids
        labels[labels == tok.pad_token_id] = -100
        return src.input_ids, src.attention_mask, labels

    nl_s = len(decoder_blocks(student))
    nl_t = len(decoder_blocks(teacher))
    ds, dt = student.config.d_model, teacher.config.d_model
    streams_s = [np.zeros((0, ds), np.float32) for _ in range(nl_s + 1)]
    streams_t = [np.zeros((0, dt), np.float32) for _ in range(nl_t + 1)]
    loader = DataLoader(P(), batch_size=BS, shuffle=False,
                        collate_fn=collate, num_workers=2)
    with torch.no_grad():
        for ids, am, labels in loader:
            ids, am, labels = ids.cuda(), am.cuda(), labels.cuda()
            mask = labels != -100
            if not bool(mask.any()):
                continue
            tout = teacher(input_ids=ids, attention_mask=am, labels=labels,
                           output_hidden_states=True)
            sout = student(input_ids=ids, attention_mask=am, labels=labels,
                           output_hidden_states=True)
            for k in range(nl_s + 1):
                take = sout.decoder_hidden_states[k].float()[mask]
                if len(take):
                    streams_s[k] = np.concatenate(
                        [streams_s[k], take.cpu().numpy()])
            for l in range(nl_t + 1):
                take = tout.decoder_hidden_states[l].float()[mask]
                if len(take):
                    streams_t[l] = np.concatenate(
                        [streams_t[l], take.cpu().numpy()])
            if len(streams_s[0]) >= CAL_POSITIONS:
                break
    streams_s = [s[:CAL_POSITIONS] for s in streams_s]
    streams_t = [s[:CAL_POSITIONS] for s in streams_t]
    if len(streams_s[0]) < 1000:
        raise SystemExit(f"capture too small ({len(streams_s[0])} positions) "
                         "— refusing to solve a degenerate Gram")
    log(f"captured {len(streams_s[0])} positions: student {nl_s + 1} "
        f"streams ({ds}d), teacher {nl_t + 1} streams ({dt}d)")
    return streams_s, streams_t


def run_scan():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    CKPT.mkdir(parents=True, exist_ok=True)
    tdir, labels_path = fetch_teacher_and_labels()
    tok = AutoTokenizer.from_pretrained(STUDENT_INIT)
    teacher = AutoModelForSeq2SeqLM.from_pretrained(
        tdir, attn_implementation="eager").cuda().eval().half()
    student = AutoModelForSeq2SeqLM.from_pretrained(STUDENT_INIT).cuda().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    for p in student.parameters():
        p.requires_grad_(False)

    pairs = load_pairs(labels_path)
    streams_s, streams_t = capture_streams(teacher, student, tok, pairs)
    del teacher
    torch.cuda.empty_cache()

    n_s, n_t = len(streams_s), len(streams_t)
    scores = np.zeros((n_s, n_t), np.float64)
    chunk = 8192
    for k in range(n_s):
        X = streams_s[k]
        for l in range(n_t):
            D = streams_t[l]
            g = Gram(X.shape[1], D.shape[1])
            for t0 in range(0, min(len(X), len(D)), chunk):
                g.add(X[t0:t0 + chunk], D[t0:t0 + chunk])
            _, scores[k, l] = g.solve()
            log(f"scan R2 student L{k} -> teacher L{l}: {scores[k, l]:.4f}")

    # scrambled control: same solve against row-permuted teacher streams
    rng = np.random.default_rng(7)
    armed, controls = [], {}
    for k in range(n_s):
        l = int(np.argmax(scores[k]))
        D = streams_t[l].copy()
        X = streams_s[k]
        g = Gram(X.shape[1], D.shape[1])
        perm = rng.permutation(len(D))
        for t0 in range(0, min(len(X), len(D)), chunk):
            g.add(X[t0:t0 + chunk], D[perm[t0:t0 + chunk]])
        _, ctrl = g.solve()
        controls[f"L{k}:L{l}"] = ctrl
        log(f"control student L{k} -> teacher L{l}: {ctrl:.4f} "
            f"(real {scores[k, l]:.4f})")
        armed.append((k, l))

    best = float(scores.max())
    best_pair = tuple(int(v) for v in np.unravel_index(scores.argmax(),
                                                       scores.shape))
    best_ctrl = controls.get(f"L{best_pair[0]}:L{best_pair[1]}", 0.0)
    verdict = ("ARMED" if best >= KILL_BAR
               and best >= KILL_MARGIN * max(best_ctrl, 1e-9) else "KILLED")
    log(f"best pair L{best_pair[0]}->L{best_pair[1]}: R2={best:.4f} "
        f"control={best_ctrl:.4f} -> {verdict}")

    bridges_path = CKPT / "bridges.npz"
    if verdict == "ARMED":
        payload, meta = {}, []
        for k, l in armed:
            X, D = streams_s[k], streams_t[l]
            g = Gram(X.shape[1], D.shape[1])
            for t0 in range(0, min(len(X), len(D)), chunk):
                g.add(X[t0:t0 + chunk], D[t0:t0 + chunk])
            W, r2 = g.solve()
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            r = min(BRIDGE_RANK, len(S))
            # W (d_s x d_t) = B @ A with B (d_s x r), A (r x d_t);
            # apply as pred = (h_s @ B) @ A + bias
            B = U[:, :r] * S[:r]
            A = Vt[:r]
            bias = D.mean(0).astype(np.float64) - X.mean(0) @ W
            payload[f"B_L{k}"] = B.astype(np.float32)
            payload[f"A_L{k}"] = A.astype(np.float32)
            payload[f"bias_L{k}"] = bias.astype(np.float32)
            meta.append({"student_layer": k, "teacher_layer": l,
                         "r2": float(r2), "rank": r})
            log(f"bridge L{k}->L{l}: r2={r2:.4f} rank={r}")
        np.savez(bridges_path, **payload)
    scan = {
        "stage": "scan",
        "run": RUN,
        "scores": scores.tolist(),
        "controls": controls,
        "armed_pairs": [list(p) for p in armed],
        "best_pair": list(best_pair),
        "best_r2": best,
        "best_control": best_ctrl,
        "kill_bar": KILL_BAR,
        "kill_margin": KILL_MARGIN,
        "verdict": verdict,
        "labels_sha256": LABELS_SHA,
        "cal_positions": int(len(streams_s[0])),
        "student_layers": n_s,
        "teacher_layers": n_t,
    }
    (CKPT / "scan.json").write_text(json.dumps(scan, indent=1))
    (CKPT / "final_eval.json").write_text(json.dumps(
        {"stage": "scan", "run": RUN, "verdict": verdict,
         "best_pair": list(best_pair), "best_r2": best,
         "best_control": best_ctrl, "labels_sha256": LABELS_SHA}, indent=1))
    log(f"scan complete: {verdict} -> {bridges_path if verdict == 'ARMED' else '(no bridges)'}")


def run_distill():
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)
    from wo20_muon import Muon, split_parameters

    CKPT.mkdir(parents=True, exist_ok=True)
    if SEED:
        # WO20u M1 multi-seed: explicit independent draw of the identical
        # recipe (seed 1 ran unseeded — entropy seed — recorded as such).
        import random

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        log(f"seed: WO20_SEED={SEED} (torch/numpy/python/data-worker)")
    bridges_file = SCAN_CKPT / "bridges.npz"
    if not bridges_file.exists():
        raise SystemExit("bridges.npz missing — run the scan stage first "
                         "(the kill rule gates this stage)")
    scan = json.loads((SCAN_CKPT / "scan.json").read_text())
    if scan.get("verdict") != "ARMED":
        raise SystemExit(f"scan verdict is {scan.get('verdict')} — "
                         "distillation is not armed")
    npz = np.load(bridges_file)
    pairs_meta = scan["armed_pairs"]

    tdir, labels_path = fetch_teacher_and_labels()
    tok = AutoTokenizer.from_pretrained(STUDENT_INIT)
    teacher = AutoModelForSeq2SeqLM.from_pretrained(
        tdir, attn_implementation="eager").cuda().eval().half()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = AutoModelForSeq2SeqLM.from_pretrained(STUDENT_INIT).cuda()
    student.gradient_checkpointing_enable()
    student.config.use_cache = False
    student.train()

    # frozen bridges: pred_t = (h_s @ B) @ A + bias
    bridges = []
    for meta in pairs_meta:
        if isinstance(meta, dict):
            k, l = meta["student_layer"], meta["teacher_layer"]
        else:
            k, l = int(meta[0]), int(meta[1])
        B = torch.from_numpy(npz[f"B_L{k}"]).cuda()
        A = torch.from_numpy(npz[f"A_L{k}"]).cuda()
        bias = torch.from_numpy(npz[f"bias_L{k}"]).cuda()
        bridges.append((k, l, B, A, bias))
    for _, _, B, A, _ in bridges:
        B.requires_grad_(False)
        A.requires_grad_(False)

    rows = load_pairs(labels_path)

    class P(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            return rows[i]

    def collate(batch):
        src = tok([s for s, _ in batch], padding=True, truncation=True,
                  max_length=PAIRS_MAX_BYTES, return_tensors="pt")
        labels = tok([t for _, t in batch], padding=True, truncation=True,
                     max_length=LABEL_CAP_BYTES,
                     return_tensors="pt").input_ids
        labels[labels == tok.pad_token_id] = -100
        return src.input_ids, src.attention_mask, labels

    loader_gen = None
    worker_init = None
    if SEED:
        def _worker_init(wid):
            wseed = SEED + wid
            random.seed(wseed)
            np.random.seed(wseed)
            torch.manual_seed(wseed)
        loader_gen = torch.Generator()
        loader_gen.manual_seed(SEED)
        worker_init = _worker_init
    loader = DataLoader(P(), batch_size=BS, shuffle=True, collate_fn=collate,
                        num_workers=2, drop_last=True, generator=loader_gen,
                        worker_init_fn=worker_init)
    total_steps = len(loader) * EPOCHS
    muon_params, adamw_params = split_parameters(
        list(student.named_parameters()))
    optimizer = Muon(muon_params, lr=0.01, momentum=0.95, weight_decay=0.01)
    optimizer.add_adamw_group(adamw_params, lr=1e-4, weight_decay=0.0)
    log(f"optimizer: muon {len(muon_params)} matrix / "
        f"adamw {len(adamw_params)} embedding-like params")
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                total_steps // 20, total_steps)

    labels_digest = LABELS_SHA[:12] if LABELS_SHA else "none"

    def _usable(ck: Path) -> bool:
        marker = ck / "labels.sha"
        return (marker.exists()
                and marker.read_text().strip() == labels_digest)

    ckpts = sorted((c for c in CKPT.glob("step-*") if _usable(c)),
                   key=lambda p: int(p.name.split("-")[1]))
    step = 0
    if ckpts:
        student.load_state_dict(torch.load(ckpts[-1] / "student.pt",
                                           map_location="cuda",
                                           weights_only=True))
        optimizer.load_state_dict(torch.load(ckpts[-1] / "optim.pt",
                                             map_location="cuda",
                                             weights_only=True))
        step = int(ckpts[-1].name.split("-")[1])
        for _ in range(step):
            scheduler.step()
        log(f"resume from {ckpts[-1].name}")

    for old in ckpts[:-2]:
        for f in old.iterdir():
            f.unlink()
        old.rmdir()

    def save(step_no):
        ck = CKPT / f"step-{step_no}"
        ck.mkdir(exist_ok=True)
        torch.save(student.state_dict(), ck / "student.pt")
        torch.save(optimizer.state_dict(), ck / "optim.pt")
        (ck / "labels.sha").write_text(labels_digest)
        keep = sorted(CKPT.glob("step-*"),
                      key=lambda p: int(p.name.split("-")[1]))
        for old in keep[:-2]:
            for f in old.iterdir():
                f.unlink()
            old.rmdir()

    log(f"train: {len(rows)} pairs x {EPOCHS} epochs = {total_steps} steps, "
        f"alpha={ALPHA} T={TEMP} beta={BETA} seed={SEED or 'unseeded'} "
        f"bridges={[(k, l) for k, l, *_ in bridges]}")

    while step < total_steps:
        for ids, am, labels in loader:
            if step >= total_steps:
                break
            ids, am, labels = ids.cuda(), am.cuda(), labels.cuda()
            with torch.no_grad():
                tout = teacher(input_ids=ids, attention_mask=am, labels=labels,
                               output_hidden_states=True)
            s_out = student(input_ids=ids, attention_mask=am, labels=labels,
                            output_hidden_states=True)
            mask = labels != -100
            kd = F.kl_div(
                F.log_softmax(s_out.logits[mask] / TEMP, dim=-1),
                F.softmax(tout.logits.float()[mask] / TEMP, dim=-1),
                reduction="batchmean") * (TEMP ** 2)
            loss = ALPHA * kd + (1 - ALPHA) * s_out.loss
            if bridges:
                mse_terms = []
                for k, l, B, A, bias in bridges:
                    h_s = s_out.decoder_hidden_states[k][mask].float()
                    h_t = tout.decoder_hidden_states[l][mask].float()
                    pred = (h_s @ B.float() @ A.float()) + bias.float()
                    mse_terms.append(F.mse_loss(pred, h_t))
                bridge_mse = torch.stack(mse_terms).mean()
                loss = loss + BETA * bridge_mse
            else:
                bridge_mse = torch.zeros((), device=ids.device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 50 == 0:
                log(f"step {step}/{total_steps} loss={float(loss):.4f} "
                    f"ce={float(s_out.loss):.4f} kd={float(kd):.4f} "
                    f"mse={float(bridge_mse):.4f}")
            if step % SAVE_EVERY == 0:
                save(step)

    out_dir = CKPT / "best"
    out_dir.mkdir(exist_ok=True)
    student.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    log(f"final student saved -> {out_dir}")
    student.gradient_checkpointing_disable()
    student.config.use_cache = True
    run_eval(student, tok, teacher)


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


def paired_bootstrap(der_a: np.ndarray, der_b: np.ndarray,
                     counts: np.ndarray, boot: int = 10000,
                     seed: int = 7):
    """Documents as resampling unit (the microkimi eval_compare math,
    count-weighted). Returns (delta = b - a, lo, hi, p_delta_gt_0)."""
    keep = counts > 0
    a = np.asarray(der_a, np.float64)[keep] * counts[keep]
    b = np.asarray(der_b, np.float64)[keep] * counts[keep]
    c = counts[keep].astype(np.float64)
    delta = b.sum() / c.sum() - a.sum() / c.sum()
    rng = np.random.default_rng(seed)
    n = len(c)
    idx = rng.integers(0, n, size=(boot, n))
    da = a[idx].sum(axis=1)
    db = b[idx].sum(axis=1)
    ct = c[idx].sum(axis=1)
    deltas = (db - da) / ct
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = float((deltas > 0).mean())
    return float(delta), float(lo), float(hi), p


def run_eval(student, tok, teacher):
    """Full 1,200-paragraph windowed zero-skip protocol for the student
    AND the in-job teacher; paired bootstrap vs r7."""
    import pyarrow.parquet as pq
    import torch
    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    teacher.eval()
    student.eval()

    table = pq.read_table(Path("/train_data") / "sadeeddiac-25.parquet")
    inputs = [DIACRITICS_RE.sub("", t) for t in table.column("input").to_pylist()]
    outputs = table.column("output").to_pylist()
    log(f"eval paragraphs: {len(inputs)}")

    windows, counts = [], []
    for text in inputs:
        ws = split_windows(text)
        counts.append(len(ws))
        windows.extend(ws)

    def decode_all(model, name, prog_name, autocast_bf16: bool):
        prog = CKPT / prog_name
        saved = {}
        if prog.exists():
            for line in prog.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    saved[row["i"]] = row["pred"]
        missing = [i for i in range(len(windows)) if i not in saved]
        model.eval()
        bs = 8
        with torch.no_grad():
            for t0 in range(0, len(missing), bs):
                chunk_ids = missing[t0:t0 + bs]
                batch = [windows[i] for i in chunk_ids]
                enc = tok(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=WINDOW).to("cuda")
                ctx = (torch.autocast("cuda", torch.bfloat16)
                       if autocast_bf16 else torch.autocast("cuda",
                                                            enabled=False))
                with ctx:
                    gen = model.generate(
                        **enc,
                        max_new_tokens=int(2 * enc.input_ids.shape[1]),
                        num_beams=1)
                preds = tok.batch_decode(gen, skip_special_tokens=True)
                for i, p in zip(chunk_ids, preds):
                    saved[i] = p
                with prog.open("a", encoding="utf-8") as out:
                    for i, p in zip(chunk_ids, preds):
                        out.write(json.dumps({"i": i, "pred": p},
                                             ensure_ascii=False) + "\n")
                if (t0 // bs) % 25 == 0:
                    log(f"[{name}] {len(saved)}/{len(windows)}")
        k = 0
        paras = []
        for c in counts:
            paras.append(" ".join(saved[i] for i in range(k, k + c)))
            k += c
        return paras

    student_paras = decode_all(student, "student",
                               "eval_progress_student.jsonl", True)
    (CKPT / "student_preds.jsonl").write_text(
        "\n".join(json.dumps({"src": i, "student": p}, ensure_ascii=False)
                  for i, p in enumerate(student_paras)))
    teacher_paras = decode_all(teacher, "teacher",
                               "eval_progress_teacher.jsonl", False)
    (CKPT / "teacher_preds.jsonl").write_text(
        "\n".join(json.dumps({"src": i, "teacher": p}, ensure_ascii=False)
                  for i, p in enumerate(teacher_paras)))

    ev = E()
    gt = list(outputs)
    n = len(gt)
    s_der = np.zeros(n)
    t_der = np.zeros(n)
    letters = np.zeros(n)
    s_wer = np.zeros(n)
    t_wer = np.zeros(n)
    words = np.zeros(n)
    for i in range(n):
        wc, lc, _, sw, _, sd, _ = ev.caculate_error_on_single_sentence(
            student_paras[i], gt[i], gt_missing_diacritic_is_error=False)
        _, _, _, tw, _, td, _ = ev.caculate_error_on_single_sentence(
            teacher_paras[i], gt[i], gt_missing_diacritic_is_error=False)
        words[i], letters[i] = wc, max(lc, 1)
        s_wer[i], t_wer[i] = sw, tw
        s_der[i] = 100.0 * sd / max(lc, 1)
        t_der[i] = 100.0 * td / max(lc, 1)

    s_total = float(100.0 * (s_der * letters).sum() / letters.sum())
    t_total = float(100.0 * (t_der * letters).sum() / letters.sum())
    delta, lo, hi, p_worse = paired_bootstrap(t_der, s_der, letters)
    t_wer_total = float(100.0 * (t_wer * words).sum() / max(words.sum(), 1))
    s_wer_total = float(100.0 * (s_wer * words).sum() / max(words.sum(), 1))
    gate = t_total + 0.5
    verdict = ("GATE-FAIL" if s_total > gate else "GATE-PASS")
    result = {
        "stage": "distill",
        "run": RUN,
        "recipe": {"alpha": ALPHA, "temperature": TEMP, "beta": BETA,
                   "epochs": EPOCHS, "bs": BS,
                   "seed": SEED or "unseeded",
                   "pairs_max_bytes": PAIRS_MAX_BYTES,
                   "student_init": STUDENT_INIT,
                   "labels_sha256": LABELS_SHA},
        "teacher_full_der_ce": t_total,
        "student_full_der_ce": s_total,
        "student_full_wer": s_wer_total,
        "teacher_full_wer": t_wer_total,
        "gate_teacher_plus_halfpp": gate,
        "paired_delta_pp_student_minus_teacher": delta,
        "paired_ci95": [lo, hi],
        "p_student_worse": p_worse,
        "verdict": verdict,
        "n_paragraphs": n,
    }
    log("RESULT " + json.dumps(result))
    (CKPT / "final_eval.json").write_text(json.dumps(result, indent=1))


def main():
    if STAGE == "scan":
        run_scan()
    elif STAGE == "distill":
        run_distill()
    else:
        raise SystemExit(f"unknown WO20_STAGE: {STAGE}")


if __name__ == "__main__":
    main()

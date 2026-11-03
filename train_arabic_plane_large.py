"""Arabic run-017 — plane-factorized diacritization (Stoicheia WO).

TODO.sota-2026/04, entered 2026-10-03 (owner: "make it work -
ultimately the best SOTA"). The task is a LENGTH-PRESERVING PLANE
PROJECTION: letters plane given (skeleton), haraqat plane predicted
per position with BIDIRECTIONAL context - the exact component the
left-to-right seq2seq rungs cannot see (iʿrāb depends on sentence
structure ahead; 33% of the residual is word-final case endings).

Model: ByT5-small ENCODER (pretrained; byte+3 table reused) + a
diacritic-plane embedding (n_classes + MASK) + a per-position
classification head over the corpus haraqat-combo inventory.
Mask-Predict self-conditioning: trained at random corruption levels
q ~ U(0,1]; inference runs K passes of fully parallel prediction
(pass k>1 conditions on pass k-1 argmax).

Corpus: benchmark-convention RUNNING TEXT only (r5-units + news mix)
- the r9 convention-drift lesson applied by design. NO paradigm tables.

Gates (pre-registered, windowed zero-skip SadeedDiac-25 full 1,200):
- rung gate: beat the student tier 4.5701 (on-device class);
- SOTA-dedicated gate: beat the teacher tier 2.2864;
- CPU latency benchmark vs ara-diac-small-int8static-2.1 (post-hoc).

Usage:
    modal run --detach train_arabic_plane.py
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)

RUN = "rababa_arabic_plane/run-019-plane-large"
UNIT_BYTES = 1400
N_VAL = 2_000
EPOCHS = 1
# teacher-tier: ByT5-LARGE encoder (~1.3B) - SOTA-dedicated gate 2.2864
MAX_LINES = 600_000
NEWS_UPSAMPLE = 3
GOLD_UPSAMPLE = 4
K_PASSES = 2

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate>=1.1.0",
        "pandas",
        "tqdm",
        "pyarrow",
        "pyarabic",
        "prettytable",
    )
    .add_local_file("sadeed_evaluator.py", "/opt/rababa/sadeed_evaluator.py", copy=True)
    .add_local_file("haraqat_planes.py", "/opt/rababa/haraqat_planes.py", copy=True)
    .add_local_dir("data/sadeed-diac-25", "/opt/rababa/data/sadeed-diac-25", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("rababa-arabic-plane-large", image=image)


def make_unit(text: str) -> str | None:
    t = text.strip()
    if not t or DIACRITICS_RE.sub("", t).encode("utf-8") == b"":
        return None
    if len(t.encode("utf-8")) > 1450:
        return None
    return t


def split_windows(text: str, budget: int = UNIT_BYTES) -> list[str]:
    if len(text.encode("utf-8")) <= budget:
        return [text]
    words = text.split()
    wins, cur, n = [], [], 0
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


@app.function(
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def train() -> dict:
    import hashlib
    import json

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_cosine_schedule_with_warmup

    import haraqat_planes as HP

    datasets_volume.reload()
    checkpoints_volume.reload()

    done_marker = Path("/checkpoints") / RUN / "EVAL_DONE"
    if done_marker.exists():
        return {"run": RUN, "status": "already-done"}

    # ---- corpus: benchmark-convention running text ----
    cache = Path("/datasets/r5-units")
    news_dir = Path("/datasets/arabic-news-r5")
    lines: list[str] = []
    for name in ("domain.txt", "replay.txt"):
        lines += [l for l in (cache / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    news = [l for l in (news_dir / "news.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = [l for l in (news_dir / "wikinews2014_gold.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    units = lines + news * NEWS_UPSAMPLE + gold * GOLD_UPSAMPLE
    random.Random(42).shuffle(units)
    units = [u for u in (make_unit(u) for u in units[:MAX_LINES]) if u]
    print(f"[data] units={len(units)}", flush=True)

    # ---- inventory (canonical combos over the corpus) ----
    ck_dir = Path("/checkpoints") / RUN
    ck_dir.mkdir(parents=True, exist_ok=True)
    inv_path = ck_dir / "inventory.json"
    if inv_path.exists():
        classes = json.loads(inv_path.read_text(encoding="utf-8"))
    else:
        classes, counts = HP.plane_classes(units)
        inv_path.write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")
        print(f"[inventory] {len(classes)} classes (incl. none); top: "
              f"{[(c, counts[c]) for c in classes[1:9]]}", flush=True)
    class_to_id = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    MASK_ID = n_classes  # plane vocab = n_classes + 1 (mask)
    labels_digest = hashlib.sha256(inv_path.read_bytes()).hexdigest()[:12]
    checkpoints_volume.commit()

    tok = AutoTokenizer.from_pretrained("google/byt5-large")

    def encode_unit(text: str):
        skel, labels = HP.split_planes(text)
        ids = tok(skel).input_ids[:-1]  # drop EOS: one token per byte
        # char -> byte-span map (Arabic chars are multi-byte)
        label_ids = [0] * len(ids)
        pos = 0
        for ch, lab in zip(skel, labels):
            n = len(ch.encode("utf-8"))
            lab = lab.split("\x00", 1)[-1]
            cid = class_to_id.get(HP.canon_combo(lab), 0)
            for t in range(pos, min(pos + n, len(label_ids))):
                label_ids[t] = cid
            pos += n
        return ids, label_ids

    class PlaneDS(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return encode_unit(self.rows[i])

    def collate(batch):
        mx = max(len(x[0]) for x in batch)
        ids = torch.zeros((len(batch), mx), dtype=torch.long)
        lab = torch.zeros((len(batch), mx), dtype=torch.long)
        am = torch.zeros((len(batch), mx), dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, : len(x)] = torch.tensor(x)
            lab[i, : len(y)] = torch.tensor(y)
            am[i, : len(x)] = 1
        return ids, am, lab

    val_rows = units[-N_VAL:]
    train_rows = units[:-N_VAL]
    train_loader = DataLoader(
        PlaneDS(train_rows), batch_size=8, shuffle=True, collate_fn=collate,
        num_workers=4, drop_last=True,
    )
    total_steps = len(train_loader) * EPOCHS
    print(f"[data] train={len(train_rows)} val={len(val_rows)} steps={total_steps} "
          f"classes={n_classes}", flush=True)

    backbone = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-large")
    encoder = backbone.encoder.cuda().train()  # pretrained; decoder discarded
    d_model = encoder.config.d_model

    plane_emb = nn.Embedding(n_classes + 1, d_model).cuda().train()
    nn.init.normal_(plane_emb.weight, std=0.02)
    head = nn.Linear(d_model, n_classes).cuda().train()

    params = list(encoder.parameters()) + list(plane_emb.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 500, total_steps)
    ce = nn.CrossEntropyLoss()

    def usable(ck: Path) -> bool:
        m = ck / "labels.sha"
        return m.exists() and m.read_text().strip() == labels_digest

    ckpts = sorted((c for c in ck_dir.glob("step-*") if usable(c)),
                   key=lambda p: int(p.name.split("-")[1]))
    step = 0
    if ckpts:
        sd = torch.load(ckpts[-1] / "model.pt", map_location="cpu", weights_only=True)
        encoder.load_state_dict(sd["encoder"])
        plane_emb.load_state_dict(sd["plane_emb"])
        head.load_state_dict(sd["head"])
        optimizer.load_state_dict(torch.load(ckpts[-1] / "optim.pt", map_location="cpu", weights_only=True))
        step = int(ckpts[-1].name.split("-")[1])
        for _ in range(step):
            scheduler.step()
        print(f"[resume] step-{step}", flush=True)

    def forward_logits(ids, am, plane_in):
        embeds = encoder.embed_tokens(ids) + plane_emb(plane_in)
        out = encoder(inputs_embeds=embeds, attention_mask=am)
        return head(out.last_hidden_state)

    for _epoch in range(EPOCHS):
      for ids, am, lab in train_loader:
        if step >= total_steps:
            break
        ids, am, lab = ids.cuda(), am.cuda(), lab.cuda()
        q = random.random()  # corruption level; q near 1 = first pass
        keep = (torch.rand_like(lab, dtype=torch.float) >= q).long() * am
        plane_in = torch.where(keep == 1, lab, torch.full_like(lab, MASK_ID))
        with torch.autocast("cuda", torch.bfloat16):
            logits = forward_logits(ids, am, plane_in)
        loss = ce(logits.float().reshape(-1, n_classes), lab.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1
        if step % 50 == 0:
            print(f"[{RUN} step {step}/{total_steps}] ce={float(loss):.4f} q={q:.2f}", flush=True)
        if step % 500 == 0:
            ck = ck_dir / f"step-{step}"
            ck.mkdir(exist_ok=True)
            (ck / "labels.sha").write_text(labels_digest)
            torch.save({"encoder": encoder.state_dict(),
                        "plane_emb": plane_emb.state_dict(),
                        "head": head.state_dict()}, ck / "model.pt")
            torch.save(optimizer.state_dict(), ck / "optim.pt")
            checkpoints_volume.commit()
            print(f"[volume] committed at step {step}", flush=True)

    best = ck_dir / "best"
    best.mkdir(exist_ok=True)
    torch.save({"encoder": encoder.state_dict(),
                "plane_emb": plane_emb.state_dict(),
                "head": head.state_dict(),
                "classes": classes}, best / "model.pt")
    checkpoints_volume.commit()

    # ---- ID gate: windowed zero-skip SadeedDiac, K-pass ----
    import pandas as pd
    import pyarrow.parquet as pq

    encoder.eval()
    plane_emb.eval()
    head.eval()
    table = pq.read_table("data/sadeed-diac-25/train.parquet")
    inputs = [DIACRITICS_RE.sub("", t) for t in table.column("input").to_pylist()]
    outputs = table.column("output").to_pylist()

    all_windows: list[str] = []
    counts: list[int] = []
    for text in inputs:
        ws = split_windows(text)
        counts.append(len(ws))
        all_windows.extend(ws)
    print(f"[eval] {len(inputs)} paragraphs -> {len(all_windows)} windows", flush=True)

    prog = ck_dir / "eval_progress.jsonl"
    saved: dict[int, str] = {}
    if prog.exists():
        for line in prog.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                saved[row["i"]] = row["pred"]
        print(f"[gen] resuming with {len(saved)} saved windows", flush=True)

    @torch.no_grad()
    def predict_window(skel: str) -> str:
        ids, _ = encode_unit(skel)
        t_ids = torch.tensor([ids], device="cuda")
        am = torch.ones_like(t_ids)
        plane = torch.full_like(t_ids, MASK_ID)
        preds = None
        for _ in range(K_PASSES):
            with torch.autocast("cuda", torch.bfloat16):
                logits = forward_logits(t_ids, am, plane)
            preds = logits.argmax(-1)
            plane = preds
        # per-char class = majority vote of the char's byte-token
        # predictions (multi-byte chars span >1 token)
        pos = 0
        char_preds: list[str] = []
        for ch in skel:
            n = len(ch.encode("utf-8"))
            votes = preds[0, pos : pos + n].tolist()
            cid = max(set(votes), key=votes.count)
            char_preds.append(classes[cid] if cid < n_classes else "")
            pos += n
        return HP.render(skel, char_preds)

    missing = [i for i in range(len(all_windows)) if i not in saved]
    n_new = 0
    with prog.open("a", encoding="utf-8") as prog_out:
        for bi in range(0, len(missing), 8):
            idxs = missing[bi : bi + 8]
            for i in idxs:
                pred = predict_window(all_windows[i])
                prog_out.write(json.dumps({"i": i, "pred": pred}, ensure_ascii=False) + "\n")
                saved[i] = pred
            n_new += len(idxs)
            if n_new % 160 == 0:
                prog_out.flush()
                checkpoints_volume.commit()
                print(f"[gen] {len(saved)}/{len(all_windows)} (committed)", flush=True)
    checkpoints_volume.commit()

    k = 0
    paragraphs = []
    for _text, c in zip(inputs, counts):
        paragraphs.append(" ".join(saved[i] for i in range(k, k + c)))
        k += c

    csv_path = Path("/tmp/sadeed_plane_windowed.csv")
    pd.DataFrame({"gt": outputs, "pred": paragraphs}).to_csv(csv_path, index=False, header=False)
    (ck_dir / "sadeed_preds_windowed.csv").write_text(csv_path.read_text(), encoding="utf-8")
    checkpoints_volume.commit()

    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    print("\n===== run-017 plane, windowed zero-skip (ID gate; "
          "rung gate 4.5701 / SOTA gate 2.2864) =====", flush=True)
    E.report_errors_on_csv_file(
        str(csv_path), ground_truth_column_index=0, predicted_column_index=1, has_header=False,
        gt_missing_diacritic_is_error=False)

    done_marker.touch()
    checkpoints_volume.commit()
    return {"run": RUN, "steps": step, "k_passes": K_PASSES}


@app.local_entrypoint()
def main():
    handle = train.spawn()
    print(f"spawned {handle.object_id}; completion = EVAL_DONE at "
          f"rababa_checkpoints:{RUN}/EVAL_DONE", flush=True)

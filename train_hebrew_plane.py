"""Hebrew run-021 — nikud plane-factorized diacritization (TODO.final 12).

The Arabic plane recipe (run-018 v2: ByT5-small encoder + plane
embedding + per-position head, K=2 mask-predict, 2 epochs) applied to
Hebrew over the v4 combined corpus (nakdimon + sefaria + distilled
v1/v2 + expanded_v2; /datasets/hebrew-v4). The plane module preserves
corpus cluster order as written (nikud_planes.py).

Gate (pre-registered, nakdimon test, greedy, seq2seq_der - the exact
heb-diac-1.1 protocol): match or beat the shipped seq2seq student's
16.44 DER at the plane's on-device economics.

Usage:
    modal run --detach train_hebrew_plane.py
"""

from __future__ import annotations

import random
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)

RUN = "rababa_hebrew_plane/run-021-heb-plane"
UNIT_BYTES = 1400
N_VAL = 2_000
EPOCHS = 2
MAX_LINES = 600_000
K_PASSES = 2

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate>=1.1.0",
        "pandas",
        "tqdm",
    )
    .add_local_file("nikud_planes.py", "/opt/rababa/nikud_planes.py", copy=True)
    .add_local_dir("src/rababa", "/opt/rababa/rababa", copy=True)
    .workdir("/opt/rababa")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("rababa-heb-plane", image=image)


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
    from collections import Counter

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_cosine_schedule_with_warmup

    import nikud_planes as NP
    from rababa.evaluate import seq2seq_der

    datasets_volume.reload()
    checkpoints_volume.reload()

    done_marker = Path("/checkpoints") / RUN / "EVAL_DONE"
    prog_name = f"eval_progress_{done_marker.name}.jsonl"  # per-marker: no stale reuse

    # ---- corpus: the v4 combined labeled set ----
    import pandas as pd

    train_rows = pd.read_json("/datasets/hebrew-v4/train.jsonl", lines=True)
    units = []
    for t in train_rows["tgt"].tolist():
        t = t.strip()
        if not t or len(t.encode("utf-8")) > 1450:
            continue
        units.append(t)
    random.Random(42).shuffle(units)
    units = units[:MAX_LINES]
    print(f"[data] units={len(units)}", flush=True)

    ck_dir = Path("/checkpoints") / RUN
    ck_dir.mkdir(parents=True, exist_ok=True)
    inv_path = ck_dir / "inventory.json"
    if inv_path.exists():
        classes = json.loads(inv_path.read_text(encoding="utf-8"))
    else:
        counts: Counter = Counter()
        for u in units:
            _, cls = NP.split_planes(u)
            counts.update(c for c in cls if c)
        classes = [""] + sorted(counts)
        inv_path.write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")
        print(f"[inventory] {len(classes)} classes (incl. none); top: "
              f"{counts.most_common(8)}", flush=True)
    class_to_id = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    MASK_ID = n_classes
    labels_digest = hashlib.sha256(inv_path.read_bytes()).hexdigest()[:12]
    checkpoints_volume.commit()

    tok = AutoTokenizer.from_pretrained("google/byt5-small")

    def encode_unit(text: str):
        skel, labels = NP.split_planes(text)
        ids = tok(skel).input_ids[:-1]
        label_ids = [0] * len(ids)
        pos = 0
        for ch, lab in zip(skel, labels):
            n = len(ch.encode("utf-8"))
            cid = class_to_id.get(lab, 0)
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
    train_units = units[:-N_VAL]
    train_loader = DataLoader(
        PlaneDS(train_units), batch_size=16, shuffle=True, collate_fn=collate,
        num_workers=4, drop_last=True,
    )
    total_steps = len(train_loader) * EPOCHS
    print(f"[data] train={len(train_units)} val={len(val_rows)} steps={total_steps} "
          f"classes={n_classes}", flush=True)

    backbone = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")
    encoder = backbone.encoder.cuda().train()
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
        q = random.random()
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

    # ---- gate: nakdimon test, K-pass, seq2seq_der (the 16.44 protocol) ----
    encoder.eval()
    plane_emb.eval()
    head.eval()

    test_rows = pd.read_json("/datasets/hebrew-v4/test.jsonl", lines=True)
    targets = test_rows["tgt"].tolist()

    all_windows: list[str] = []
    counts_w: list[int] = []
    for text in targets:
        ws = split_windows(text.strip())
        counts_w.append(len(ws))
        all_windows.extend(ws)
    print(f"[eval] {len(targets)} examples -> {len(all_windows)} windows", flush=True)

    prog = ck_dir / prog_name
    saved: dict[int, str] = {}
    if prog.exists():
        for line in prog.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                saved[row["i"]] = row["pred"]
        print(f"[gen] resuming with {len(saved)} saved windows", flush=True)

    @torch.no_grad()
    def predict_window(text: str) -> str:
        skel, _ = NP.split_planes(text)
        ids = tok(skel).input_ids[:-1]
        t_ids = torch.tensor([ids], device="cuda")
        am = torch.ones_like(t_ids)
        plane = torch.full_like(t_ids, MASK_ID)
        preds = None
        for _ in range(K_PASSES):
            with torch.autocast("cuda", torch.bfloat16):
                logits = forward_logits(t_ids, am, plane)
            preds = logits.argmax(-1)
            plane = preds
        pos = 0
        char_preds: list[str] = []
        for ch in skel:
            n = len(ch.encode("utf-8"))
            votes = preds[0, pos : pos + n].tolist()
            cid = max(set(votes), key=votes.count)
            char_preds.append(classes[cid] if cid < n_classes else "")
            pos += n
        return NP.render(skel, char_preds)

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

    # stitch with ORIGINAL separators: " ".join collapses double
    # spaces/newlines, shifting every unit after the first anomaly
    # (run-021 first verdict: 32.64% from this bug; true DER 12.48%)
    import re as _re

    k = 0
    total_wrong = 0.0
    total_positions = 0
    preds_rows = []
    for tgt, c in zip(targets, counts_w):
        text = tgt.strip()
        words = text.split()
        seps = _re.findall(r"\s+", text)
        sep_for = {i: (seps[i] if i < len(seps) else "") for i in range(len(words))}
        rebuilt: list[str] = []
        for _g in range(c):
            pred = saved[k]; k += 1
            for w in pred.split():
                wi = sum(len(part.split()) for part in rebuilt)
                rebuilt.append(w + sep_for.get(wi, " "))
        pred = "".join(rebuilt)
        total_wrong += der * n
        total_positions += n
        preds_rows.append({"gt": tgt, "pred": pred, "der": der, "n": n})

    der = total_wrong / max(1, total_positions)
    pd.DataFrame(preds_rows).to_json(ck_dir / "eval_rows.jsonl", orient="records", lines=True, force_ascii=False)
    print(f"\n===== {RUN} nakdimon greedy DER: {der:.4f} "
          f"({total_positions} positions) — gate: heb-diac-1.1 16.44 =====", flush=True)
    checkpoints_volume.commit()

    done_marker.touch()
    checkpoints_volume.commit()
    return {"run": RUN, "steps": step, "k_passes": K_PASSES, "der": der}


@app.local_entrypoint()
def main():
    handle = train.spawn()
    print(f"spawned {handle.object_id}; completion = EVAL_DONE at "
          f"rababa_checkpoints:{RUN}/EVAL_DONE", flush=True)

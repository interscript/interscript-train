# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.5.1",
#   "transformers==4.46.3",
#   "accelerate>=1.1.0",
# ]
# ///
"""run-094 — Hebrew stage-2 gold SFT on HF Jobs (WO92, owner go 2026-11-19).

Curriculum stage 2: the EXACT crown gold recipe (heb-vocalized-plane-2.0
lineage: byt5-base, 4 epochs gold-only) resumed from the run-093
weak-pretrain best checkpoint. Gold-only corpus = hebrew-v4 train.jsonl
(50,433 rows; 50,303 after the 1450-byte filter). Same inventory as
stage 1 via the labels.sha contract: classes are READ from
run-093-heb-weak-pretrain/inventory.json (never rebuilt) — the vocab
path consistency law.

Gate (pre-registered in WO92): crown protocol, single-row greedy K=3
mask-predict, 1400-byte windows, original-separator stitching,
seq2seq_der on the 1,864-row human-verified test set —
**DER < 8.18 clears the crown gate** (heb-vocalized-plane-2.0); a
result >= 8.18 is a recorded negative and the crown stays shipped.

Engineering laws carried (unchanged from stage 1): ByT5's internal
T5-norm as-pretrained (bf16 autocast + fp32 CE proven at
run-021/022/026/093); ids from AutoTokenizer only (input_ids[:-1]
strips trailing EOS); beam never used on flat byte students.

Usage:
    hf jobs uv run --flavor a100-large -d --timeout 6h -s HF_TOKEN \
      -v hf://datasets/Interscript/hebrew-v4:/train_data:ro \
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \
      -v /tmp/heb-weak-code:/code:ro \
      --name run-094-heb-gold-sft train_hebrew_sft_hf.py
    # arms via env: HEB_RUN / HEB_EPOCHS / HEB_BS / HEB_ACCUM /
    # HEB_INIT (default run-093-heb-weak-pretrain) / HEB_CKPT_EVERY
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/code")
sys.path.insert(0, "/code/src")

DATA = Path("/train_data")        # hebrew-v4 (gold train + human-verified test)
CKPT = Path("/ckpt")
RUN = os.environ.get("HEB_RUN", "run-094-heb-gold-sft")
INIT_RUN = os.environ.get("HEB_INIT", "run-093-heb-weak-pretrain")
BACKBONE = os.environ.get("HEB_BACKBONE", "google/byt5-base")
EPOCHS = int(os.environ.get("HEB_EPOCHS", "4"))  # crown gold recipe verbatim
K_PASSES = int(os.environ.get("HEB_K", "3"))
MICRO_BS = int(os.environ.get("HEB_BS", "8"))
GRAD_ACCUM = int(os.environ.get("HEB_ACCUM", "2"))
CKPT_EVERY = int(os.environ.get("HEB_CKPT_EVERY", "1000"))
UNIT_BYTES = 1400
N_VAL = 2_000

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_cosine_schedule_with_warmup

import nikud_planes as NP
from rababa.evaluate import seq2seq_der


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


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    run_dir = CKPT / RUN
    run_dir.mkdir(parents=True, exist_ok=True)
    done_marker = run_dir / "EVAL_DONE"
    prog_name = f"eval_progress_{done_marker.name}.jsonl"

    # ---- inventory: READ from the stage-1 run, never rebuilt ----
    inv_src = CKPT / INIT_RUN / "inventory.json"
    classes = json.loads(inv_src.read_text(encoding="utf-8"))
    inv_path = run_dir / "inventory.json"
    inv_path.write_bytes(inv_src.read_bytes())  # byte-identical copy in the new run dir
    labels_digest = hashlib.sha256(inv_src.read_bytes()).hexdigest()[:12]
    class_to_id = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    MASK_ID = n_classes
    print(f"[inventory] inherited {n_classes} classes from {INIT_RUN} "
          f"(labels.sha {labels_digest})", flush=True)

    # ---- gold-only corpus (the crown's corpus, unmodified) ----
    units = []
    for row in load_jsonl(DATA / "train.jsonl"):
        t = row["tgt"].strip()
        if not t or len(t.encode("utf-8")) > 1450:
            continue
        units.append(t)
    random.Random(42).shuffle(units)
    print(f"[data] gold units={len(units)}", flush=True)

    # ---- init from the stage-1 best checkpoint ----
    init_ckpt = CKPT / INIT_RUN / "best" / "model.pt"
    sd = torch.load(init_ckpt, map_location="cpu", weights_only=True)
    saved_classes = sd.get("classes")
    if saved_classes is not None and saved_classes != classes:
        raise RuntimeError("stage-1 best checkpoint classes != inventory — "
                           "vocab path consistency violated; refusing to init")
    print(f"[init] {init_ckpt} (weak={sd.get('corpus', {}).get('weak')} "
          f"gold={sd.get('corpus', {}).get('gold')}, steps "
          f"{sd.get('steps', 'n/a')})", flush=True)

    tok = AutoTokenizer.from_pretrained(BACKBONE)

    def encode_unit(text: str):
        skel, labels = NP.split_planes(text)
        ids = tok(skel).input_ids[:-1]  # ByT5 byte table via tokenizer; drop EOS
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
        PlaneDS(train_units), batch_size=MICRO_BS, shuffle=True, collate_fn=collate,
        num_workers=2, drop_last=True,
    )
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    print(f"[data] train={len(train_units)} val={len(val_rows)} steps={total_steps} "
          f"classes={n_classes}", flush=True)

    backbone = AutoModelForSeq2SeqLM.from_pretrained(BACKBONE)
    encoder = backbone.encoder.cuda().train()
    encoder.load_state_dict(sd["encoder"])
    d_model = encoder.config.d_model

    plane_emb = nn.Embedding(n_classes + 1, d_model).cuda().train()
    plane_emb.load_state_dict(sd["plane_emb"])
    head = nn.Linear(d_model, n_classes).cuda().train()
    head.load_state_dict(sd["head"])

    # crown gold recipe verbatim
    params = list(encoder.parameters()) + list(plane_emb.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 500, total_steps)
    ce = nn.CrossEntropyLoss()

    def usable(ck: Path) -> bool:
        m = ck / "labels.sha"
        return m.exists() and m.read_text().strip() == labels_digest

    ckpts = sorted((c for c in run_dir.glob("step-*") if usable(c)),
                   key=lambda p: int(p.name.split("-")[1]))
    step = 0
    if ckpts:
        rsd = torch.load(ckpts[-1] / "model.pt", map_location="cpu", weights_only=True)
        encoder.load_state_dict(rsd["encoder"])
        plane_emb.load_state_dict(rsd["plane_emb"])
        head.load_state_dict(rsd["head"])
        optimizer.load_state_dict(torch.load(ckpts[-1] / "optim.pt", map_location="cpu", weights_only=True))
        step = int(ckpts[-1].name.split("-")[1])
        for _ in range(step):
            scheduler.step()
        print(f"[resume] step-{step}", flush=True)

    def forward_logits(ids, am, plane_in):
        embeds = encoder.embed_tokens(ids) + plane_emb(plane_in)
        out = encoder(inputs_embeds=embeds, attention_mask=am)
        return head(out.last_hidden_state)

    def save_ckpt(s: int) -> None:
        ck = run_dir / f"step-{s}"
        ck.mkdir(exist_ok=True)
        (ck / "labels.sha").write_text(labels_digest)
        torch.save({"encoder": encoder.state_dict(),
                    "plane_emb": plane_emb.state_dict(),
                    "head": head.state_dict()}, ck / "model.pt")
        torch.save(optimizer.state_dict(), ck / "optim.pt")
        print(f"[ckpt] step-{s} saved", flush=True)

    optimizer.zero_grad()
    micro = 0
    t0 = time.time()
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
            loss = ce(logits.float().reshape(-1, n_classes), lab.reshape(-1)) / GRAD_ACCUM
            loss.backward()
            micro += 1
            if micro % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % 50 == 0:
                    rate = step / max(1e-9, time.time() - t0)
                    eta_h = (total_steps - step) / max(1e-9, rate) / 3600.0
                    print(f"[{RUN} step {step}/{total_steps}] "
                          f"ce={float(loss) * GRAD_ACCUM:.4f} q={q:.2f} "
                          f"eta={eta_h:.1f}h", flush=True)
                if step % CKPT_EVERY == 0:
                    save_ckpt(step)

    best = run_dir / "best"
    best.mkdir(exist_ok=True)
    torch.save({"encoder": encoder.state_dict(),
                "plane_emb": plane_emb.state_dict(),
                "head": head.state_dict(),
                "classes": classes,
                "init": INIT_RUN,
                "corpus": {"gold": len(units)}},
               best / "model.pt")
    print("[ckpt] best saved", flush=True)

    # ---- crown-protocol gate: greedy K=3 on the human-verified test ----
    encoder.eval()
    plane_emb.eval()
    head.eval()

    targets = [r["tgt"] for r in load_jsonl(DATA / "test.jsonl")]
    all_windows: list[str] = []
    counts_w: list[int] = []
    for text in targets:
        ws = split_windows(text.strip())
        counts_w.append(len(ws))
        all_windows.extend(ws)
    print(f"[eval] {len(targets)} examples -> {len(all_windows)} windows", flush=True)

    prog = run_dir / prog_name
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
    with prog.open("a", encoding="utf-8") as prog_out:
        for n_done, i in enumerate(missing, 1):
            pred = predict_window(all_windows[i])
            prog_out.write(json.dumps({"i": i, "pred": pred}, ensure_ascii=False) + "\n")
            saved[i] = pred
            if n_done % 200 == 0:
                prog_out.flush()
                print(f"[gen] {len(saved)}/{len(all_windows)}", flush=True)

    # stitch with ORIGINAL separators (run-021 lesson)
    k = 0
    total_wrong = 0.0
    total_positions = 0
    for tgt, c in zip(targets, counts_w):
        text = tgt.strip()
        words = text.split()
        seps = re.findall(r"\s+", text)
        sep_for = {i: (seps[i] if i < len(seps) else "") for i in range(len(words))}
        rebuilt: list[str] = []
        for _g in range(c):
            pred = saved[k]; k += 1
            for w in pred.split():
                wi = sum(len(part.split()) for part in rebuilt)
                rebuilt.append(w + sep_for.get(wi, " "))
        pred = "".join(rebuilt)
        d, n = seq2seq_der(pred, tgt)
        total_wrong += d * n
        total_positions += n

    der = total_wrong / max(1, total_positions)
    cleared = der < 8.18
    verdict = {"run": RUN, "stage": "gold-sft (WO92 stage 2, owner go)",
               "init": INIT_RUN, "corpus": {"gold": len(units)},
               "steps": step, "k_passes": K_PASSES, "der": der,
               "gate": "crown heb-vocalized-plane-2.0 DER 8.18 — "
                       + ("CLEARED" if cleared else "NOT cleared; recorded negative, crown stays shipped"),
               "gate_cleared": cleared,
               "crown_baseline": 8.18,
               "stage1_informational_der": 9.1316}
    (run_dir / "verdict.json").write_text(json.dumps(verdict, indent=1), encoding="utf-8")
    print(f"\n===== {RUN} gold-SFT crown-protocol DER: {der:.4f} "
          f"(gate < 8.18: {'CLEARED' if cleared else 'NOT CLEARED'}; "
          f"stage-1 informational was 9.1316) =====", flush=True)
    done_marker.touch()
    print("[done] EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()

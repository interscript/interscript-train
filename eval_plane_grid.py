"""run-017 checkpoint x K grid on the SadeedDiac subset.

First full-set verdict (K=4, last checkpoint): 4.7615 total DER —
0.19pp off the 4.5701 rung gate, with train CE ~0.001 (memorized).
Suspects: checkpoint selection (overfit) and inference depth (K).
This grid measures both; subset numbers are labeled and never quoted
as gates. Full-set at the winner afterwards.

Usage:
    modal run --detach eval_plane_grid.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("rababa-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)

RUN = "rababa_arabic_plane/run-017-plane"
UNIT_BYTES = 1400
SUBSET = 300
K_LIST = (2, 4, 8)
CKPTS = ("step-10000", "step-15000", "best")

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "pandas", "pyarrow", "pyarabic", "prettytable", "tqdm")
    .add_local_file("sadeed_evaluator.py", "/opt/rababa/sadeed_evaluator.py", copy=True)
    .add_local_file("haraqat_planes.py", "/opt/rababa/haraqat_planes.py", copy=True)
    .add_local_dir("data/sadeed-diac-25", "/opt/rababa/data/sadeed-diac-25", copy=True)
    .workdir("/opt/rababa")
)

app = modal.App("rababa-plane-grid", image=image)


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


@app.function(gpu="A10G", timeout=6 * 60 * 60,
              volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume})
def sweep() -> dict:
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    import torch.nn as nn
    from sadeed_evaluator import ArabicDiacritizationEvaluator as E
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    import haraqat_planes as HP

    checkpoints_volume.reload()
    ck = Path("/checkpoints") / RUN
    classes = torch.load(ck / "best" / "model.pt", map_location="cpu",
                         weights_only=True)["classes"]
    n_classes = len(classes)
    MASK_ID = n_classes
    tok = AutoTokenizer.from_pretrained("google/byt5-small")

    def load_ckpt(name):
        s = torch.load(ck / name / "model.pt", map_location="cpu", weights_only=True)
        enc = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small").encoder.cuda().eval()
        enc.load_state_dict(s["encoder"])
        pe = nn.Embedding(n_classes + 1, enc.config.d_model).cuda().eval()
        pe.load_state_dict(s["plane_emb"])
        hh = nn.Linear(enc.config.d_model, n_classes).cuda().eval()
        hh.load_state_dict(s["head"])
        return enc, pe, hh

    table = pq.read_table("data/sadeed-diac-25/train.parquet")
    inputs = [DIACRITICS_RE.sub("", t) for t in table.column("input").to_pylist()[:SUBSET]]
    outputs = table.column("output").to_pylist()[:SUBSET]

    for cname in CKPTS:
        encoder, plane_emb, head = load_ckpt(cname)

        @torch.no_grad()
        def predict(skel: str, k: int) -> str:
            ids = tok(skel).input_ids[:-1]
            t_ids = torch.tensor([ids], device="cuda")
            am = torch.ones_like(t_ids)
            plane = torch.full_like(t_ids, MASK_ID)
            preds = None
            for _ in range(k):
                embeds = encoder.embed_tokens(t_ids) + plane_emb(plane)
                out = encoder(inputs_embeds=embeds, attention_mask=am)
                preds = head(out.last_hidden_state).argmax(-1)
                plane = preds
            pos, char_preds = 0, []
            for ch in skel:
                n = len(ch.encode("utf-8"))
                votes = preds[0, pos : pos + n].tolist()
                cid = max(set(votes), key=votes.count)
                char_preds.append(classes[cid] if cid < n_classes else "")
                pos += n
            return HP.render(skel, char_preds)

        for k in K_LIST:
            paragraphs = [" ".join(predict(w, k) for w in split_windows(t)) for t in inputs]
            csv_path = Path(f"/tmp/plane_{cname}_k{k}.csv")
            pd.DataFrame({"gt": outputs, "pred": paragraphs}).to_csv(
                csv_path, index=False, header=False)
            print(f"\n===== {cname} K={k} (subset {SUBSET} — NOT a gate number) =====",
                  flush=True)
            E.report_errors_on_csv_file(
                str(csv_path), ground_truth_column_index=0, predicted_column_index=1,
                has_header=False, gt_missing_diacritic_is_error=False)
        del encoder, plane_emb, head
        torch.cuda.empty_cache()

    (ck / "grid_subset.json").write_text(json.dumps({"ckpts": CKPTS, "k": list(K_LIST)}),
                                         encoding="utf-8")
    checkpoints_volume.commit()
    return {"ckpts": CKPTS, "k": list(K_LIST)}


@app.local_entrypoint()
def main():
    print(sweep.remote())

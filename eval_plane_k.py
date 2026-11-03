"""run-017 K-pass sweep: load best/, evaluate at K in {1,2,4,8}.

The built-in gate runs K=4; this sweep finds the inference-depth /
quality optimum (subset-first, full-set at the best K only when
ASK_FULL=1). Protocol note: subset numbers are labeled as such and
never quoted as the gate.

Usage:
    modal run --detach eval_plane_k.py
    modal run --detach eval_plane_k.py --ask-full 1
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
K_LIST = (1, 2, 4, 8)

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "pandas", "pyarrow", "pyarabic", "prettytable", "tqdm")
    .add_local_file("sadeed_evaluator.py", "/opt/rababa/sadeed_evaluator.py", copy=True)
    .add_local_file("haraqat_planes.py", "/opt/rababa/haraqat_planes.py", copy=True)
    .add_local_dir("data/sadeed-diac-25", "/opt/rababa/data/sadeed-diac-25", copy=True)
    .workdir("/opt/rababa")
)

app = modal.App("rababa-plane-ksweep", image=image)


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
def sweep(ask_full: int = 0) -> dict:
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer

    import haraqat_planes as HP

    checkpoints_volume.reload()
    ck = Path("/checkpoints") / RUN
    sd = torch.load(ck / "best" / "model.pt", map_location="cpu", weights_only=True)
    classes = sd["classes"]
    n_classes = len(classes)
    MASK_ID = n_classes

    from transformers import AutoModelForSeq2SeqLM

    backbone = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")
    encoder = backbone.encoder.cuda().eval()
    encoder.load_state_dict(sd["encoder"])
    import torch.nn as nn

    plane_emb = nn.Embedding(n_classes + 1, encoder.config.d_model).cuda().eval()
    plane_emb.load_state_dict(sd["plane_emb"])
    head = nn.Linear(encoder.config.d_model, n_classes).cuda().eval()
    head.load_state_dict(sd["head"])

    tok = AutoTokenizer.from_pretrained("google/byt5-small")

    table = pq.read_table("data/sadeed-diac-25/train.parquet")
    inputs_all = [DIACRITICS_RE.sub("", t) for t in table.column("input").to_pylist()]
    outputs_all = table.column("output").to_pylist()
    inputs, outputs = inputs_all[:SUBSET], outputs_all[:SUBSET]

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

    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    results = {}
    for k in K_LIST:
        paragraphs = []
        for text in inputs:
            windows = split_windows(text)
            paragraphs.append(" ".join(predict(w, k) for w in windows))
        csv_path = Path(f"/tmp/plane_k{k}.csv")
        pd.DataFrame({"gt": outputs, "pred": paragraphs}).to_csv(csv_path, index=False, header=False)
        print(f"\n===== K={k} (subset {SUBSET} paras — NOT a gate number) =====", flush=True)
        E.report_errors_on_csv_file(
            str(csv_path), ground_truth_column_index=0, predicted_column_index=1,
            has_header=False, gt_missing_diacritic_is_error=False)

    ck.mkdir(parents=True, exist_ok=True)
    (ck / "ksweep_subset.json").write_text(json.dumps(results or {"k_list": list(K_LIST)}), encoding="utf-8")
    checkpoints_volume.commit()
    return {"k_list": list(K_LIST), "subset": SUBSET}


@app.local_entrypoint()
def main(ask_full: int = 0):
    print(sweep.remote(ask_full=ask_full))

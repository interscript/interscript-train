"""WO51 opener — find the first diverging decode step: torch vs ORT.

Loads the shipped urd-diac-base zip, runs the same greedy decode in
torch (fp32, CPU) and ORT (fp32, CPU) on rows from the diac test
split, and for the first row whose outputs diverge prints the
per-step top-token agreement and the logit deltas at the divergence
point. Localizes WHICH decoder step (and roughly which layer class)
first breaks — the input to the opset/export fix.

Usage: python probe_logit_diff.py [n_rows=5]
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ZIP = Path("/tmp/urd-diac-rt/urd-diac-base-1.0.zip")
TEST = Path("/tmp/fas-data/test.jsonl")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5

tmp = Path("/tmp/urd-diac-rt/x")
tmp.mkdir(exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    for name in ("encoder.onnx", "decoder.onnx"):
        (tmp / name).write_bytes(z.read(name))

rows = [json.loads(l) for l in TEST.read_text(
    encoding="utf-8").splitlines() if l.strip()][:N]

tok = AutoTokenizer.from_pretrained(
    "/tmp/urd-diac-rt", use_fast=False) if (tmp / "tokenizer_config.json").exists() else None

import onnxruntime as ort

es = ort.InferenceSession(str(tmp / "encoder.onnx"),
                          providers=["CPUExecutionProvider"])
ds = ort.InferenceSession(str(tmp / "decoder.onnx"),
                          providers=["CPUExecutionProvider"])

model = AutoModelForSeq2SeqLM.from_pretrained(
    "/tmp/urd-diac-rt", from_tf=False).eval()
# the HF save needs tokenizer files — copy from the zip's sibling best/
for f in ("tokenizer_config.json", "special_tokens_map.json",
          "added_tokens.json"):
    src = Path("/tmp/fas-83601/checkpoint-83601") / f
    if src.exists() and not (tmp / f).exists():
        (tmp / f).write_bytes(src.read_bytes())
tok = AutoTokenizer.from_pretrained(str(tmp))

for ri, row in enumerate(rows):
    src = row["src"]
    ids = torch.tensor([[b + 3 for b in src.encode("utf-8")][:512]])
    with torch.no_grad():
        h_t = model.get_encoder()(input_ids=ids)[0]
    h_o = es.run(None, {"input_ids": ids.numpy().astype(np.int64)})[0]
    enc_gap = float(np.abs(h_t.numpy().astype(np.float32)
                           - h_o.astype(np.float32)).max())
    cur_t = torch.zeros((1, 1), dtype=torch.long)
    cur_o = np.zeros((1, 1), dtype=np.int64)
    first_div = None
    dec_out = []
    for step in range(200):
        with torch.no_grad():
            lg_t = model.lm_head(model.get_decoder()(
                input_ids=cur_t, encoder_hidden_states=h_t)[0])
        lg_o = ds.run(None, {"input_ids": cur_o,
                             "encoder_hidden_states": h_o})[0]
        t_t = int(lg_t[:, -1, :].argmax(-1)[0])
        t_o = int(lg_o[:, -1, :].argmax(-1)[0])
        gap = float(np.abs(lg_t[:, -1, :].numpy().astype(np.float32)
                           - lg_o.astype(np.float32)).max())
        if t_t != t_o and first_div is None:
            first_div = (step, t_t, t_o, gap)
        dec_out.append(t_o)
        cur_t = torch.cat([cur_t, [[t_t]]], 1)
        cur_o = np.concatenate([cur_o, [[t_o]]], 1)
        if t_o == 1 and t_t == 1:
            break
    pred = bytes(max(0, t - 3) for t in dec_out).decode(
        "utf-8", errors="replace")
    print(json.dumps({
        "row": ri, "enc_max_gap": round(enc_gap, 5),
        "first_divergence": first_div,
        "pred_has_diacritics": any("ً" <= c <= "ٟ" or c == "ٰ"
                                   for c in pred),
        "pred_head": pred[:50],
    }, ensure_ascii=False), flush=True)

# NEXT-SESSION NOTE: run this as an HF job (l4x1, 30 min) with
# MODEL pointed at /ckpt/run-066-urd-diac-base/best (HF torch model)
# instead of the local onnx-only dir — the local run lacks the
# torch weights. The ORT half is unchanged.

"""WO51 closing probe — is the export defect real, or is it batched
decoding without an attention mask?

The exported encoder takes ONLY input_ids (no attention_mask). Any
batched decode that right-pads with id 0 corrupts encoder states;
single-row decode cannot. Measured on the urd-diac test split:

  1. PAD_GAPS  — ORT encoder: single vs padded-batch hidden states
  2. torch CER — batched generate WITH attention mask (gate mode)
  3. ORT CER   — strictly single-row decode (runtime mode)

Decision rule:
  pad gaps > 0 AND ort_single ~ torch -> export faithful; the
  divergence was batching; runtime contract = single-row (or
  re-export the encoder with an attention_mask input).

Usage (l4x1): python /ckpt/run-066-urd-diac-base/probe_single_vs_batch.py
"""

import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

CKPT = os.environ.get("P_CKPT", "/ckpt/run-066-urd-diac-base/best")
ZIP = os.environ.get("P_ZIP", "/ckpt/run-066-urd-diac-base/export/urd-diac-base-1.0.zip")
DATA = os.environ.get("P_DATA", "/data-dia/data-diacrit/test.jsonl")
N = int(os.environ.get("P_N", "300"))


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()][:N]
srcs = [r["src"] for r in rows]
golds = [r["tgt"] for r in rows]

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForSeq2SeqLM.from_pretrained(CKPT).eval().cuda()
enc_ids = [tok(s, truncation=True, max_length=512).input_ids for s in srcs]

import onnxruntime as ort

tmp = tempfile.mkdtemp()
with zipfile.ZipFile(ZIP) as z:
    for n in ("encoder.onnx", "decoder.onnx"):
        (Path(tmp) / n).write_bytes(z.read(n))
es = ort.InferenceSession(str(Path(tmp) / "encoder.onnx"),
                          providers=["CPUExecutionProvider"])
ds = ort.InferenceSession(str(Path(tmp) / "decoder.onnx"),
                          providers=["CPUExecutionProvider"])

sub = enc_ids[:8]
mx = max(len(e) for e in sub)
pad = np.zeros((len(sub), mx), dtype=np.int64)
for k, e in enumerate(sub):
    pad[k, :len(e)] = e
hb = es.run(None, {"input_ids": pad})[0]
gaps = []
for k, e in enumerate(sub):
    hs = es.run(None, {"input_ids": pad[k:k + 1, :len(e)].copy()})[0]
    gaps.append(round(float(np.abs(hs[0].astype(np.float32)
                                   - hb[k, :len(e)].astype(np.float32)).max()), 4))
print("PAD_GAPS", gaps, flush=True)

preds_t = []
with torch.no_grad():
    for i in range(0, len(srcs), 8):
        b = tok(srcs[i:i + 8], return_tensors="pt", padding=True,
                truncation=True, max_length=512).to("cuda")
        gen = model.generate(**b, max_new_tokens=512, num_beams=1)
        preds_t.extend(tok.batch_decode(gen, skip_special_tokens=True))
tc = sum(lev(p, g) / max(1, len(g)) for p, g in zip(preds_t, golds)) / len(golds)
print(f"[torch-batched-mask] cer={tc:.4f}", flush=True)
del model
torch.cuda.empty_cache()


def ort_single(e):
    h = es.run(None, {"input_ids": np.array([e], dtype=np.int64)})[0]
    cur = np.zeros((1, 1), dtype=np.int64)
    seq = []
    for _ in range(512):
        lg = ds.run(None, {"input_ids": cur, "encoder_hidden_states": h})[0]
        t = int(lg[:, -1, :].argmax(-1)[0])
        if t == 1:
            break
        seq.append(t)
        cur = np.concatenate([cur, [[t]]], 1)
    return bytes(max(0, t - 3) for t in seq).decode("utf-8", "replace")


preds_o = [ort_single(e) for e in enc_ids]
oc = sum(lev(p, g) / max(1, len(g)) for p, g in zip(preds_o, golds)) / len(golds)
agree = sum(a == b for a, b in zip(preds_t, preds_o)) / len(preds_t)
res = {"rows": len(rows), "pad_gaps": gaps,
       "torch_batched_cer": round(tc, 4), "ort_single_cer": round(oc, 4),
       "single_vs_batched_agree": round(agree, 4)}
print(json.dumps(res), flush=True)
out = Path(os.environ.get("P_OUT", "/ckpt/run-066-urd-diac-base/export/probe_single_vs_batch.json"))
out.write_text(json.dumps(res, indent=1))
try:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    dst = f"buckets/Interscript/isx-training/{out.relative_to(Path('/ckpt'))}"
    with open(out, "rb") as f, fs.open(dst, "wb") as g:
        g.write(f.read())
    print("synced", fs.exists(dst), flush=True)
except Exception as e:  # noqa: BLE001
    print(f"sync failed: {e}", flush=True)

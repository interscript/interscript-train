"""WO51 remedy (b) verification — batched-with-mask decode parity.

Exports the encoder WITH an attention_mask input (EXPORT_ATTENTION_MASK=1)
and proves the WO51 defect class is gone: batched padded decode WITH
masks must match torch. PAD_GAPS must be ~0 (the maskless artifact
showed up to 0.13).

Usage (l4x1, after export_imf_hf.py ran with EXPORT_ATTENTION_MASK=1):
    python /ckpt/run-066-urd-diac-base/export/probe_batched_mask.py
"""

import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

CKPT = os.environ.get("P_CKPT", "/ckpt/run-066-urd-diac-base/best")
ZIP = os.environ.get("P_ZIP", "/ckpt/run-066-urd-diac-base/export/urd-diac-base-1.0-masked.zip")
DATA = os.environ.get("P_DATA", "/data-dia/data-diacrit/test.jsonl")
N = 64


def lev(a, b):
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
tok = AutoTokenizer.from_pretrained(CKPT)
enc_ids = [tok(s, truncation=True, max_length=512).input_ids for s in srcs]

tmp = tempfile.mkdtemp()
with zipfile.ZipFile(ZIP) as z:
    for n in ("encoder.onnx", "decoder.onnx"):
        (Path(tmp) / n).write_bytes(z.read(n))
es = ort.InferenceSession(str(Path(tmp) / "encoder.onnx"),
                          providers=["CPUExecutionProvider"])
ds = ort.InferenceSession(str(Path(tmp) / "decoder.onnx"),
                          providers=["CPUExecutionProvider"])
enc_inputs = {i.name for i in es.get_inputs()}
HAS_MASK = "attention_mask" in enc_inputs
print(f"[zip] encoder inputs: {sorted(enc_inputs)} | masked={HAS_MASK}",
      flush=True)


def pad_batch(batch_ids):
    mx = max(len(e) for e in batch_ids)
    ids = np.zeros((len(batch_ids), mx), dtype=np.int64)
    mask = np.zeros((len(batch_ids), mx), dtype=np.int64)
    for k, e in enumerate(batch_ids):
        ids[k, :len(e)] = e
        mask[k, :len(e)] = 1
    return ids, mask


# 1) encoder-state: batched-with-mask vs single (PAD_GAPS must be ~0)
sub = enc_ids[:8]
bids, bmask = pad_batch(sub)
feed = {"input_ids": bids, "attention_mask": bmask} if HAS_MASK else \
    {"input_ids": bids}
hb = es.run(None, feed)[0]
gaps = []
for k, e in enumerate(sub):
    single = np.array([e], dtype=np.int64)
    hf = {"input_ids": single, "attention_mask": np.ones_like(single)} \
        if HAS_MASK else {"input_ids": single}
    hs = es.run(None, hf)[0]
    gaps.append(round(float(np.abs(hs[0] - hb[k, :len(e)]).max()), 4))
print("PAD_GAPS", gaps, flush=True)

# 2) batched decode: ORT (padded + mask) vs torch generate (padded + mask)
model = AutoModelForSeq2SeqLM.from_pretrained(CKPT).eval().cuda()
preds_t = []
with torch.no_grad():
    for i in range(0, len(srcs), 8):
        b = tok(srcs[i:i + 8], return_tensors="pt", padding=True,
                truncation=True, max_length=512).to("cuda")
        gen = model.generate(**b, max_new_tokens=512, num_beams=1)
        preds_t.extend(tok.batch_decode(gen, skip_special_tokens=True))
del model
torch.cuda.empty_cache()

preds_o = []
for i in range(0, len(enc_ids), 8):
    bids, bmask = pad_batch(enc_ids[i:i + 8])
    feed = {"input_ids": bids, "attention_mask": bmask} if HAS_MASK else \
        {"input_ids": bids}
    h = es.run(None, feed)[0]
    cur = np.zeros((len(bids), 1), dtype=np.int64)
    done = np.zeros(len(bids), dtype=bool)
    seqs = [[] for _ in range(len(bids))]
    for _ in range(512):
        lg = ds.run(None, {"input_ids": cur,
                           "encoder_hidden_states": h})[0]
        nxt = lg[:, -1, :].argmax(-1)
        for j in range(len(bids)):
            if done[j]:
                continue
            t = int(nxt[j])
            if t == 1:
                done[j] = True
            else:
                seqs[j].append(t)
        cur = np.concatenate([cur, np.where(done, np.int64(1),
                                            nxt.astype(np.int64))[:, None]],
                             1)
        if done.all():
            break
    preds_o.extend(bytes(max(0, t - 3) for t in q)
                   .decode("utf-8", "replace") for q in seqs)

agree = sum(a == b for a, b in zip(preds_t, preds_o)) / len(preds_t)
res = {
    "rows": len(srcs), "masked_export": HAS_MASK,
    "pad_gaps": gaps,
    "torch_batched_cer": round(sum(lev(p, g) / max(1, len(g))
                                   for p, g in zip(preds_t, golds)) / N, 4),
    "ort_batched_cer": round(sum(lev(p, g) / max(1, len(g))
                                 for p, g in zip(preds_o, golds)) / N, 4),
    "batched_agree": round(agree, 4),
}
print(json.dumps(res), flush=True)
out = Path(ZIP).parent / "probe_batched_mask.json"
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

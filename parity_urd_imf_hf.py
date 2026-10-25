"""WO38 export parity — torch vs int8-ONNX on the same 300 test rows.

Runs AFTER export_imf_hf.py in the same job. Greedy byte decode on
both engines; reports raw CER for each + row-level agreement. The
byt5-byte-table rule: ids are byte+3 with trailing EOS; decode is
(ids - 3) -> bytes -> utf-8, not tok.decode assumptions.

Env: R8_RUN (default run-059-urd-g2p-base), PARITY_N (default 300).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

RUN = os.environ.get("R8_RUN", "run-059-urd-g2p-base")
N = int(os.environ.get("PARITY_N", "300"))
PRECISION = os.environ.get("PARITY_PRECISION", "int8")  # int8 | fp32
CKPT = Path("/ckpt") / RUN / "best"
EXPORT = Path("/ckpt") / RUN / "export"
DATA = Path(os.environ.get("PARITY_DATA", "/data-ru/data/test.jsonl"))
MAXLEN = int(os.environ.get("PARITY_MAXLEN", "512"))


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines()
        if l.strip()][:N]
srcs = [r["src"] for r in rows]
golds = [r["tgt"] for r in rows]
print(f"[data] rows={len(rows)}", flush=True)


def cer_of(preds):
    return float(np.mean([lev(p, g) / max(1, len(g)) for p, g in zip(preds, golds)]))


# ---- torch engine (the gate engine) ----
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForSeq2SeqLM.from_pretrained(CKPT).eval().cuda()
torch_preds = []
with torch.no_grad():
    for i in range(0, len(srcs), 16):
        enc = tok(srcs[i:i+16], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAXLEN).to("cuda")
        with torch.autocast("cuda", torch.bfloat16):
            gen = model.generate(**enc, max_new_tokens=MAXLEN, num_beams=1,
                                 repetition_penalty=float(os.environ.get("PARITY_REP_PENALTY", "1.0")),
                                 no_repeat_ngram_size=int(os.environ.get("PARITY_NO_REPEAT_NGRAM", "0")))
        torch_preds.extend(tok.batch_decode(gen, skip_special_tokens=True))
torch_cer = cer_of(torch_preds)
print(f"[torch] cer={torch_cer:.4f}", flush=True)
del model
torch.cuda.empty_cache()

# ---- ONNX int8 engine ----
import onnxruntime as ort  # noqa: E402

EOS, PAD = 1, 0
import tempfile
import zipfile

if PRECISION == "zip":
    zips = sorted(EXPORT.glob("*.zip"))
    assert zips, "no staged zip in export dir"
    _tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(zips[0]) as z:
        for name in ("encoder.onnx", "decoder.onnx"):
            open(Path(_tmp) / name, "wb").write(z.read(name))
    ENC, DEC = Path(_tmp) / "encoder.onnx", Path(_tmp) / "decoder.onnx"
    print(f"[zip] verifying shipped bytes: {zips[0].name}", flush=True)
else:
    suffix = "" if PRECISION == "int8" else "-fp32"
    ENC, DEC = EXPORT / f"encoder{suffix}.onnx", EXPORT / f"decoder{suffix}.onnx"
provs = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
es = ort.InferenceSession(str(ENC), providers=provs)
ds = ort.InferenceSession(str(DEC), providers=provs)
ENC_IN = es.get_inputs()[0].name
DEC_IN = [i.name for i in ds.get_inputs()]
print(f"[onnx] encoder input: {ENC_IN}; decoder inputs: {DEC_IN}", flush=True)
onnx_preds = []
BS = int(os.environ.get('PARITY_BS', '8'))
with torch.no_grad():
    for i in range(0, len(srcs), BS):
        batch = srcs[i:i+BS]
        enc_ids = [tok(s, truncation=True, max_length=MAXLEN).input_ids
                   for s in batch]
        mx = max(len(e) for e in enc_ids)
        ids = np.zeros((len(batch), mx), dtype=np.int64)
        for j, e in enumerate(enc_ids):
            ids[j, :len(e)] = e
        h = es.run(None, {ENC_IN: ids})[0]

        cur = np.full((len(batch), 1), PAD, dtype=np.int64)
        done = np.zeros(len(batch), dtype=bool)
        seqs = [[] for _ in batch]
        for _ in range(MAXLEN):
            lg = ds.run(None, {DEC_IN[0]: cur, DEC_IN[1]: h})[0]
            nxt = lg[:, -1, :].argmax(-1)
            ngram = int(os.environ.get("PARITY_NO_REPEAT_NGRAM", "0"))
            for j in range(len(batch)):
                if done[j]:
                    continue
                t = int(nxt[j])
                if ngram and len(seqs[j]) >= ngram - 1:
                    prefix = tuple(seqs[j][-(ngram - 1):])
                    seen = {tuple(seqs[j][k:k + ngram])
                            for k in range(len(seqs[j]) - ngram + 1)}
                    if (prefix + (t,)) in seen:
                        blocked = lg[j, -1, :].copy()
                        blocked[t] = np.float32("-inf")
                        t = int(blocked.argmax(-1))
                if t == EOS:
                    done[j] = True
                else:
                    seqs[j].append(t)
            nxt_col = np.where(done, np.int64(PAD),
                               nxt.astype(np.int64)).reshape(-1, 1)
            cur = np.concatenate([cur, nxt_col], axis=1)
            if done.all():
                break
        for seq in seqs:
            onnx_preds.append(bytes(max(0, t - 3) for t in seq)
                              .decode("utf-8", errors="replace"))
onnx_cer = cer_of(onnx_preds)
agree = float(np.mean([a == b for a, b in zip(torch_preds, onnx_preds)]))
verdict = {"rows": len(rows), "precision": PRECISION,
           "torch_cer": round(torch_cer, 4),
           "onnx_cer": round(onnx_cer, 4),
           "cer_delta": round(abs(torch_cer - onnx_cer), 4),
           "row_exact_match": round(agree, 4)}
print(json.dumps(verdict), flush=True)
for g, t, o in list(zip(golds, torch_preds, onnx_preds))[:3]:
    print(f"  gold:  {g[:50]}\n  torch: {t[:50]}\n  onnx:  {o[:50]}", flush=True)
(EXPORT / "parity.json").write_text(json.dumps(verdict, indent=1))

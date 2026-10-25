"""WO51 — the three-way numeric probe (torch-fp32 vs torch-bf16 vs ORT-fp32).

On N identical test rows, greedy-decode the same checkpoint three
ways and report row-exact agreement + pairwise CERs:
  A. torch bf16 autocast on cuda (the gate engine)
  B. torch fp32 on cuda
  C. ORT fp32 on the shipped zip's graphs

Decision rule:
  B≈A and C≈B  -> export is faithful; artifact CER is the model's
  B≈A, C != B  -> ONNX conversion defect (fix the export)
  B != A       -> the model is numerically fragile (bf16 trained in
                  fp32-land cascades); retrain-for-robustness decision

Env: W_RUN (checkpoint), Z_RUN (dir holding export/<id>.zip),
W_N (rows). Usage (l4x1): python probe3way.py
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

W_RUN = os.environ.get("W_RUN", "run-066-urd-diac-base")
Z_RUN = W_RUN
W_N = int(os.environ.get("W_N", "150"))
CKPT = Path("/ckpt") / W_RUN / "best"
EXPORT = Path("/ckpt") / Z_RUN / "export"
DATA = Path(os.environ.get("PARITY_DATA", "/data-dia/data-diacrit/test.jsonl"))
MAXLEN = 512


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    rows = [json.loads(l) for l in DATA.read_text(
        encoding="utf-8").splitlines() if l.strip()][:W_N]
    srcs = [r["src"] for r in rows]
    golds = [r["tgt"] for r in rows]

    def cer(preds):
        return sum(lev(p, g) / max(1, len(g)) for p, g in zip(preds, golds)) / len(golds)

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT)

    def torch_decode(model):
        out = []
        with torch.no_grad():
            for i in range(0, len(srcs), 8):
                enc = tok(srcs[i:i + 8], return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to("cuda")
                gen = model.generate(**enc, max_new_tokens=512, num_beams=1)
                out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return out

    model_bf = AutoModelForSeq2SeqLM.from_pretrained(CKPT).eval().cuda()
    preds_bf = torch_decode(model_bf)
    print(f"[A bf16] cer={cer(preds_bf):.4f}", flush=True)

    model_f32 = AutoModelForSeq2SeqLM.from_pretrained(
        CKPT, torch_dtype=torch.float32).eval().cuda()
    preds_f32 = torch_decode(model_f32)
    print(f"[B fp32] cer={cer(preds_f32):.4f}", flush=True)
    del model_bf, model_f32
    torch.cuda.empty_cache()

    import onnxruntime as ort

    zips = sorted(EXPORT.glob("*.zip"))
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(zips[0]) as z:
        for name in ("encoder.onnx", "decoder.onnx"):
            open(Path(tmp) / name, "wb").write(z.read(name))
    es = ort.InferenceSession(str(Path(tmp) / "encoder.onnx"),
                              providers=["CPUExecutionProvider"])
    ds = ort.InferenceSession(str(Path(tmp) / "decoder.onnx"),
                              providers=["CPUExecutionProvider"])
    ENC_IN = es.get_inputs()[0].name
    DEC_IN = [i.name for i in ds.get_inputs()]
    preds_ort = []
    for i in range(0, len(srcs), 8):
        batch = srcs[i:i + 8]
        enc_ids = [tok(s, truncation=True, max_length=512).input_ids
                   for s in batch]
        mx = max(len(e) for e in enc_ids)
        ids = np.zeros((len(batch), mx), dtype=np.int64)
        for j, e in enumerate(enc_ids):
            ids[j, :len(e)] = e
        h = es.run(None, {ENC_IN: ids})[0]
        cur = np.zeros((len(batch), 1), dtype=np.int64)
        done = np.zeros(len(batch), dtype=bool)
        seqs = [[] for _ in batch]
        for _ in range(512):
            lg = ds.run(None, {DEC_IN[0]: cur, DEC_IN[1]: h})[0]
            nxt = lg[:, -1, :].argmax(-1)
            for j in range(len(batch)):
                if done[j]:
                    continue
                t = int(nxt[j])
                if t == 1:
                    done[j] = True
                else:
                    seqs[j].append(t)
            nxt_col = np.where(done, np.int64(0),
                               nxt.astype(np.int64)).reshape(-1, 1)
            cur = np.concatenate([cur, nxt_col], axis=1)
            if done.all():
                break
        preds_ort.extend(bytes(max(0, t - 3) for t in q)
                         .decode("utf-8", errors="replace") for q in seqs)
    print(f"[C ort] cer={cer(preds_ort):.4f}", flush=True)

    def agree(x, y):
        return sum(a == b for a, b in zip(x, y)) / len(x)

    res = {"rows": len(rows), "n": W_N,
           "cer_bf16": round(cer(preds_bf), 4),
           "cer_fp32_torch": round(cer(preds_f32), 4),
           "cer_ort": round(cer(preds_ort), 4),
           "agree_bf16_fp32torch": round(agree(preds_bf, preds_f32), 4),
           "agree_bf16_ort": round(agree(preds_bf, preds_ort), 4),
           "agree_fp32torch_ort": round(agree(preds_f32, preds_ort), 4)}
    print(json.dumps(res, indent=1), flush=True)
    (EXPORT / "probe3way.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

"""WO44 — Arabic IMF parity: torch vs the shipped zip's ONNX bytes.

300 WikiNews-2024 windows through (a) the HF checkpoint (greedy,
windowed protocol, project_haraqat, first-alternate decode) and
(b) the staged zip's encoder/decoder.onnx via the same greedy loop.
Reports WER/DER for both engines + row agreement.

Env: R8_RUN (checkpoint run dir), PARITY_N (300).
Gate: engine WER delta small enough that the shipped artifact carries
the gate numbers (fp32: near-zero; int8: quantization damage shows).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

RUN = os.environ.get("R8_RUN", "run-047-large-strip")
N = int(os.environ.get("PARITY_N", "300"))
CKPT = Path("/ckpt") / RUN / "best"
EXPORT = Path("/ckpt") / RUN / "export"
DATA = Path("/train_data")
WINDOW = 600
DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")


def split_windows(text: str, budget: int = WINDOW) -> list[str]:
    if len(text.encode("utf-8")) <= budget:
        return [text]
    words, cur, n, wins = text.split(), [], 0, []
    for w in words:
        c = len(w.encode("utf-8")) + 1
        if cur and n + c > budget:
            wins.append(" ".join(cur)); cur, n = [], 0
        cur.append(w); n += c
    if cur:
        wins.append(" ".join(cur))
    return wins


def project_haraqat(pred: str, text: str) -> str:
    from difflib import SequenceMatcher

    ph = [""]
    for ch in pred:
        if DIACRITICS_RE.match(ch):
            ph[-1] += ch
        else:
            ph.append("")
    ph = ph[1:]
    pl = [c for c in pred if not DIACRITICS_RE.match(c)]
    tl = [c for c in text if not DIACRITICS_RE.match(c)]
    sm = SequenceMatcher(None, tl, pl, autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                out.append(tl[i1 + k] + ph[j1 + k])
        else:
            for k in range(i1, i2):
                out.append(tl[k])
    return "".join(out)


bench = [l for l in (DATA / "wikinews/WikiNews_2024_Multi_Ref.txt.diac")
         .read_text(encoding="utf-8").splitlines() if l.strip()][:N]
inputs = [DIACRITICS_RE.sub("", l) for l in bench]
windows, counts = [], []
for text in inputs:
    ws = split_windows(text)
    counts.append(len(ws))
    windows.extend(ws)
print(f"[data] paras={len(inputs)} windows={len(windows)}", flush=True)


def first_alt(p):
    return " ".join(w.split("/", 1)[0] for w in p.split())


def wer_der(paras):
    import eval_wikinews_multiref as WN

    fa = [first_alt(p) for p in paras]
    s = WN.score(fa, bench, skip_last=False)
    return {k: round(v, 4) for k, v in s.items() if isinstance(v, (int, float))}


# ---- torch engine ----
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForSeq2SeqLM.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16).eval().cuda()


def gen_windows(sess):
    preds = []
    with torch.no_grad():
        for i in range(0, len(windows), 16):
            enc = tok(windows[i:i + 16], return_tensors="pt", padding=True,
                      truncation=True, max_length=WINDOW)
            if sess == "torch":
                out = model.generate(**{k: v.cuda() for k, v in enc.items()},
                                     max_new_tokens=WINDOW * 2, num_beams=1)
                preds.extend(tok.batch_decode(out, skip_special_tokens=True))
            else:
                ids = enc.input_ids.numpy().astype(np.int64)
                h = sess.run(None, {ENC_IN: ids})[0]
                cur = np.zeros((len(ids), 1), dtype=np.int64)
                done = np.zeros(len(ids), dtype=bool)
                seqs = [[] for _ in ids]
                for _ in range(WINDOW * 2):
                    lg = ds.run(None, {DEC_IN[0]: cur, DEC_IN[1]: h})[0]
                    nxt = lg[:, -1, :].argmax(-1)
                    for j in range(len(ids)):
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
                preds.extend(bytes(max(0, t - 3) for t in s)
                             .decode("utf-8", errors="replace") for s in seqs)
            if (i // 16) % 5 == 0:
                print(f"[{sess}] {i + 16}/{len(windows)}", flush=True)
    paras = []
    k = 0
    for text, c in zip(inputs, counts):
        paras.append(project_haraqat(" ".join(preds[k:k + c]), text))
        k += c
    return paras


torch_paras = gen_windows("torch")
del model
torch.cuda.empty_cache()
torch_scores = wer_der(torch_paras)
print(f"[torch] {json.dumps(torch_scores)}", flush=True)

import onnxruntime as ort  # noqa: E402

zips = sorted(EXPORT.glob("*.zip"))
assert zips, f"no staged zip in {EXPORT}"
_tmp = tempfile.mkdtemp()
with zipfile.ZipFile(zips[0]) as z:
    for name in ("encoder.onnx", "decoder.onnx"):
        open(Path(_tmp) / name, "wb").write(z.read(name))
print(f"[zip] verifying shipped bytes: {zips[0].name}", flush=True)
es = ort.InferenceSession(str(Path(_tmp) / "encoder.onnx"),
                          providers=["CPUExecutionProvider"])
ds = ort.InferenceSession(str(Path(_tmp) / "decoder.onnx"),
                          providers=["CPUExecutionProvider"])
ENC_IN = es.get_inputs()[0].name
DEC_IN = [i.name for i in ds.get_inputs()]
print(f"[onnx] encoder input: {ENC_IN}; decoder inputs: {DEC_IN}", flush=True)
onnx_paras = gen_windows(es)
onnx_scores = wer_der(onnx_paras)
print(f"[onnx] {json.dumps(onnx_scores)}", flush=True)

verdict = {"run": RUN, "zip": zips[0].name, "torch": torch_scores,
           "onnx_zip": onnx_scores}
print(json.dumps(verdict), flush=True)
(EXPORT / "parity_ar.json").write_text(json.dumps(verdict, indent=1))

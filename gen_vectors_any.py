"""Generate conformance vectors for any shipped IMF zip (WO55 follow-up).

Generalizes gen_conformance_vectors.py: env-driven so every model gets
frozen fixtures from the same reference decoder.

Env: G_ZIP, G_CKPT (unused — vectors need only the zip), G_DATA,
G_DATA_KEY (json field holding src; default src), G_OUT, G_MODEL, G_N.
"""

import json
import os
import tempfile
import zipfile
from pathlib import Path

G_ZIP = os.environ["G_ZIP"]
G_DATA = os.environ["G_DATA"]
G_OUT = os.environ["G_OUT"]
G_MODEL = os.environ["G_MODEL"]
G_SRC_KEY = os.environ.get("G_SRC_KEY", "src")
G_N = int(os.environ.get("G_N", "12"))
G_SOURCE = os.environ.get("G_SOURCE", "")

rows = [json.loads(l) for l in open(G_DATA, encoding="utf-8") if l.strip()]
srcs = [r[G_SRC_KEY].strip() for r in rows][:G_N]

tmp = Path(tempfile.mkdtemp())
with zipfile.ZipFile(G_ZIP) as z:
    for name in ("encoder.onnx", "decoder.onnx"):
        (tmp / name).write_bytes(z.read(name))
import numpy as np
import onnxruntime as ort

opts = ort.SessionOptions()
opts.intra_op_num_threads = 8
es = ort.InferenceSession(str(tmp / "encoder.onnx"), opts,
                          providers=["CPUExecutionProvider"])
ds = ort.InferenceSession(str(tmp / "decoder.onnx"), opts,
                          providers=["CPUExecutionProvider"])
enc_in = es.get_inputs()[0].name
dec_in = [i.name for i in ds.get_inputs()]


def decode(src: str) -> str:
    ids = np.array([[b + 3 for b in src.encode("utf-8")] + [1]], dtype=np.int64)
    h = es.run(None, {enc_in: ids})[0]
    cur = np.zeros((1, 1), dtype=np.int64)
    seq = []
    for _ in range(512):
        lg = ds.run(None, {dec_in[0]: cur, dec_in[1]: h})[0]
        t = int(lg[:, -1, :].argmax(-1)[0])
        if t == 1:
            break
        seq.append(t)
        cur = np.concatenate([cur, [[t]]], 1)
    return bytes(max(0, t - 3) for t in seq).decode("utf-8", "replace")


vectors = {
    "model": G_MODEL,
    "index_version": 12,
    "decode": {"mode": "single-row-greedy"},
    "source": G_SOURCE or f"first {G_N} rows of the gate split",
    "vectors": [{"src": s, "out": decode(s)} for s in srcs],
}
out = Path(G_OUT)
out.write_text(json.dumps(vectors, ensure_ascii=False, indent=1),
               encoding="utf-8")
print(json.dumps({"rows": len(vectors["vectors"]),
                  "first_out": vectors["vectors"][0]["out"][:24]}), flush=True)
try:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    dst = f"buckets/Interscript/isx-training/{out.relative_to(Path('/ckpt'))}"
    with open(out, "rb") as f, fs.open(dst, "wb") as g:
        g.write(f.read())
    print("synced", fs.exists(dst), flush=True)
except Exception as e:  # noqa: BLE001
    print(f"sync failed: {e}", flush=True)

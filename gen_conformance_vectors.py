"""Generate the urd-diac-base conformance vectors (WO52 M5).

Decodes the first 12 rows of the urd-diac test split through the fp16
zip with the reference single-row ORT decoder and writes the frozen
vectors file (model id from models.yaml, mode single-row-greedy).

Usage (cpu-basic or l4x1):
    python /ckpt/run-066-urd-diac-base/gen_conformance_vectors.py
"""

import json
import os
import tempfile
import zipfile
from pathlib import Path

VECTORS_OUT = Path("/ckpt/run-066-urd-diac-base/urd-diac-base-vectors.json")
ZIP = Path("/ckpt/run-066-urd-diac-base/export/urd-diac-base-1.0-fp16.zip")
TEST = Path("/data-dia/data-diacrit/test.jsonl")
N = 12

rows = [json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()]
srcs = [r["src"].strip() for r in rows][:N]

tmp = Path(tempfile.mkdtemp())
with zipfile.ZipFile(ZIP) as z:
    for name in ("encoder.onnx", "decoder.onnx"):
        (tmp / name).write_bytes(z.read(name))
import numpy as np
import onnxruntime as ort

opts = ort.SessionOptions()
opts.intra_op_num_threads = 8
es = ort.InferenceSession(str(tmp / "encoder.onnx"), opts, providers=["CPUExecutionProvider"])
ds = ort.InferenceSession(str(tmp / "decoder.onnx"), opts, providers=["CPUExecutionProvider"])
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
    "model": "urd-diac-base-1.0",
    "index_version": 1,
    "decode": {"mode": "single-row-greedy"},
    "source": "first 12 rows of urd-dia#test (sha256 6d822ae7…)",
    "vectors": [{"src": s, "out": decode(s)} for s in srcs],
}
VECTORS_OUT.write_text(json.dumps(vectors, ensure_ascii=False, indent=1),
                       encoding="utf-8")
print(json.dumps({"rows": len(vectors["vectors"]),
                  "first_out": vectors["vectors"][0]["out"][:30]}), flush=True)
try:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    dst = f"buckets/Interscript/isx-training/{VECTORS_OUT.relative_to(Path('/ckpt'))}"
    with open(VECTORS_OUT, "rb") as f, fs.open(dst, "wb") as g:
        g.write(f.read())
    print("synced", fs.exists(dst), flush=True)
except Exception as e:  # noqa: BLE001
    print(f"sync failed: {e}", flush=True)

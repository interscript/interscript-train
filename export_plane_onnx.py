"""Export run-017 plane model to ONNX (+int8) and benchmark CPU decode
latency vs the shipped seq2seq int8static artifact.

One graph: (input_ids, plane_ids) -> per-position class logits. The
K-pass loop is host-side (feed argmax back as plane_ids) — no KV cache,
no per-step decode; each pass is a fully parallel forward.

Usage:
    python3 export_plane_onnx.py            # export + quantize + bench
    python3 export_plane_onnx.py --bench-only
    python3 export_plane_onnx.py --checkpoint /tmp/ck.pt --backbone google/byt5-small \
        --prefix plane-int8 --bench-skip
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import modal

OUT_DIR = Path("/tmp/plane-onnx")
RUN = "rababa_arabic_plane/run-017-plane"
SAMPLE = ("وَقَالَ رَسُولُ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ خَيْرُ النَّاسِ "
          "أَنْفَعُهُمْ لِلنَّاسِ " * 18)  # ~1.4KB window


def fetch_best() -> Path:
    ck = OUT_DIR / "model.pt"
    if ck.exists():
        return ck
    import subprocess

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", "rababa-checkpoints", f"{RUN}/best/model.pt", str(ck)],
        check=True, cwd=str(Path(__file__).resolve().parent),
    )
    return ck


def build_model(sd, backbone_name: str = "google/byt5-small"):
    import torch
    import torch.nn as nn
    from transformers import AutoModelForSeq2SeqLM

    classes = sd["classes"]
    n_classes = len(classes)
    backbone = AutoModelForSeq2SeqLM.from_pretrained(backbone_name)
    encoder = backbone.encoder
    encoder.load_state_dict(sd["encoder"])
    plane_emb = nn.Embedding(n_classes + 1, encoder.config.d_model)
    plane_emb.load_state_dict(sd["plane_emb"])
    head = nn.Linear(encoder.config.d_model, n_classes)
    head.load_state_dict(sd["head"])

    class PlaneModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.plane_emb = plane_emb
            self.head = head

        def forward(self, input_ids, plane_ids):
            embeds = self.encoder.embed_tokens(input_ids) + self.plane_emb(plane_ids)
            out = self.encoder(inputs_embeds=embeds, attention_mask=torch.ones_like(input_ids))
            return self.head(out.last_hidden_state)

    m = PlaneModel().eval()
    return m, classes


def export(checkpoint: Path | None = None, backbone: str = "google/byt5-small",
           prefix: str = "plane-1.0") -> None:
    import torch

    sd = torch.load(checkpoint or fetch_best(), map_location="cpu", weights_only=True)
    model, classes = build_model(sd, backbone)
    n_classes = len(classes)
    tok_ids = torch.tensor([[b + 3 for b in SAMPLE.encode("utf-8")[:1400]]], dtype=torch.long)
    plane = torch.full_like(tok_ids, n_classes)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp32 = OUT_DIR / f"{prefix}-fp32.onnx"
    torch.onnx.export(
        model, (tok_ids, plane), str(fp32),
        input_names=["input_ids", "plane_ids"],
        output_names=["class_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "plane_ids": {0: "batch", 1: "seq"},
            "class_logits": {0: "batch", 1: "seq"},
        },
        opset_version=14,
    )
    print(f"exported {fp32} ({fp32.stat().st_size / 1e6:.1f} MB)")

    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8 = OUT_DIR / f"{prefix}-int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    print(f"quantized {int8} ({int8.stat().st_size / 1e6:.1f} MB)")
    (OUT_DIR / "classes.json").write_text(
        __import__("json").dumps(classes, ensure_ascii=False), encoding="utf-8")


def bench(prefix: str = "plane-1.0") -> None:
    import json

    import numpy as np
    import onnxruntime as ort

    classes = json.loads((OUT_DIR / "classes.json").read_text(encoding="utf-8"))
    n_classes = len(classes)
    ids = np.array([[b + 3 for b in SAMPLE.encode("utf-8")[:1400]]], dtype=np.int64)

    for name in (f"{prefix}-fp32.onnx", f"{prefix}-int8.onnx"):
        sess = ort.InferenceSession(str(OUT_DIR / name), providers=["CPUExecutionProvider"])
        for k in (1, 2, 4):
            # warmup
            plane = np.full_like(ids, n_classes)
            for _ in range(k):
                logits = sess.run(None, {"input_ids": ids, "plane_ids": plane})[0]
                plane = logits.argmax(-1)
            t0 = time.perf_counter()
            for _ in range(3):
                plane = np.full_like(ids, n_classes)
                for _ in range(k):
                    logits = sess.run(None, {"input_ids": ids, "plane_ids": plane})[0]
                    plane = logits.argmax(-1)
            dt = (time.perf_counter() - t0) / 3
            print(f"{name} K={k}: {dt * 1000:.0f} ms/window "
                  f"({len(ids[0])} tokens, {k} parallel passes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--bench-skip", action="store_true")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--backbone", default="google/byt5-small")
    ap.add_argument("--prefix", default="plane-1.0")
    args = ap.parse_args()
    if not args.bench_only:
        export(args.checkpoint, args.backbone, args.prefix)
    if not args.bench_skip:
        bench(args.prefix)

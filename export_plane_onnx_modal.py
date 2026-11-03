"""Export the run-019 plane-large artifact ON MODAL.

The 5.2 GB fp32 checkpoint tears during host-side volume downloads
(two independent `modal volume get` calls returned different bytes),
so the export runs where the volume mounts natively: torch.load,
ONNX fp32 export, dynamic int8 quantize, then the int8 graph +
classes.json are staged back on the volume with their sha256s.

Usage:
    modal run --detach export_plane_onnx_modal.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

app = modal.App("rababa-plane-export-large")

RUN = "rababa_arabic_plane/run-019-plane-large"
BACKBONE = "google/byt5-large"
PREFIX = "plane-large-e2"
OUT = Path("/checkpoints") / RUN / "export"

checkpoints_volume = modal.Volume.from_name("rababa-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "onnx", "onnxruntime", "onnxscript", "numpy")
    .add_local_file(Path(__file__).parent / "export_plane_onnx.py", "/opt/export_plane_onnx.py", copy=True)
)


@app.function(image=image, timeout=3600, cpu=8, memory=32768,
              volumes={"/checkpoints": checkpoints_volume})
def export() -> dict:
    import sys
    sys.path.insert(0, "/opt")
    import torch
    from export_plane_onnx import build_model

    ck = Path("/checkpoints") / RUN / "best" / "model.pt"
    sd = torch.load(ck, map_location="cpu", weights_only=True)
    model, classes = build_model(sd, BACKBONE)
    n_classes = len(classes)

    tok_ids = torch.tensor([[b + 3 for b in ("و" * 100).encode("utf-8")[:1400]]], dtype=torch.long)
    plane = torch.full_like(tok_ids, n_classes)

    fp32 = Path("/tmp") / f"{PREFIX}-fp32.onnx"
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
        dynamo=False,
    )
    print(f"exported {fp32} ({fp32.stat().st_size / 1e6:.1f} MB)", flush=True)

    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8 = Path("/tmp") / f"{PREFIX}-int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    classes_bytes = json.dumps(classes, ensure_ascii=False, indent=1).encode("utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    int8_bytes = int8.read_bytes()
    (OUT / f"{PREFIX}-int8.onnx").write_bytes(int8_bytes)
    (OUT / "classes.json").write_bytes(classes_bytes)

    def sha(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    result = {
        "int8_sha256": sha(int8_bytes),
        "int8_size": len(int8_bytes),
        "classes_sha256": sha(classes_bytes),
    }
    (OUT / "sha256s.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    checkpoints_volume.commit()
    print(json.dumps(result, indent=1), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    handle = export.spawn()
    print(f"spawned {handle.object_id}; output = "
          f"rababa_checkpoints:{RUN}/export/{PREFIX}-int8.onnx", flush=True)

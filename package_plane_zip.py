"""Build a plane-model artifact zip per the IMF plane contract.

Bundle layout (verified by secryst-py PlaneModel.from_zip):
    metadata.yaml   id, precision, kind=plane, k_passes, classes, member sha256s
    plane.onnx      one graph (input_ids, plane_ids) -> per-position class logits
    classes.json    class id -> harakat combo ("" = bare)

Usage:
    python3 package_plane_zip.py --onnx /tmp/plane-onnx/plane-int8.onnx \
        --classes /tmp/plane-onnx/classes.json \
        --id ara-diac-plane-1.0 --precision int8 --out dist/ara-diac-plane-1.0.zip
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path


def build_plane_zip(model_id: str, precision: str, onnx: bytes, classes: list[str],
                    k_passes: int = 2) -> bytes:
    classes_bytes = json.dumps(classes, ensure_ascii=False, indent=1).encode("utf-8")
    members = {
        "plane.onnx": onnx,
        "classes.json": classes_bytes,
    }
    meta = (
        "id: {id}\n"
        "kind: plane\n"
        "precision: {precision}\n"
        "k_passes: {k}\n"
        "members:\n"
        "  plane.onnx: {onnx_sha}\n"
        "  classes.json: {classes_sha}\n"
    ).format(
        id=model_id,
        precision=precision,
        k=k_passes,
        onnx_sha=hashlib.sha256(onnx).hexdigest(),
        classes_sha=hashlib.sha256(classes_bytes).hexdigest(),
    ).encode("utf-8")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, data in [("metadata.yaml", meta), *members.items()]:
            zf.writestr(name, data)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--precision", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-passes", type=int, default=2)
    args = ap.parse_args()

    data = build_plane_zip(
        args.id, args.precision,
        Path(args.onnx).read_bytes(),
        json.loads(Path(args.classes).read_text(encoding="utf-8")),
        k_passes=args.k_passes,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"{out} ({len(data) / 1e6:.1f} MB) sha256={hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()

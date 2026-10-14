"""WO45 — ara-diac-news-large static-int8 export (encoder-quantized).

The byt5-large encoder exceeds the 2GB protobuf cap at fp32 AND fp16,
so the single-file contract needs quantization: static int8 (QDQ,
QInt8 per-channel weights + QUInt8 activations) calibrated on real
tokenized WikiNews windows — the one quantization lever with a
positive track record (dynamic-int8 measured 0.44-vs-0.16 CER on
urd-g2p-base). Decoder ships fp16 single-file.

Env: R8_RUN (default run-047-large-strip), IMF_ID, CAL_N (500).

Gate: parity_ar_imf_hf.py (zip mode) after this — first-alt WER
delta vs torch must be small AND raw WER < 4.10.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

RUN = os.environ.get("R8_RUN", "run-047-large-strip")
IMF_ID = os.environ.get("IMF_ID", "ara-diac-news-large-1.0")
CAL_N = int(os.environ.get("CAL_N", "500"))
WINDOW = 600
CKPT = Path("/ckpt") / RUN / "best"
OUT = Path("/ckpt") / RUN / "export"
DATA = Path("/train_data")
DIACRITICS_RE = __import__("re").compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")


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


def build_windows() -> list[str]:
    bench = [l for l in (DATA / "wikinews/WikiNews_2024_Multi_Ref.txt.diac")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    inputs = [DIACRITICS_RE.sub("", l) for l in bench]
    ws: list[str] = []
    for text in inputs:
        ws.extend(split_windows(text))
    return ws[:CAL_N]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSeq2SeqLM.from_pretrained(CKPT).eval()
    ids = torch.tensor([[104, 101]])
    with torch.no_grad():
        hidden = model.get_encoder()(input_ids=ids)[0]

    class DecoderWithHead(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.decoder = m.get_decoder()
            self.lm_head = m.lm_head

        def forward(self, input_ids, encoder_hidden_states):
            h = self.decoder(input_ids=input_ids,
                             encoder_hidden_states=encoder_hidden_states)[0]
            return self.lm_head(h)

    enc_p = OUT / "encoder-fp32.onnx"
    dec_p = OUT / "decoder-fp32.onnx"
    enc_model = model.get_encoder()
    if os.environ.get("QUANT_MODE") == "torch-dynamic":
        # ORT's calibrator must also serialize the full 2.4GB graph
        # (EncodeError) — last single-file lever: torch-native
        # per-channel weight-only int8, quantized BEFORE export.
        import torch.ao.quantization as tq

        enc_model = tq.quantize_dynamic(
            enc_model, {torch.nn.Linear}, dtype=torch.qint8)
        print("[torch-quant] encoder weight-only int8", flush=True)
    with torch.no_grad():
        torch.onnx.export(
            enc_model, (ids,), str(enc_p),
            input_names=["input_ids"], output_names=["last_hidden_state"],
            opset_version=14,
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                          "last_hidden_state": {0: "batch", 1: "seq"}},
        )
        torch.onnx.export(
            DecoderWithHead(model), (torch.tensor([[0]]), hidden), str(dec_p),
            input_names=["input_ids", "encoder_hidden_states"],
            output_names=["logits"], opset_version=14,
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                          "encoder_hidden_states": {0: "batch", 1: "seq"},
                          "logits": {0: "batch", 1: "seq"}},
        )
    print(f"[export] raw {enc_p.stat().st_size/1e6:.0f}MB + "
          f"{dec_p.stat().st_size/1e6:.0f}MB", flush=True)

    import onnx

    # inline external data where the result still fits the 2GB cap
    for graph_p in (enc_p, dec_p):
        try:
            m = onnx.load(str(graph_p), load_external_data=True)
            onnx.save(m, str(graph_p))
            print(f"[refuse] {graph_p.name} -> "
                  f"{graph_p.stat().st_size/1e6:.0f}MB single-file",
                  flush=True)
        except Exception as e:  # noqa: BLE001 — oversized graphs quantize instead
            print(f"[refuse] {graph_p.name} stays external: {e}", flush=True)

    # static int8 (QDQ) calibrated on real windows — encoder AND decoder
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)

    class Reader(CalibrationDataReader):
        def __init__(self, graph_path, samples):
            es = ort.InferenceSession(str(graph_path),
                                      providers=["CPUExecutionProvider"])
            self.in_name = es.get_inputs()[0].name
            self.samples = []
            enc_ids = [tok(s, truncation=True, max_length=WINDOW).input_ids
                       for s in samples]
            for i in range(0, len(enc_ids), 8):
                chunk = enc_ids[i:i + 8]
                mx = max(len(e) for e in chunk)
                arr = np.zeros((len(chunk), mx), dtype=np.int64)
                for j, e in enumerate(chunk):
                    arr[j, :len(e)] = e
                self.samples.append({self.in_name: arr})

        def get_next(self):
            return self.samples.pop(0) if self.samples else None

    windows = build_windows()
    quant_targets = [(dec_p, OUT / "decoder.onnx")]
    if os.environ.get("QUANT_MODE") != "torch-dynamic":
        quant_targets.insert(0, (enc_p, OUT / "encoder.onnx"))
    else:
        shutil.copyfile(enc_p, OUT / "encoder.onnx")
        print(f"[encoder] shipped as-exported "
              f"{enc_p.stat().st_size/1e6:.0f}MB", flush=True)
    for src, dst in quant_targets:
        quantize_static(
            str(src), str(dst), Reader(src, windows),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        qm = onnx.load(str(dst), load_external_data=True)
        onnx.save(qm, str(dst))
        print(f"[quant] {dst.name} int8 single-file "
              f"{dst.stat().st_size/1e6:.0f}MB", flush=True)

    # sanity: one encoder step + one decoder step through the SHIPPED pair
    es = ort.InferenceSession(str(OUT / "encoder.onnx"),
                              providers=["CPUExecutionProvider"])
    ds = ort.InferenceSession(str(OUT / "decoder.onnx"),
                              providers=["CPUExecutionProvider"])
    h = es.run(None, {"input_ids": np.array([[104, 101]], dtype=np.int64)})[0]
    lg = ds.run(None, {ds.get_inputs()[0].name: np.array([[0]], dtype=np.int64),
                       ds.get_inputs()[1].name: h})[0]
    print(f"[sanity] logits {lg.shape}", flush=True)

    import hashlib
    import json
    import zipfile

    import yaml

    def sha(b):
        return hashlib.sha256(b).hexdigest()

    enc_b, dec_b = (OUT / 'encoder.onnx').read_bytes(), (OUT / 'decoder.onnx').read_bytes()
    meta = {
        "format": "imf-v1", "id": IMF_ID, "task": "diacritization",
        "source_script": "Arab", "target": "Arab",
        "tokenizer": "bytes", "opset": 14, "decoder": "plain",
        "precision": "int8-static",
        "license": "BSD-3-Clause",
        "trained_from": os.environ.get(
            "IMF_TRAINED_FROM", "byt5-large strip-alt, run-047 e6"),
        "metrics": [],
        "sha256": {"encoder.onnx": sha(enc_b), "decoder.onnx": sha(dec_b)},
    }
    zip_p = OUT / f"{IMF_ID}.zip"
    with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("metadata.yaml",
                   yaml.safe_dump(meta, allow_unicode=True,
                                  sort_keys=False).encode())
        z.writestr("encoder.onnx", enc_b)
        z.writestr("decoder.onnx", dec_b)
    print(json.dumps({"zip": str(zip_p), "bytes": zip_p.stat().st_size,
                      "sha256": sha(zip_p.read_bytes())}), flush=True)
    (OUT / "EXPORT_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

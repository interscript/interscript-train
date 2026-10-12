# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.5.1",
#   "transformers==4.46.3",
#   "onnx==1.17.0",
#   "onnxruntime==1.20.1",
#   "numpy<2",
# ]
# ///
"""Export an r8 HF checkpoint (byt5 seq2seq) to the IMF v1 zip.

Plain-decoder tier (encoder.onnx + decoder.onnx, decoder: plain) — the
same contract the loaders accept; the KV tier (decoder-kv.onnx, used by
the ts speculative path) is a follow-up if the owner wants that tier.

Usage:
    hf jobs uv run --flavor l4x1 -d --timeout 2h \\
      -v hf://buckets/Interscript/isx-training:/ckpt:rw \\
      -e R8_RUN=run-024-arabic-r8-qcri \\
      -e IMF_ID=ara-diac-news-1.0 \\
      -e IMF_METRIC_DER=4.3911 \\
      -e IMF_METRIC_WER_WN=11.0305 \\
      export_imf_hf.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM

RUN = os.environ.get("R8_RUN", "run-024-arabic-r8-qcri")
IMF_ID = os.environ.get("IMF_ID", "ara-diac-news-1.0")
METRIC_DER = os.environ.get("IMF_METRIC_DER", "")
METRIC_WER = os.environ.get("IMF_METRIC_WER_WN", "")
CKPT = Path("/ckpt") / RUN / "best"
OUT_DIR = Path("/ckpt") / RUN / "export"


class DecoderWithHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.get_decoder()
        self.lm_head = model.lm_head

    def forward(self, input_ids, encoder_hidden_states):
        hidden = self.decoder(
            input_ids=input_ids, encoder_hidden_states=encoder_hidden_states
        )[0]
        return self.lm_head(hidden)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(CKPT).eval()
    ids = torch.tensor([[104, 101]])
    with torch.no_grad():
        hidden = model.get_encoder()(input_ids=ids)[0]

    enc_p, dec_p = OUT_DIR / "encoder-fp32.onnx", OUT_DIR / "decoder-fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model.get_encoder(), (ids,), str(enc_p),
            input_names=["input_ids"],
            output_names=["last_hidden_state"],
            opset_version=14,
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                          "last_hidden_state": {0: "batch", 1: "seq"}},
            use_external_data_format=False,
        )
        torch.onnx.export(
            DecoderWithHead(model), (torch.tensor([[0]]), hidden), str(dec_p),
            input_names=["input_ids", "encoder_hidden_states"],
            output_names=["logits"], opset_version=14,
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                          "encoder_hidden_states": {0: "batch", 1: "seq"},
                          "logits": {0: "batch", 1: "seq"}},
            use_external_data_format=False,
        )
    print(f"[export] fp32 {enc_p.stat().st_size/1e6:.0f}MB + {dec_p.stat().st_size/1e6:.0f}MB", flush=True)

    from onnxruntime.quantization import QuantType, quantize_dynamic

    enc_q, dec_q = OUT_DIR / "encoder.onnx", OUT_DIR / "decoder.onnx"
    quantize_dynamic(str(enc_p), str(enc_q), weight_type=QuantType.QInt8)
    quantize_dynamic(str(dec_p), str(dec_q), weight_type=QuantType.QInt8)
    enc_b = enc_q.read_bytes()
    dec_b = dec_q.read_bytes()
    print(f"[export] int8 {len(enc_b)/1e6:.0f}MB + {len(dec_b)/1e6:.0f}MB", flush=True)

    # sanity: single greedy step on CPU ORT
    import numpy as np
    import onnxruntime as ort

    es = ort.InferenceSession(str(enc_q), providers=["CPUExecutionProvider"])
    ds = ort.InferenceSession(str(dec_q), providers=["CPUExecutionProvider"])
    h = es.run(None, {"input_ids": np.array([[104, 101]], dtype=np.int64)})[0]
    lg = ds.run(None, {"input_ids": np.array([[0]], dtype=np.int64),
                       "encoder_hidden_states": h})[0]
    print(f"[sanity] logits shape {lg.shape}", flush=True)

    metrics = []
    if METRIC_DER:
        metrics.append({"name": "der_total_greedy", "value": float(METRIC_DER),
                        "source": f"interscript/interscript-models RESULTS {RUN}"})
    if METRIC_WER:
        metrics.append({"name": "wer_wikinews2024_multiref", "value": float(METRIC_WER),
                        "source": f"interscript/interscript-models RESULTS {RUN}"})

    meta = {
        "format": "imf-v1", "id": IMF_ID, "task": "diacritization",
        "source_script": "Arab", "target": "Arab",
        "tokenizer": "bytes", "opset": 14, "decoder": "plain",
        "precision": "int8", "license": "BSD-3-Clause",
        "trained_from": f"r7-init noisy-student, run {RUN} (TODO.sota WO17)",
        "metrics": metrics,
        "sha256": {"encoder.onnx": sha256_bytes(enc_b),
                   "decoder.onnx": sha256_bytes(dec_b)},
    }
    import yaml

    meta_b = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).encode()
    zip_p = OUT_DIR / f"{IMF_ID}.zip"
    with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("metadata.yaml", meta_b)
        z.writestr("encoder.onnx", enc_b)
        z.writestr("decoder.onnx", dec_b)
    print(json.dumps({"zip": str(zip_p), "bytes": zip_p.stat().st_size,
                      "sha256": sha256_bytes(zip_p.read_bytes())}), flush=True)
    (OUT_DIR / "EXPORT_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

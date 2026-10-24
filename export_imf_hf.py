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
METRIC_CER = os.environ.get("IMF_METRIC_CER", "")
METRIC_CER_NORM = os.environ.get("IMF_METRIC_CER_NORM", "")
TASK = os.environ.get("IMF_TASK", "diacritization")
SOURCE = os.environ.get("IMF_SOURCE", "Arab")
TARGET = os.environ.get("IMF_TARGET", "Arab")
TRAINED_FROM = os.environ.get("IMF_TRAINED_FROM",
                              "r7-init noisy-student, run {RUN} (TODO.sota WO17)")
# WO44: dynamic int8 quantization destroyed seq2seq decode quality
# (urd parity 0.44 vs 0.16 fp32 CER) — artifacts ship FP32 by default.
PRECISION = os.environ.get("EXPORT_PRECISION", "fp32")
# byt5-LARGE fp32 graphs exceed the 2GB protobuf boundary — torch emits
# external-data files no matter the pin, and the zip would ship a stub.
# fp16 halves each graph below the boundary (gate numbers were measured
# under bf16 autocast, so fp16 is the faithful tier).
EXPORT_DTYPE = os.environ.get("EXPORT_DTYPE", "fp32")
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
    _dt = {"fp32": torch.float32, "fp16": torch.float16}[EXPORT_DTYPE]
    MASKED = os.environ.get("EXPORT_ATTENTION_MASK", "") == "1"
    model = AutoModelForSeq2SeqLM.from_pretrained(CKPT, torch_dtype=_dt).eval()
    ids = torch.tensor([[104, 101]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        if MASKED:
            hidden = model.get_encoder()(input_ids=ids, attention_mask=mask)[0]
        else:
            hidden = model.get_encoder()(input_ids=ids)[0]

    enc_p, dec_p = OUT_DIR / "encoder-fp32.onnx", OUT_DIR / "decoder-fp32.onnx"
    with torch.no_grad():
        if MASKED:
            # WO51 remedy (b): the encoder takes attention_mask, so
            # batched padded decode is faithful; gated by a batched
            # parity probe before any release (parity.mode batched-mask)
            torch.onnx.export(
                model.get_encoder(), (ids, mask), str(enc_p),
                input_names=["input_ids", "attention_mask"],
                output_names=["last_hidden_state"],
                opset_version=14,
                dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                              "attention_mask": {0: "batch", 1: "seq"},
                              "last_hidden_state": {0: "batch", 1: "seq"}},
                use_external_data_format=False,
            )
        else:
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
    print(f"[export] attention_mask={MASKED}", flush=True)
    # torch emits external-data files for large graphs even with the pin;
    # re-fuse into single-file protobufs (fp16 graphs fit under 2GB)
    import onnx

    for graph_p in (enc_p, dec_p):
        m = onnx.load(str(graph_p), load_external_data=True)
        onnx.save(m, str(graph_p))
        print(f"[refuse] {graph_p.name} -> "
              f"{graph_p.stat().st_size/1e6:.0f}MB single-file", flush=True)
    print(f"[export] fp32 {enc_p.stat().st_size/1e6:.0f}MB + {dec_p.stat().st_size/1e6:.0f}MB", flush=True)

    if EXPORT_DTYPE == "fp16":
        # WO79/WO83: raw-fp16 graph IO is unportable (ruby has no Float16,
        # node ORT corrupts the boundary) — weights stay fp16, the graph
        # boundary is always fp32.
        from onnx import TensorProto, helper

        def _dims(t):
            return [d.dim_value if d.HasField("dim_value") else d.dim_param
                    for d in t.shape.dim]

        def widen_outputs(m):
            g = m.graph
            for out in list(g.output):
                t = out.type.tensor_type
                if t.elem_type != TensorProto.FLOAT16:
                    continue
                orig, raw = out.name, out.name + "__raw"
                out.name = raw
                for n in g.node:
                    for i, x in enumerate(n.output):
                        if x == orig:
                            n.output[i] = raw
                g.node.append(helper.make_node(
                    "Cast", [raw], [orig], to=TensorProto.FLOAT))
                g.output.append(helper.make_tensor_value_info(
                    orig, TensorProto.FLOAT, _dims(t)))
                g.output.remove(out)

        def narrow_input(m, name):
            g = m.graph
            inp = next((i for i in g.input if i.name == name), None)
            if inp is None or inp.type.tensor_type.elem_type != TensorProto.FLOAT16:
                return
            internal = name + "__f16"
            inp.type.tensor_type.elem_type = TensorProto.FLOAT
            for n in g.node:
                for i, x in enumerate(n.input):
                    if x == name:
                        n.input[i] = internal
            g.node.insert(0, helper.make_node(
                "Cast", [name], [internal], to=TensorProto.FLOAT16))

        for graph_p in (enc_p, dec_p):
            m = onnx.load(str(graph_p))
            widen_outputs(m)
            onnx.save(m, str(graph_p))
        dm = onnx.load(str(dec_p))
        narrow_input(dm, "encoder_hidden_states")
        onnx.save(dm, str(dec_p))
        print("[io] fp16 weights kept; graph boundary normalized to fp32 (WO79)", flush=True)

    if PRECISION == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        enc_q, dec_q = OUT_DIR / "encoder.onnx", OUT_DIR / "decoder.onnx"
        quantize_dynamic(str(enc_p), str(enc_q), weight_type=QuantType.QInt8)
        quantize_dynamic(str(dec_p), str(dec_q), weight_type=QuantType.QInt8)
        enc_b, dec_b = enc_q.read_bytes(), dec_q.read_bytes()
        shipped = "int8"
        print(f"[export] int8 {len(enc_b)/1e6:.0f}MB + {len(dec_b)/1e6:.0f}MB", flush=True)
    else:
        enc_b, dec_b = enc_p.read_bytes(), dec_p.read_bytes()
        shipped = "fp32"
        print(f"[export] shipping fp32 {len(enc_b)/1e6:.0f}MB + {len(dec_b)/1e6:.0f}MB", flush=True)

    # sanity: single greedy step on CPU ORT, over the SHIPPED bytes
    import tempfile

    import numpy as np
    import onnxruntime as ort

    _sd = tempfile.mkdtemp()
    (Path(_sd) / "encoder.onnx").write_bytes(enc_b)
    (Path(_sd) / "decoder.onnx").write_bytes(dec_b)
    es = ort.InferenceSession(str(Path(_sd) / "encoder.onnx"),
                              providers=["CPUExecutionProvider"])
    ds = ort.InferenceSession(str(Path(_sd) / "decoder.onnx"),
                              providers=["CPUExecutionProvider"])
    if MASKED:
        h = es.run(None, {"input_ids": np.array([[104, 101]], dtype=np.int64),
                          "attention_mask": np.ones((1, 2), dtype=np.int64)})[0]
    else:
        h = es.run(None, {"input_ids": np.array([[104, 101]], dtype=np.int64)})[0]
    # architecture guard: a wrong-checkpoint export (env-name trap,
    # run-080-thai-v6a-export: the Arabic r8 byt5-BASE shipped under a
    # Thai byt5-small id) passes the shape-only sanity — the encoder
    # width must equal the loaded config's d_model
    if h.shape[-1] != model.config.d_model:
        raise SystemExit(
            f"[sanity] FATAL: encoder d_model {h.shape[-1]} != config "
            f"{model.config.d_model} — the exported graph is not this "
            "checkpoint's architecture")
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
    if METRIC_CER:
        metrics.append({"name": "cer", "value": float(METRIC_CER),
                        "source": f"interscript/interscript-models RESULTS {RUN}"})
    if METRIC_CER_NORM:
        metrics.append({"name": "cer_normalized", "value": float(METRIC_CER_NORM),
                        "source": f"interscript/interscript-models RESULTS {RUN}"})

    meta = {
        "format": "imf-v1", "id": IMF_ID, "task": TASK,
        "source_script": SOURCE, "target": TARGET,
        "tokenizer": "bytes", "opset": 14, "decoder": "plain",
        "precision": shipped if EXPORT_DTYPE == "fp32" else EXPORT_DTYPE, "license": "BSD-3-Clause",
        "trained_from": TRAINED_FROM.replace("{RUN}", RUN),
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

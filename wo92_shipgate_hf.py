# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.5.1",
#   "transformers==4.46.3",
#   "onnx==1.17.0",
#   "onnxruntime==1.20.1",
#   "numpy>=1.26",
# ]
# ///
"""WO92 ship-gate job — run-094 best: export + int8-static + parity + golden.

Precedent: models PRs #447-#449 (the int8 RC publication round) and the
crown plane export (export_plane_onnx_hf.py). One job produces every
ship-gate artifact for the run-094 lane:

  1. torch fp32 reference decode of the 600-row eval600 heb slice
     (single-row greedy K=3 — the shipped decode mode), slice sha
     verified against the committed manifest;
  2. fp32 plane ONNX export (input_ids + plane_ids — the mask-predict
     feedback is the shipped decode contract, WO51 law) at opset 17;
  3. int8 STATIC quantization (quantize_static QDQ + calibration reader
     over slice rows with realistic plane_ids: MASK rows and reference
     prediction rows both fed — the deepseek/WO20u static verdict);
  4. ORT CPU (x86) decode of the same 600 rows;
  5. parity gates: mean per-row CER(candidate, reference) <= 2.0pp AND
     output flip rate <= 1.0% (the published round's gates);
  6. artifacts to isx-training:/wo92-rc/ with SHA256SUMS sidecars:
     plane fp32+int8static onnx, candidate zip (kind: plane,
     precision int8-static, k_passes 3), golden jsonl
     ({"input", "output"} x 600), parity report.

Publish boundary: nothing here ships — no models.yaml entry, no index
bump, no release, NO VERSION NUMBER (owner decision, WO92 ledger).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, "/code")
sys.path.insert(0, "/code/src")

DATA = Path("/train_data")
CKPT = Path("/ckpt")
RC = CKPT / "wo92-rc"
INIT_RUN = "run-094-heb-gold-sft"
BACKBONE = "google/byt5-base"
SLICE_ROWS = 600
SLICE_SHA = "993581cc411b73f3196e6441d0e4f8ae2a01810c2e2e7974a2651b179cdce5c3"
CHANNEL_SHA = "77e9b5dc3a3b9e7d70924e670f7361455b714ec8212e046662ea04187b153ccc"
K_PASSES = 3
CER_GATE_PP = 2.0
FLIP_GATE = 0.01
CALIB_ROWS = 100

import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer

import nikud_planes as NP


def log(*a):
    print("[shipgate]", *a, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cer(a: str, b: str) -> float:
    """Character error rate between two strings (Levenshtein / len(ref))."""
    if not b:
        return 0.0 if not a else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(b)


def load_slice() -> list[dict]:
    path = DATA / "test.jsonl"
    digest = sha256_file(path)
    if digest != CHANNEL_SHA:
        raise RuntimeError(f"eval channel sha mismatch: {digest}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if len(rows) >= SLICE_ROWS:
                break
            if line.strip():
                rows.append(json.loads(line))
    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    slice_sha = hashlib.sha256(blob).hexdigest()
    if slice_sha != SLICE_SHA:
        raise RuntimeError(f"eval600 slice sha mismatch: {slice_sha}")
    log(f"slice: {len(rows)} rows, sha verified against the committed manifest")
    return rows


class PlaneOnnx(nn.Module):
    """The crown export graph: embed + byt5 encoder + per-char head."""

    def __init__(self, encoder, plane_emb, head):
        super().__init__()
        self.encoder = encoder
        self.plane_emb = plane_emb
        self.head = head

    def forward(self, input_ids, plane_ids):
        embeds = self.encoder.embed_tokens(input_ids) + self.plane_emb(plane_ids)
        out = self.encoder(inputs_embeds=embeds)
        return self.head(out.last_hidden_state)


def main() -> None:
    from transformers import AutoModelForSeq2SeqLM

    t0 = time.time()
    RC.mkdir(parents=True, exist_ok=True)

    # ---- inventory + checkpoint ----
    inv_path = CKPT / INIT_RUN / "inventory.json"
    classes = json.loads(inv_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(inv_path.read_bytes()).hexdigest()[:12]
    if digest != "edd595d96d09":
        raise RuntimeError(f"inventory digest mismatch: {digest}")
    n_classes = len(classes)
    MASK_ID = n_classes
    class_to_id = {c: i for i, c in enumerate(classes)}
    sd = torch.load(CKPT / INIT_RUN / "best" / "model.pt",
                    map_location="cpu", weights_only=True)
    log(f"inventory {n_classes} classes (digest {digest}); ckpt loaded")

    tok = AutoTokenizer.from_pretrained(BACKBONE)

    def encode_ids(text: str) -> list[int]:
        return tok(text).input_ids[:-1]  # ByT5 byte table; drop EOS

    def render(classes_ids, skel: str, preds) -> str:
        pos = 0
        char_preds: list[str] = []
        for ch in skel:
            n = len(ch.encode("utf-8"))
            votes = preds[0, pos: pos + n].tolist()
            cid = max(set(votes), key=votes.count)
            char_preds.append(classes[cid] if cid < n_classes else "")
            pos += n
        return NP.render(skel, char_preds)

    # ---- 1. torch fp32 reference decode (shipped decode mode) ----
    backbone = AutoModelForSeq2SeqLM.from_pretrained(BACKBONE)
    encoder = backbone.encoder.cuda().eval()
    d_model = encoder.config.d_model
    plane_emb = nn.Embedding(n_classes + 1, d_model).cuda().eval()
    head = nn.Linear(d_model, n_classes).cuda().eval()
    encoder.load_state_dict(sd["encoder"])
    plane_emb.load_state_dict(sd["plane_emb"])
    head.load_state_dict(sd["head"])

    def torch_predict(ids) -> torch.Tensor:
        # accepts a cpu long tensor of shape [1, seq] or a flat id list
        t = ids.to("cuda") if isinstance(ids, torch.Tensor) else torch.tensor([ids], device="cuda")
        am = torch.ones_like(t)
        plane = torch.full_like(t, MASK_ID)
        preds = None
        for _ in range(K_PASSES):
            with torch.autocast("cuda", torch.bfloat16):
                embeds = encoder.embed_tokens(t) + plane_emb(plane)
                out = encoder(inputs_embeds=embeds, attention_mask=am)
                logits = head(out.last_hidden_state)
            preds = logits.argmax(-1)
            plane = preds
        return preds

    slice_rows = load_slice()
    ref_rows = []
    calib_inputs: list[torch.Tensor] = []   # real encoded ids (cpu)
    calib_preds: list[torch.Tensor] = []    # reference predictions (cpu)
    for i, row in enumerate(slice_rows):
        src = row["src"]
        skel, _ = NP.split_planes(src)
        ids_cpu = torch.tensor([encode_ids(skel)], dtype=torch.long)
        preds = torch_predict(ids_cpu)
        ref_rows.append({"input": src, "output": render(None, skel, preds)})
        if i < CALIB_ROWS:
            calib_inputs.append(ids_cpu)
            calib_preds.append(preds.detach().cpu())
        if (i + 1) % 100 == 0:
            log(f"reference decode {i + 1}/{len(slice_rows)} "
                f"({time.time() - t0:.0f}s)")
    ref_path = RC / "reference-torch-fp32.jsonl"
    ref_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ref_rows),
        encoding="utf-8")
    log(f"reference decode done ({time.time() - t0:.0f}s)")

    # ---- 2. fp32 ONNX export ----
    plane_onnx = PlaneOnnx(encoder.cpu(), plane_emb.cpu(), head.cpu()).eval()
    fp32_path = RC / "plane-fp32.onnx"
    dummy_ids = torch.ones((1, 16), dtype=torch.long)
    dummy_plane = torch.full_like(dummy_ids, MASK_ID)
    torch.onnx.export(
        plane_onnx,
        (dummy_ids, dummy_plane),
        str(fp32_path),
        input_names=["input_ids", "plane_ids"],
        output_names=["class_logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "plane_ids": {0: "batch", 1: "seq"},
                      "class_logits": {0: "batch", 1: "seq"}},
        opset_version=17, dynamo=False,
    )
    log(f"fp32 onnx {fp32_path.stat().st_size / 1e6:.1f} MB")

    # ---- 3. int8 static quantization (calibration with realistic plane_ids) ----
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    # calibration rows: reference predictions (passes 2..K) + MASK rows
    # (pass 1), captured during the reference decode — no GPU work after
    # the modules move to CPU for export.
    calib_preds_all = list(calib_preds)
    calib_inputs_all = list(calib_inputs)
    for j in range(0, len(calib_preds), 2):
        calib_preds_all.append(torch.full_like(calib_preds[j], MASK_ID))
        calib_inputs_all.append(calib_inputs[j])
    log(f"calibration rows: {len(calib_preds_all)} "
        f"(predictions + MASK mix from the first {len(calib_preds)} slice rows)")

    class PlaneCalibReader(CalibrationDataReader):
        def __init__(self):
            self._iter = iter(zip(calib_inputs_all, calib_preds_all))

        def get_next(self):
            for ids, preds in self._iter:
                return {"input_ids": ids.numpy(), "plane_ids": preds.numpy()}
            return None

    int8_path = RC / "plane-int8static.onnx"
    quantize_static(
        str(fp32_path), str(int8_path),
        calibration_data_reader=PlaneCalibReader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    log(f"int8static onnx {int8_path.stat().st_size / 1e6:.1f} MB "
        f"({len(calib_preds_all)} calibration rows)")

    # ---- 4. ORT CPU (x86) decode of the candidate ----
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(str(int8_path), so,
                                providers=["CPUExecutionProvider"])
    log("ort providers: " + str(sess.get_providers()))

    cand_rows = []
    t_onnx = time.time()
    for i, row in enumerate(slice_rows):
        skel, _ = NP.split_planes(row["src"])
        ids = torch.tensor([encode_ids(skel)], dtype=torch.long)
        plane = torch.full_like(ids, MASK_ID)
        preds = None
        for _ in range(K_PASSES):
            logits = sess.run(None, {"input_ids": ids.numpy(),
                                     "plane_ids": plane.numpy()})[0]
            preds = torch.from_numpy(logits).argmax(-1)
            plane = preds
        cand_rows.append({"input": row["src"], "output": render(None, skel, preds)})
        if (i + 1) % 100 == 0:
            log(f"ort decode {i + 1}/{len(slice_rows)} ({time.time() - t_onnx:.0f}s)")

    # ---- 5. parity gates ----
    deltas = [cer(c["output"], r["output"]) for c, r in zip(cand_rows, ref_rows)]
    flips = sum(1 for c, r in zip(cand_rows, ref_rows) if c["output"] != r["output"])
    mean_delta_pp = 100.0 * sum(deltas) / len(deltas)
    max_delta_pp = 100.0 * max(deltas)
    flip_rate = flips / len(cand_rows)
    parity_pass = mean_delta_pp <= CER_GATE_PP and flip_rate <= FLIP_GATE
    report = {
        "lane": "WO92 run-094 ship-gate (heb plane int8static RC)",
        "init": INIT_RUN,
        "slice": {"file": "conformance/wo20/eval600/heb_eval_600.jsonl",
                  "rows": len(slice_rows), "sha256": SLICE_SHA},
        "decode_mode": "single-row greedy K=3 (shipped)",
        "ort_providers": sess.get_providers(),
        "parity": {
            "mean_cer_delta_pp": round(mean_delta_pp, 4),
            "max_cer_delta_pp": round(max_delta_pp, 4),
            "flips": flips,
            "flip_rate": round(flip_rate, 6),
            "samples": len(cand_rows),
            "gates": {"cer_delta_pp_max": CER_GATE_PP, "flip_rate_max": FLIP_GATE},
            "pass": parity_pass,
        },
        "reference": "torch fp32 CUDA bf16-autocast (run-094 best; itself "
                     "verdict-gated at DER 6.6078 on the full 1,864-row set)",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (RC / "parity-report.json").write_text(json.dumps(report, indent=1),
                                           encoding="utf-8")
    log(f"parity: mean_delta={mean_delta_pp:.4f}pp max={max_delta_pp:.4f}pp "
        f"flips={flips}/{len(cand_rows)} pass={parity_pass}")

    # ---- 6. candidate zip + golden + SHA256SUMS ----
    classes_bytes = json.dumps(classes, ensure_ascii=False, indent=1).encode("utf-8")
    onnx_bytes = int8_path.read_bytes()
    meta = (
        "id: heb-vocalized-plane-wo92rc  # RC lane id; final heb-vocalized-plane-{major.minor} = owner decision\n"
        "task: diacritization\n"
        "target: Hebr\n"
        "kind: plane\n"
        "precision: int8-static\n"
        "k_passes: 3\n"
        "decode: single-row-greedy\n"
        "license: BSD-3-Clause\n"
        "trained_from: interscript-train train_hebrew_sft_hf.py (WO92 stage-2 gold SFT)\n"
        "  on run-093-heb-weak-pretrain best (knesset-v6 deduped weak-pretrain + gold);\n"
        "  checkpoint isx-training:/run-094-heb-gold-sft/best/model.pt\n"
        "metrics:\n"
        "- name: der_total_greedy\n"
        "  value: 6.6078\n"
        "  protocol: crown protocol (greedy K=3, seq2seq_der), Nakdimon test 1,864 rows\n"
        "  source: isx-training:/run-094-heb-gold-sft/verdict.json\n"
        "parity:\n"
        f"  samples: {len(cand_rows)}\n"
        f"  cer_delta: {round(mean_delta_pp, 4)}\n"
        f"  flip_rate: {round(flip_rate, 6)}\n"
        "sha256:\n"
        f"  plane-int8static.onnx: {hashlib.sha256(onnx_bytes).hexdigest()}\n"
        f"  classes.json: {hashlib.sha256(classes_bytes).hexdigest()}\n"
    )
    zip_buf = __import__("io").BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("metadata.yaml", meta)
        zf.writestr("classes.json", classes_bytes)
        zf.writestr("plane-int8static.onnx", onnx_bytes)
    zip_path = RC / "heb-vocalized-plane-wo92rc-int8static.zip"
    zip_path.write_bytes(zip_buf.getvalue())

    golden_path = RC / "heb-vocalized-plane-wo92rc-int8-golden.jsonl"
    golden_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cand_rows),
        encoding="utf-8")

    sums = [
        f"{sha256_file(fp32_path)}  plane-fp32.onnx",
        f"{sha256_file(int8_path)}  plane-int8static.onnx",
        f"{sha256_file(zip_path)}  {zip_path.name}",
        f"{sha256_file(golden_path)}  {golden_path.name}",
        f"{sha256_file(ref_path)}  reference-torch-fp32.jsonl",
        f"{hashlib.sha256((RC / 'parity-report.json').read_bytes()).hexdigest()}  parity-report.json",
    ]
    (RC / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")

    summary = {"parity_pass": parity_pass,
               "mean_cer_delta_pp": round(mean_delta_pp, 4),
               "flip_rate": round(flip_rate, 6),
               "samples": len(cand_rows),
               "zip": zip_path.name,
               "zip_bytes": zip_path.stat().st_size,
               "elapsed_s": round(time.time() - t0)}
    (RC / "SHIPGATE_DONE").write_text(json.dumps(summary, indent=1) + "\n",
                                      encoding="utf-8")
    log("summary " + json.dumps(summary))


if __name__ == "__main__":
    main()

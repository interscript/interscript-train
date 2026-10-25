"""B3/WO64 stage 3 — the SadeedDiac-25 protocol eval for the r8 stage-2
student (run-080-r8-stage2/eps[-1]).

Loads the SadeedDiac-25 test paragraphs, decodes with the stage-2
checkpoint (single-row greedy, the 512-step contract cap), writes the
predictions JSONL, and scores DER-CE via sadeedbench (the protocol
package). Env: E_RUN (default run-080-r8-stage2), E_N (max paragraphs,
default all).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CKPT = Path("/ckpt") / os.environ.get("E_RUN", "run-080-r8-stage2")
E_N = int(os.environ.get("E_N", "0")) or None


def main() -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from sadeedbench.scoring import score_predictions

    eps = sorted((p for p in CKPT.glob("oepoch-*")
                  if p.is_dir()), key=lambda p: int(p.name.split("-")[1]))
    init = eps[-1] if eps else CKPT / "best"
    print(f"[eval] checkpoint: {init}", flush=True)

    ds = load_dataset("Misraj/SadeedDiac-25")["train"]
    inputs = ds["input"]
    gts = ds["output"]
    if E_N:
        inputs, gts = inputs[:E_N], gts[:E_N]
    print(f"[eval] paragraphs: {len(inputs)}", flush=True)

    tok = AutoTokenizer.from_pretrained(init)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        init, torch_dtype=torch.bfloat16).eval().cuda()

    preds_path = CKPT / "sadeed-preds.jsonl"
    preds = []
    BS = 16
    with preds_path.open("w", encoding="utf-8") as f, torch.no_grad():
        for i in range(0, len(inputs), BS):
            batch = inputs[i:i + BS]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to("cuda")
            gen = model.generate(**enc, max_new_tokens=512, num_beams=1)
            for text, p in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
                preds.append(p)
                f.write(json.dumps({"input": text, "pred": p},
                                   ensure_ascii=False) + "\n")
            f.flush()
            if (i // BS) % 25 == 0:
                print(f"[eval] {i + BS}/{len(inputs)}", flush=True)

    result = score_predictions(preds, gts)
    print(f"[eval] DER: {result}", flush=True)
    name = init.name if init.is_dir() else "best"
    (CKPT / f"sadeed-verdict-{name}.json").write_text(
        json.dumps({"checkpoint": str(init), "verdict": result},
                   ensure_ascii=False, indent=2))
    print("[eval] verdict written", flush=True)


if __name__ == "__main__":
    main()

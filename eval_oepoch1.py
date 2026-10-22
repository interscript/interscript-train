"""WO39 — eval-only for the saved oepoch-1 (the training crashed at
the eval import: pandas was not in the spine's default deps; the
checkpoint is fine, so only the surfaces need to run).

Usage (a100-large, mounts isx-training:/ckpt:rw + arabic-r8:/train_data:ro):
    python /ckpt/run-065-oracle-distill/eval_oepoch1.py
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/ckpt/run-065-oracle-distill")

import continue_oracle_train_hf as C  # noqa: E402  (helpers; main is guarded)

CKPT = Path("/ckpt/run-065-oracle-distill/oepoch-1")
DATA = C.DATA

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForSeq2SeqLM.from_pretrained(CKPT).cuda()
model.config.use_cache = True
model.eval()

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

table = pq.read_table(DATA / "sadeeddiac-25.parquet")
inputs = [C.DIACRITICS_RE.sub("", t) for t in table.column("input").to_pylist()]
outputs = table.column("output").to_pylist()
windows, counts = [], []
for text in inputs:
    ws = C.split_windows(text)
    counts.append(len(ws))
    windows.extend(ws)
preds = []
with torch.no_grad():
    for i in range(0, len(windows), 32):
        enc = tok(windows[i:i + 32], return_tensors="pt", padding=True,
                  truncation=True, max_length=C.WINDOW).to("cuda")
        gen = model.generate(**enc, max_new_tokens=C.WINDOW * 2, num_beams=1)
        preds.extend(tok.batch_decode(gen, skip_special_tokens=True))
k = 0
paragraphs = []
for text, c in zip(inputs, counts):
    paragraphs.append(C.project_haraqat(" ".join(preds[k:k + c]), text))
    k += c
csv_path = Path("/tmp/sadeed_oracle.csv")
pd.DataFrame({"gt": outputs, "pred": paragraphs}).to_csv(
    csv_path, index=False, header=False)
from sadeed_evaluator import ArabicDiacritizationEvaluator as E  # noqa: E402

print("===== SadeedDiac-25 (default protocol) =====", flush=True)
E.report_errors_on_csv_file(str(csv_path), ground_truth_column_index=0,
                            predicted_column_index=1, has_header=False,
                            gt_missing_diacritic_is_error=False)

import eval_wikinews_multiref as WN  # noqa: E402

bench = [l for l in (DATA / "wikinews/WikiNews_2024_Multi_Ref.txt.diac")
         .read_text(encoding="utf-8").splitlines() if l.strip()]
winputs = [C.DIACRITICS_RE.sub("", l) for l in bench]
wwindows, wcounts = [], []
for text in winputs:
    ws = C.split_windows(text)
    wcounts.append(len(ws))
    wwindows.extend(ws)
wpreds = []
with torch.no_grad():
    for i in range(0, len(wwindows), 32):
        enc = tok(wwindows[i:i + 32], return_tensors="pt", padding=True,
                  truncation=True, max_length=C.WINDOW).to("cuda")
        gen = model.generate(**enc, max_new_tokens=C.WINDOW * 2, num_beams=1)
        wpreds.extend(tok.batch_decode(gen, skip_special_tokens=True))
k = 0
wparas = []
for text, c in zip(winputs, wcounts):
    wparas.append(C.project_haraqat(" ".join(wpreds[k:k + c]), text))
    k += c
wscore = WN.score(wparas, bench, skip_last=False)
print(json.dumps({"run": "oepoch-1", "surface": "wikinews-2024-multiref",
                  **wscore}, ensure_ascii=False), flush=True)
fa = [" ".join(w.split("/", 1)[0] for w in p.split()) for p in wparas]
fascore = WN.score(fa, bench, skip_last=False)
verdict = {"run": "oepoch-1", "surface": "wikinews-firstalt",
           "gate": "wer < 3.5607", "teacher": 3.5607, **fascore}
print(json.dumps(verdict, ensure_ascii=False), flush=True)
out = CKPT.parent / "oracle_verdict.json"
out.write_text(json.dumps({"wikinews_raw": wscore,
                           "wikinews_firstalt": verdict},
                          ensure_ascii=False, indent=1))
try:
    from huggingface_hub import HfFileSystem
    import os

    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    dst = f"buckets/Interscript/isx-training/{out.relative_to(Path('/ckpt'))}"
    with open(out, "rb") as f, fs.open(dst, "wb") as g:
        g.write(f.read())
    print("synced", fs.exists(dst), flush=True)
except Exception as e:  # noqa: BLE001
    print(f"sync failed: {e}", flush=True)

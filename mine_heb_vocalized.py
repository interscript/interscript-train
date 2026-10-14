"""WO48 — mine fully-vocalized Hebrew Wikipedia into nikud pairs.

Streams wikimedia/wikipedia (20231101.he) in-job, keeps paragraphs
whose Hebrew letters carry >= VOC_FRAC niqqud (and >= MIN_LETTERS
letters), splits to sentences, emits {src: unpointed, tgt: pointed}
pairs. Deterministic; resumable via an output line count is NOT
supported (streaming) — cap sets the run size.

Env: M_RUN (run-071-heb-vocalized-mine), M_CAP (2000000),
    VOC_FRAC (0.55), MIN_LETTERS (40).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

OUT = Path("/ckpt") / os.environ.get("M_RUN", "run-071-heb-vocalized-mine")
CAP = int(os.environ.get("M_CAP", "2000000"))
VOC_FRAC = float(os.environ.get("VOC_FRAC", "0.55"))
MIN_LETTERS = int(os.environ.get("MIN_LETTERS", "40"))

HEBREW_LETTER = re.compile("[א-ת]")
NIQQUD = re.compile("[֑-ׇ]")
SENT_SPLIT = re.compile(r"(?<=[.!?־])\s+")
VAL_FRAC = 0.02


def paragraph_vocalized(text: str) -> bool:
    letters = HEBREW_LETTER.findall(text)
    if len(letters) < MIN_LETTERS:
        return False
    # a letter counts as marked when a niqqud char immediately follows it
    marked_n = 0
    for m in HEBREW_LETTER.finditer(text):
        if m.end() < len(text) and NIQQUD.match(text[m.end()]):
            marked_n += 1
    return marked_n / len(letters) >= VOC_FRAC


def main() -> None:
    from datasets import load_dataset

    OUT.mkdir(parents=True, exist_ok=True)
    f_train = (OUT / "heb-vocalized-train.jsonl").open("w", encoding="utf-8")
    f_val = (OUT / "heb-vocalized-val.jsonl").open("w", encoding="utf-8")
    n = 0
    n_paras = 0
    ds = load_dataset("wikimedia/wikipedia", "20231101.he",
                      split="train", streaming=True)
    for art in ds:
        if n >= CAP:
            break
        for para in art["text"].split("\n"):
            para = para.strip()
            if not paragraph_vocalized(para):
                continue
            n_paras += 1
            for sent in SENT_SPLIT.split(para):
                sent = sent.strip()
                letters = HEBREW_LETTER.findall(sent)
                if len(letters) < 8:
                    continue
                tgt = sent
                src = NIQQUD.sub("", sent)
                if src == tgt:
                    continue
                n += 1
                row = json.dumps({"src": src, "tgt": tgt},
                                 ensure_ascii=False)
                if n % int(1 / VAL_FRAC) == 0:
                    f_val.write(row + "\n")
                else:
                    f_train.write(row + "\n")
                if n >= CAP:
                    break
        if n % 50000 < 40:
            print(f"[mine] articles='{art['title']}' pairs={n} "
                  f"vocalized_paras={n_paras}", flush=True)
    f_train.close()
    f_val.close()
    stats = {"pairs": n, "vocalized_paragraphs": n_paras,
             "voc_frac": VOC_FRAC, "cap": CAP}
    print(json.dumps(stats), flush=True)
    (OUT / "mine_stats.json").write_text(json.dumps(stats, indent=1))
    (OUT / "MINE_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

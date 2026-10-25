"""WO31 stage 2 — relabel FLEURS with the TUNED ASR + v1 student.

The teacher fix from stage 1 (run-038): labels now come from AUDIO
(tuned CTC decode), not from the universal model that heard Hebrew at
PER 0.395. Student: byt5-small text -> IPA (v0 recipe, byte-correct
collate). Gate vs heb-g2p-benchmark: CER < 0.2393 (our rules layer).
NO LLM labels anywhere. Run on an apt-enabled image (espeak-ng for
edge cases only).

    python /ckpt/run-039-heb-ipa-v1/heb_ipa_v1_hf.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import tarfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download

CKPT = Path("/ckpt") / os.environ.get("V1_RUN", "run-039-heb-ipa-v1")
POINTED = os.environ.get("V1_POINTED", "")  # text->pointed map for label filtering
FILTER_PER = float(os.environ.get("V1_FILTER_PER", "0"))  # 0 = off
TEACHER = Path(os.environ.get("V1_TEACHER", "/ckpt/run-038-heb-asr-ft/best"))
GT = Path("/ckpt/run-032-heb-ipa-v0/gt.tsv")
EPOCHS = int(os.environ.get("V1_EPOCHS", "60"))
BS = int(os.environ.get("V1_BS", "16"))
STRESS = "ˈˌ.-"


def norm_target(ph: str) -> str:
    for ch in STRESS:
        ph = ph.replace(ch, "")
    return " ".join(ph.split())


def build_pairs() -> list[dict]:
    pairs_path = CKPT / "fleurst-ipa-v1.jsonl"
    if pairs_path.exists():
        return [json.loads(l) for l in pairs_path.read_text(
            encoding="utf-8").splitlines() if l.strip()]

    from transformers import AutoModelForCTC, AutoProcessor

    proc = AutoProcessor.from_pretrained(TEACHER)
    tok = proc.tokenizer
    tok.do_phonemize = False
    asr = AutoModelForCTC.from_pretrained(TEACHER).eval().cuda()
    unk = tok.unk_token_id

    rows = []
    for split in ("train", "dev", "test"):
        tsv = hf_hub_download("google/fleurs", f"data/he_il/{split}.tsv",
                              repo_type="dataset")
        with open(tsv, encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) >= 3:
                    rows.append((split, row))
    wanted: dict[str, dict[str, str]] = {}
    for split, row in rows:
        name = row[1] if row[1].endswith(".wav") else row[1] + ".wav"
        wanted.setdefault(split, {})[name] = row[2].strip()

    pointed = {}
    if POINTED:
        for line in Path(POINTED).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                pointed[r["text"]] = r["pointed"]
        tok.init_backend("he")

    def ref_phones(text: str) -> str:
        pt = pointed.get(text, text)
        tok.do_phonemize = True   # espeak(pointed text): vowels + stress
        ids = tok(pt, add_special_tokens=False).input_ids
        tok.do_phonemize = False  # back to decode mode
        toks = tok.convert_ids_to_tokens([i for i in ids if i != -100])
        return norm_target(" ".join(toks))

    def ncer(a: str, b: str) -> float:
        na, nb = a.replace(" ", ""), b.replace(" ", "")
        if not nb:
            return 1.0
        prev = list(range(len(nb) + 1))
        for i, ca in enumerate(na, 1):
            cur = [i]
            for j, cb in enumerate(nb, 1):
                cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb)))
            prev = cur
        return prev[-1] / len(nb)

    pairs = []
    t0 = time.time()
    with pairs_path.open("w", encoding="utf-8") as out:
        for split in ("train", "dev", "test"):
            tar = hf_hub_download("google/fleurs",
                                  f"data/he_il/audio/{split}.tar.gz",
                                  repo_type="dataset")
            with tarfile.open(tar, "r:gz") as tf:
                for m in tf.getmembers():
                    base = m.name.split("/")[-1]
                    if base not in wanted[split]:
                        continue
                    f = tf.extractfile(m)
                    if not f:
                        continue
                    text = wanted[split][base]
                    if not text:
                        continue
                    speech, sr = sf.read(io.BytesIO(f.read()))
                    if speech.ndim > 1:
                        speech = speech.mean(axis=1)
                    if len(speech) < 16000 or sr != 16000:
                        continue
                    feats = proc.feature_extractor(
                        speech.astype(np.float32), sampling_rate=16000,
                        return_tensors="pt")
                    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                        logits = asr(feats.input_values.cuda()).logits
                    ids = logits.argmax(-1)[0].tolist()
                    kept, seen = [], None
                    for i in ids:
                        if i != unk and i != seen:
                            kept.append(i)
                        seen = i
                    ph = norm_target(tok.decode(kept) if kept else "")
                    if len(ph) < 8:
                        continue
                    if FILTER_PER > 0:
                        # pseudo-label quality gate: audio-decode must
                        # agree with espeak(pointed text) — WO06 pattern
                        err = ncer(ph, ref_phones(text))
                        if err > FILTER_PER:
                            continue
                    pairs.append({"text": text, "ipa": ph, "split": split})
                    out.write(json.dumps(pairs[-1], ensure_ascii=False) + "\n")
    print(f"[label] {len(pairs)} pairs in {time.time()-t0:.0f}s", flush=True)
    return pairs


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs()
    train = [p for p in pairs if p["split"] == "train"]
    heldout = [p for p in pairs if p["split"] != "train"][:300]
    print(f"[data] train={len(train)} heldout={len(heldout)}", flush=True)

    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)

    tok = AutoTokenizer.from_pretrained("google/byt5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small").cuda().train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    class DS(Dataset):
        def __len__(self):
            return len(train)

        def __getitem__(self, i):
            return (train[i]["text"], train[i]["ipa"])

    def collate(b):
        # ByT5 tokens are BYTES: Hebrew chars are multi-byte — pad to
        # tokenized lengths, never char lengths
        es = [tok(t).input_ids[:-1] for t, _ in b]
        ds_ = [tok(p).input_ids for _, p in b]
        mx = max(len(e) for e in es)
        md = max(len(d) for d in ds_)
        ids = torch.zeros((len(b), mx), dtype=torch.long)
        am = torch.zeros((len(b), mx), dtype=torch.long)
        lb = torch.full((len(b), md), -100, dtype=torch.long)
        for i, (e, d) in enumerate(zip(es, ds_)):
            ids[i, :len(e)] = torch.tensor(e)
            am[i, :len(e)] = 1
            lb[i, :len(d)] = torch.tensor(d)
        return ids, am, lb

    loader = DataLoader(DS(), batch_size=BS, shuffle=True, collate_fn=collate,
                        num_workers=2, drop_last=True)
    total_steps = len(loader) * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, 100, total_steps)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    step = 0
    t0 = time.time()
    for _ep in range(EPOCHS):
        for ids, am, lb in loader:
            ids, am, lb = ids.cuda(), am.cuda(), lb.cuda()
            with torch.autocast("cuda", torch.bfloat16):
                out = model(input_ids=ids, attention_mask=am, labels=lb)
            if not torch.isfinite(out.loss):
                print(f"[nan] step {step}", flush=True)
                opt.zero_grad()
                continue
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f"[train] {step}/{total_steps} loss={float(out.loss):.3f} "
                      f"({step/(time.time()-t0):.1f} it/s)", flush=True)

    model.config.use_cache = True
    model.eval()
    best = CKPT / "best"
    best.mkdir(exist_ok=True)
    model.save_pretrained(best)
    tok.save_pretrained(best)

    # ---- gate: phonikud heb-g2p-benchmark (their CER, their conventions) ----
    gt = []
    for line in GT.read_text(encoding="utf-8").splitlines():
        s, _, p = line.partition("\t")
        if s and p:
            gt.append((s, p))
    print(f"[gate] gt sentences={len(gt)}", flush=True)

    def lev(a: str, b: str) -> int:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb)))
            prev = cur
        return prev[-1]

    preds = []
    with torch.no_grad():
        for i in range(0, len(gt), 16):
            batch = [s for s, _ in gt[i:i+16]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to("cuda")
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=256, num_beams=1,
                                    no_repeat_ngram_size=3)
            preds.extend(tok.batch_decode(gen, skip_special_tokens=True))

    def _norm_ipa(s: str) -> str:
        for ch in STRESS:
            s = s.replace(ch, "")
        return s.replace(" ", "")

    cers, cers_ns, cers_norm = [], [], []
    for (s, gold), pred in zip(gt, preds):
        g = gold.replace("ˈ", "")
        p = pred.replace("ˈ", "")
        cers.append(lev(pred, gold) / max(1, len(gold)))
        cers_ns.append(lev(p, g) / max(1, len(g)))
        ng, np_ = _norm_ipa(gold), _norm_ipa(pred)
        cers_norm.append(lev(np_, ng) / max(1, len(ng)))
    cer = sum(cers) / len(cers)
    cer_ns = sum(cers_ns) / len(cers_ns)
    verdict = {"model": "heb-ipa-v1", "teacher": str(TEACHER),
               "cer": round(cer, 4), "cer_no_stress": round(cer_ns, 4),
               "cer_normalized": round(sum(cers_norm) / len(cers_norm), 4),
               "gate": "cer < 0.2393 (rules layer)", "rules_baseline": 0.2393,
               "v0_negative": 1.18,
               "board": {"renikud": 0.0244, "phonikud": 0.0495, "nakdimon": 0.1102}}
    (CKPT / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict), flush=True)
    for (s, gold), pred in list(zip(gt, preds))[:3]:
        print(f"  src: {s[:40]}\n  pred: {pred[:60]}\n  gold: {gold[:60]}", flush=True)
    (CKPT / "V1_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

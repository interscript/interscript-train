"""WO42 — HE label-filter arm: self-consistency decode over the labels.

The student converged at nCER 0.4462 vs the 0.2393 gate — the
undertrained hypothesis is refuted; the residual is label-side
(spontaneous-speech audio the 13.8%-PER teacher faithfully
transcribed, music/hesitation segments included). This job decodes
every labeled row TWICE more under ±10% time-stretch (the CTC decode
diverges on noisy/unstable audio), keeps rows whose stretched decodes
agree with the stored label (phone error < ACCEPT), and applies a
ipa/text-ratio band guard. Output: labels-filtered.jsonl + stats.

Env: F_RUN (default run-067-label-filter), F_ACCEPT (0.15).

Usage (l4x1): python /ckpt/run-067-label-filter/filter_heb_labels.py
"""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import numpy as np
import torch

LABELS = Path("/ckpt/run-062-ivrit-student/labels.jsonl")
WDS = Path("/ckpt/run-050-ivrit-wds")
OUT = Path("/ckpt") / os.environ.get("F_RUN", "run-067-label-filter")
TEACHER = Path("/ckpt/run-061-ivrit-teacher-v3/best")
ACCEPT = float(os.environ.get("F_ACCEPT", "0.15"))
STRESS = "ˈˌ.-"


def norm_phones(ph: str) -> list[str]:
    for ch in STRESS:
        ph = ph.replace(ch, "")
    return ph.split()


def phone_lev(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def consistent(pred_phones: list[str], label_phones: list[str],
               accept: float = ACCEPT) -> bool:
    if not pred_phones or not label_phones:
        return False
    return phone_lev(pred_phones, label_phones) / len(label_phones) < accept


def ratio_band(rows: list[dict], lo_q: float = 0.01,
               hi_q: float = 0.99) -> tuple[float, float]:
    rs = sorted(len(r["ipa"]) / max(1, len(r["text"])) for r in rows)
    lo_i, hi_i = int(len(rs) * lo_q), int(len(rs) * hi_q)
    return rs[max(0, lo_i)], rs[min(len(rs) - 1, hi_i)]


def ratio_ok(ipa: str, text: str, lo: float, hi: float) -> bool:
    return lo <= len(ipa) / max(1, len(text)) <= hi


def stretch_factors() -> tuple[float, float]:
    return (0.9, 1.1)


def main() -> None:
    from transformers import AutoModelForCTC, AutoProcessor

    rows = [json.loads(l) for l in LABELS.read_text(
        encoding="utf-8").splitlines() if l.strip()]
    by_uuid = {r["uuid"]: r for r in rows}
    print(f"[labels] rows={len(rows)}", flush=True)
    lo, hi = ratio_band(rows)
    print(f"[ratio] band=[{lo:.3f}, {hi:.3f}]", flush=True)

    proc = AutoProcessor.from_pretrained(TEACHER)
    tok = proc.tokenizer
    model = AutoModelForCTC.from_pretrained(TEACHER).eval().cuda()
    unk = tok.unk_token_id

    @torch.no_grad()
    def decode(wav: np.ndarray) -> list[str]:
        feats = proc.feature_extractor(wav, sampling_rate=16000,
                                       return_tensors="pt")
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(feats.input_values.cuda()).logits
        ids = logits.argmax(-1)[0].tolist()
        kept, seen = [], None
        for i in ids:
            if i != unk and i != seen:
                kept.append(i)
            seen = i
        return norm_phones(tok.decode(kept) if kept else "")

    def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        import torchaudio  # noqa: PLC0415

        t = torch.from_numpy(wav)
        return torchaudio.functional.resample(t, orig_sr, target_sr).numpy()

    f_lo, f_hi = stretch_factors()
    smoke = int(os.environ.get("SMOKE_SHARDS", "0"))
    kept_rows, stats = [], {"total": 0, "ratio_dropped": 0,
                            "inconsistent": 0, "kept": 0}
    out_path = OUT / "labels-filtered.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = __import__("time").time()

    DEC_BS = int(os.environ.get("DEC_BS", "16"))

    @torch.no_grad()
    def decode_batch(wavs: list[np.ndarray]) -> list[list[str]]:
        feats = proc.feature_extractor(wavs, sampling_rate=16000,
                                       return_tensors="pt", padding=True)
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(feats.input_values.cuda()).logits
        out: list[list[str]] = []
        for row_ids in logits.argmax(-1).tolist():
            kept, seen = [], None
            for i in row_ids:
                if i != unk and i != seen:
                    kept.append(i)
                seen = i
            out.append(norm_phones(tok.decode(kept) if kept else ""))
        return out

    with out_path.open("w", encoding="utf-8") as out:
        shards = sorted(WDS.glob("shard-*.tar"))
        if smoke:
            shards = shards[:smoke]
        for si, name in enumerate(shards):
            t_shard = __import__("time").time()
            with tarfile.open(name) as tf:
                members = {m.name: m for m in tf.getmembers()}
                # pass 1: collect (row, wav) for ratio-passing rows
                todo = []
                for m in members.values():
                    if not m.name.endswith(".json"):
                        continue
                    uuid = m.name[:-5]
                    row = by_uuid.get(uuid)
                    if row is None:
                        continue
                    stats["total"] += 1
                    if not ratio_ok(row["ipa"], row["text"], lo, hi):
                        stats["ratio_dropped"] += 1
                        continue
                    wav_m = members.get(uuid + ".wav")
                    if wav_m is None:
                        continue
                    meta = json.load(tf.extractfile(m))
                    sr = int(meta["sr"])
                    wav = (np.frombuffer(tf.extractfile(wav_m).read(),
                                         dtype=np.int16)
                           .astype(np.float32) / 32768.0)
                    if wav.ndim > 1:
                        wav = wav.mean(1)
                    w16 = resample(wav, sr, 16000) if sr != 16000 else wav
                    todo.append((row, w16))
                # pass 2: batched stretch-decodes
                for b in range(0, len(todo), DEC_BS):
                    batch = todo[b:b + DEC_BS]
                    votes = []
                    for factor in (f_lo, f_hi):
                        stretched = [resample(w, 16000, int(16000 / factor))
                                     for _, w in batch]
                        votes.append(decode_batch(stretched))
                    for j, (row, _) in enumerate(batch):
                        label_phones = norm_phones(row["ipa"])
                        ok = all(consistent(votes[k][j], label_phones)
                                 for k in (0, 1))
                        if ok:
                            kept_rows.append(row)
                            out.write(json.dumps(row, ensure_ascii=False)
                                      + "\n")
                            stats["kept"] += 1
                        else:
                            stats["inconsistent"] += 1
            print(f"[filter] shard {si + 1} done in "
                  f"{__import__('time').time() - t_shard:.0f}s "
                  f"total={stats['total']} kept={stats['kept']}", flush=True)

    stats["band"] = [round(lo, 3), round(hi, 3)]
    print(json.dumps(stats), flush=True)
    (OUT / "filter_stats.json").write_text(json.dumps(stats, indent=1))
    (OUT / "FILTER_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=1.1",
# ]
# ///
"""WO20u M1-6ep orchestrator (detached, single-seed, sequential).

Keeps the 6-epoch bridge arm alive across its 4h windows and runs the
confirmation scorer at the end. Minimal single-thread redesign of the
M1 orchestrator after its threads wedged silently: one loop, one seed,
heartbeat every poll, every hf call wrapped in retry-once, every poll
body guarded so no exception can kill the run silently. All state on
the isx-training bucket + live `hf jobs ps`; idempotent and resumable.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time

BUCKET = "hf://buckets/Interscript/isx-training"
BUCKET_NS = "Interscript/isx-training"
ARABIC = "hf://datasets/Interscript/arabic-r8"
CODE_BKT = os.environ.get("WO20U_CODE_BKT", "")
CODE_DIR = "/code-src"
TRAINER = f"{CODE_DIR}/train_wo20_tiny_hf.py"
SCORER = f"{CODE_DIR}/wo20_seed_confirm.py"
RUN = os.environ.get("WO20U_RUN", "wo20-tiny-g2w-6ep")
SEED = os.environ.get("WO20U_SEED", "2")
EPOCHS = os.environ.get("WO20U_EPOCHS", "6")
OUT_DIR = os.environ.get("WO20U_OUT_DIR", "wo20u-m1-6ep")
BIG_FLAVORS = {"a100-large", "a100x4", "a100x8",
               "h200", "h200x2", "h200x4", "h200x8"}
MAX_BIG = 2
MAX_ROUNDS = 16
POLL_S = 240


def log(*a):
    print(f"[orch6 {time.strftime('%m-%d %H:%M:%S')}]", *a, flush=True)


def sh(args, timeout=900):
    last = None
    for attempt in (1, 2):
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired as e:
            last = e
            log(f"timeout on {args[0]} {args[1] if len(args) > 1 else ''} "
                f"(attempt {attempt})")
            time.sleep(20)
    raise RuntimeError(f"hf call kept timing out: {args[:3]}") from last


def jobs_now():
    r = sh(["hf", "jobs", "ps", "--json"])
    if r.returncode != 0:
        log("ps failed:", (r.stderr or "").strip()[:200])
        return []
    return json.loads(r.stdout)


def big_running():
    n = 0
    who = []
    for r in jobs_now():
        if r.get("status") in ("RUNNING", "SCHEDULING") \
                and r.get("flavor") in BIG_FLAVORS:
            n += 1
            who.append((r.get("labels", {}).get("name"), r.get("flavor")))
    return n, who


def job_stage(jid):
    r = sh(["hf", "jobs", "inspect", jid, "--json"])
    if r.returncode != 0:
        return "UNKNOWN"
    rows = json.loads(r.stdout)
    rows = rows if isinstance(rows, list) else [rows]
    return rows[0].get("status") or "UNKNOWN"


def wait_terminal(jid):
    while True:
        st = job_stage(jid)
        log(f"heartbeat: job {jid[:8]} stage={st}")
        if st in ("COMPLETED", "ERROR", "CANCELED", "DELETED"):
            return st
        time.sleep(POLL_S)


def bucket_get(remote: str, local: str) -> bool:
    r = sh(["hf", "buckets", "cp",
            f"hf://buckets/{BUCKET_NS}/{remote}", local])
    return r.returncode == 0


def confirm_done():
    return bucket_get(f"{OUT_DIR}/{RUN}.confirm.json",
                      f"/tmp/confirm-{RUN}.json")


def final_eval_done():
    return bucket_get(f"{RUN}/final_eval.json",
                      f"/tmp/finaleval-{RUN}.json")


def preds_complete():
    local = f"/tmp/preds-{RUN}.jsonl"
    if not bucket_get(f"{RUN}/student_preds.jsonl", local):
        return False
    try:
        with open(local, "rb") as f:
            return sum(1 for _ in f) == 1200
    except OSError:
        return False


def launch(script, name, envs, flavor):
    args = ["hf", "jobs", "uv", "run", "--flavor", flavor, "-d",
            "--timeout", "4h", "--attempts", "3",
            "--name", name, "--label", "campaign=wo20u-6ep",
            "--secrets", "HF_TOKEN",
            "-v", f"{ARABIC}:/train_data:ro",
            "-v", f"{BUCKET}:/ckpt:rw"]
    if CODE_BKT:
        args += ["-v", f"hf://buckets/{BUCKET_NS}/{CODE_BKT}:/code:ro"]
    for e in envs:
        args += ["-e", e]
    args.append(script)
    r = sh(args)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        log(f"launch {name} FAILED: {out.strip()[:400]}")
        return None
    m = re.search(r"\b[0-9a-f]{24,}\b", out)
    jid = m.group(0) if m else None
    log(f"launched {name} ({flavor}) -> {jid}")
    return jid


def main():
    t0 = time.time()
    for round_no in range(1, MAX_ROUNDS + 1):
        if confirm_done():
            break
        if not final_eval_done():
            nbig, who = big_running()
            log(f"r{round_no}: big-GPU in use {nbig} {who} "
                f"(cap {MAX_BIG})")
            while nbig >= MAX_BIG:
                time.sleep(180)
                nbig, who = big_running()
            jid = launch(TRAINER, f"wo20u-6ep-distill",
                         [f"WO20_STAGE=distill", f"WO20_RUN={RUN}",
                          f"WO20_SEED={SEED}", f"WO20_EPOCHS={EPOCHS}"],
                         "a100-large")
            if jid is None:
                time.sleep(180)
                continue
            wait_terminal(jid)
        if confirm_done():
            break
        if preds_complete():
            jid = launch(SCORER, "wo20u-6ep-confirm",
                         [f"WO20_RUN={RUN}", f"WO20U_OUT_DIR={OUT_DIR}"],
                         "cpu-basic")
            if jid is not None:
                wait_terminal(jid)
        log(f"r{round_no}: looping "
            f"(final_eval={'yes' if final_eval_done() else 'no'}, "
            f"preds={'yes' if preds_complete() else 'no'})")
    if confirm_done():
        result = json.load(open(f"/tmp/confirm-{RUN}.json"))
        log(f"CONFIRM: {json.dumps(result)}")
        print("WO20U-6EP-SUMMARY " + json.dumps(result, indent=1))
    else:
        log(f"INCOMPLETE after {MAX_ROUNDS} rounds "
            f"({(time.time() - t0) / 3600:.1f}h)")
        print("WO20U-6EP-SUMMARY {\"verdict\": \"INCOMPLETE\"}")


if __name__ == "__main__":
    main()
